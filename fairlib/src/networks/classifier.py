import logging
from typing import Dict, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward, LayerNorm
from allennlp.nn import util
from sysrev.modelling.allennlp.util import INV_MAPPING
from transformers import BertModel

from .augmentation_layer import Augmentation_layer
from .utils import BaseModel
from ..dataloaders.encoder import get_vocab


class Encoders(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 feedforward_params,
                 embedder: TextFieldEmbedder,
                 cat_feats_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 encoder_cat_feats: Seq2VecEncoder,
                 metrics: Set,
                 feat_type: Dict[str, bool] = None,
                 ablated_feat: str = None,
                 ablated_feat_type: str = None,
                 reduced_output_bert_dim: int = None):
        super().__init__(vocab)
        self.feat_type = feat_type or {"num": True}
        self.ablated_feat = ablated_feat
        self.ablated_feat_type = ablated_feat_type  # num, cat, text

        self.txt_dim = encoder.get_output_dim()
        self.reduced_output_bert_dim = reduced_output_bert_dim

        encoder_out_dim = 0
        if self.feat_type["text"]:
            self.embedder = embedder
            self.encoder = encoder
            encoder_out_dim = encoder.get_output_dim() * (
                    4 - (1 if self.ablated_feat is not None and self.ablated_feat_type == "text" else 0))
            if self.reduced_output_bert_dim is not None:
                self.reduce_dim = torch.nn.Linear(encoder_out_dim, self.reduced_output_bert_dim)
                encoder_out_dim = self.reduced_output_bert_dim
                self.nonlinearity = torch.tanh
            else:
                self.reduce_dim = lambda x: x
                self.nonlinearity = lambda x: x

        cat_encoder_out_dim = 0
        if self.feat_type["cat"]:
            self.cat_feats_embedder = cat_feats_embedder
            self.encoder_cat_feats = encoder_cat_feats
            self.reduce_dim = lambda x: x
            self.nonlinearity = lambda x: x
            cat_encoder_out_dim = encoder_cat_feats.get_output_dim() * 3  # we keep an empty placeholder feature for "topics"

        feedforward_out_dim = 0
        if self.feat_type["num"]:
            self.feedforward = FeedForward.from_params(feedforward_params)
            feedforward_out_dim = self.feedforward.get_output_dim()
        self.num_labels = vocab.get_vocab_size("labels")
        if "labels" in vocab._index_to_token:
            self.index_to_label = vocab._index_to_token["labels"]
        else:
            self.index_to_label = INV_MAPPING
            # self.index_to_label = None

        self.classifier_input_dim = feedforward_out_dim + encoder_out_dim + cat_encoder_out_dim
        self.layer_norm = LayerNorm(self.classifier_input_dim)

    def encode_text_list(self, text_list, concatenate=True, categorical=False):
        if categorical:
            embedder = self.cat_feats_embedder
            encoder = self.encoder_cat_feats
        else:
            embedder = self.embedder
            encoder = self.encoder

        embedded_text = embedder(text_list, num_wrapping_dims=1)
        mask = util.get_text_field_mask(text_list, num_wrapping_dims=1)
        len_text_list = embedded_text.shape[1]
        enc_txt = []
        for i in range(len_text_list):
            enc_txt.append(encoder(embedded_text[:, i, :, :], mask[:, i, :]))
        if concatenate:
            enc_txt = torch.cat(enc_txt, dim=1)
        else:
            enc_txt = torch.stack(enc_txt)

        enc_txt = self.nonlinearity(self.reduce_dim(enc_txt))

        return enc_txt

    def encode(self, feat, text_list, cat_feats_list):
        inputs = []
        if self.feat_type["text"]:
            # text encoding:
            # Shape: (batch_size, text_list_len, num_tokens, embedding_dim)
            enc_txt = self.encode_text_list(text_list)
            inputs.append(enc_txt)

        if self.feat_type["cat"]:
            # categorical feature encoding:
            embedded_cat_feat = self.cat_feats_embedder(cat_feats_list, num_wrapping_dims=1)
            mask = util.get_text_field_mask(cat_feats_list, num_wrapping_dims=1)
            len_cat_feats_list = embedded_cat_feat.shape[1]
            enc_cat_feats = []
            for i in range(len_cat_feats_list):
                enc_cat_feats.append(self.encoder_cat_feats(embedded_cat_feat[:, i, :, :], mask[:, i, :]))
            enc_cat_feats = torch.cat(enc_cat_feats, dim=1)
            inputs.append(enc_cat_feats)

        if self.feat_type["num"]:
            # feature encoding:
            enc = self.feedforward(feat)
            inputs.append(enc)

        classifier_inputs = torch.cat(inputs, 1)

        return classifier_inputs


class MLP(BaseModel):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args

        assert args.n_hidden >= 0, "n_hidden must be nonnegative"

        self.output_layer = nn.Linear(
            args.emb_size if args.n_hidden == 0 else args.hidden_size,
            args.num_classes if not args.regression else 1,
        )

        # Init batch norm, dropout, and activation function
        self.init_hyperparameters()
        self.cls_parameter = self.get_cls_parameter()

        # Init hidden layers
        self.hidden_layers = self.init_hidden_layers()

        # Augmentation layers
        if self.args.gated:
            if self.args.n_hidden == 0:
                logging.info("Gated component requires at least one hidden layers in the model")
                pass
            else:
                # Init the mapping for the augmentation layer
                if self.args.gated_mapping is None:
                    # For each class init a discriminator component
                    self.mapping = torch.eye(self.args.num_groups, requires_grad=False)
                else:
                    # self.mapping = torch.from_numpy(mapping, requires_grad=False)
                    raise NotImplementedError

                self.augmentation_components = Augmentation_layer(
                    mapping=self.mapping,
                    num_component=self.args.num_groups,
                    device=self.args.device,
                    sample_component=self.hidden_layers
                )

        self.init_for_training()

    def forward(self, input_data, group_label=None):
        # main out
        main_output = input_data
        for layer in self.hidden_layers:
            main_output = layer(main_output)

        # Augmentation
        if self.args.gated and self.args.n_hidden > 0:
            assert group_label is not None, "Group labels are needed for augmentation"

            specific_output = self.augmentation_components(input_data, group_label)

            main_output = main_output + specific_output

        output = self.output_layer(main_output)
        return output

    def hidden(self, input_data, group_label=None):
        assert self.args.adv_level in ["input", "last_hidden", "output"]

        if self.args.adv_level == "input":
            return input_data
        else:
            # main out
            main_output = input_data
            for layer in self.hidden_layers:
                main_output = layer(main_output)

            # Augmentation
            if self.args.gated and self.args.n_hidden > 0:
                assert group_label is not None, "Group labels are needed for augmentation"

                specific_output = self.augmentation_components(input_data, group_label)

                main_output = main_output + specific_output
            if self.args.adv_level == "last_hidden":
                return main_output
            elif self.args.adv_level == "output":
                output = self.output_layer(main_output)
                return output
            else:
                raise "not implemented yet"

    def init_hidden_layers(self):
        args = self.args

        if args.n_hidden == 0:
            return nn.ModuleList()
        else:
            hidden_layers = nn.ModuleList()

            all_hidden_layers = [nn.Linear(args.emb_size, args.hidden_size)] + [
                nn.Linear(args.hidden_size, args.hidden_size) for _ in range(args.n_hidden - 1)]

            for _hidden_layer in all_hidden_layers:
                hidden_layers.append(_hidden_layer)
                if self.dropout is not None:
                    hidden_layers.append(self.dropout)
                if self.BN is not None:
                    hidden_layers.append(self.BN)
                if self.AF is not None:
                    hidden_layers.append(self.AF)
            return hidden_layers

    def get_cls_parameter(self):
        parameters = []
        if self.args.adv_level == "output":
            return parameters
        else:
            parameters.append(
                {"params": self.output_layer.parameters(), }
            )
            if self.args.adv_level == "last_hidden":
                return parameters
            elif self.args.adv_level == "input":
                parameters.append(
                    {"params": self.hidden_layers.parameters(), }
                )
                # Augmentation
                if self.args.gated and self.args.n_hidden > 0:
                    parameters.append(
                        {"params": self.augmentation_components.parameters(), }
                    )
                return parameters
            else:
                raise "not implemented yet"


class EvidenceGRADEr(Encoders, BaseModel):
    def __init__(self, args):
        self.args = args
        self.vocab = get_vocab(self.args)

        # kwargs["metrics"] = {"accuracy", "fscore_macro", "fscore_all", "mae"}
        metrics = set()

        # read experiment params from EvidenceGRADEr allennlp config file:
        model_params = self.get_params(param_f=self.args.param_file)

        # get embedders
        model_params["cat_feats_embedder"].params["token_embedders"]["cat_feats"][
            "num_embeddings"] = self.vocab.get_vocab_size()
        embedder = TextFieldEmbedder.from_params(model_params["embedder"])
        cat_feats_embedder = TextFieldEmbedder.from_params(model_params["cat_feats_embedder"])

        # get encoders
        encoder = Seq2VecEncoder.from_params(model_params["encoder"])
        encoder_cat_feats = Seq2VecEncoder.from_params(model_params["encoder_cat_feats"])

        super().__init__(self.vocab,
                         model_params["feedforward"],
                         embedder,
                         cat_feats_embedder,
                         encoder,
                         encoder_cat_feats,
                         metrics,
                         feat_type=model_params["feat_type"])

        self.classifier = MLP(args)
        self.init_for_training()

    def forward(self,
                input_data,
                group_label=None):
        encoder_output = self.encode(feat=input_data[0],
                                     text_list=input_data[1],
                                     cat_feats_list=input_data[2])
        encoder_output = self.layer_norm(encoder_output)

        output = self.classifier(encoder_output, group_label)

        return output

    def hidden(self,
               input_data,
               group_label=None):
        encoder_output = self.encode(feat=input_data[0],
                                     text_list=input_data[1],
                                     cat_feats_list=input_data[2])
        encoder_output = self.layer_norm(encoder_output)

        output = self.classifier.hidden(encoder_output, group_label)

        return output

    def get_params(self, param_f):
        params = Params.from_file(param_f)

        return params["model"]


class BERTClassifier(BaseModel):
    model_name = 'bert-base-cased'
    n_freezed_layers = 10

    def __init__(self, args):
        super(BERTClassifier, self).__init__()
        self.args = args

        self.bert = BertModel.from_pretrained(self.model_name)

        self.bert_layers = [self.bert.embeddings,
                            self.bert.encoder.layer[0],
                            self.bert.encoder.layer[1],
                            self.bert.encoder.layer[2],
                            self.bert.encoder.layer[3],
                            self.bert.encoder.layer[4],
                            self.bert.encoder.layer[5],
                            self.bert.encoder.layer[6],
                            self.bert.encoder.layer[7],
                            self.bert.encoder.layer[8],
                            self.bert.encoder.layer[9],
                            self.bert.encoder.layer[10],
                            self.bert.encoder.layer[11],
                            self.bert.pooler]

        self.classifier = MLP(args)

        self.freeze_roberta_layers(self.n_freezed_layers)

        self.init_for_training()

    def forward(self, input_data, group_label=None):
        bert_output = self.bert(input_data)[1]

        return self.classifier(bert_output, group_label)

    def hidden(self, input_data, group_label=None):
        bert_output = self.bert(input_data)[1]

        return self.classifier.hidden(bert_output, group_label)

    def freeze_roberta_layers(self, number_of_layers):
        "number of layers: the first number of layers to be freezed"
        assert (
                number_of_layers < 14 and number_of_layers > -14), "beyond the total number of RoBERTa layer groups(14)."
        for target_layer in self.bert_layers[:number_of_layers]:
            for param in target_layer.parameters():
                param.requires_grad = False
        for target_layer in self.bert_layers[number_of_layers:]:
            for param in target_layer.parameters():
                param.requires_grad = True

    def trainable_parameter_counting(self):
        model_parameters = filter(lambda p: p.requires_grad, self.bert.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


class ConvNet(BaseModel):

    def __init__(self, args):
        super(ConvNet, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)

        self.classifier = MLP(args)

        self.init_for_training()

    def forward(self, input_data, group_label=None):
        x = input_data
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)

        return self.classifier(x, group_label)

    def hidden(self, input_data, group_label=None):
        x = input_data
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)

        return self.classifier.hidden(x, group_label)
