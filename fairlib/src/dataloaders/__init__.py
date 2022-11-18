import numpy as np
import torch
from allennlp.common import Params
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import TensorField, LabelField

from . import utils
from .encoder import text2id, get_vocab
from .loaders import default_dataset_roots, name2loader


def get_params(param_file):
    return Params.from_file(param_file)["data_loader"]


def expand_x(d):
    """
    Expand every instance in the data.X with protected label, instance weight, adv_instance_weight and regression label.
    We use allennlp's fields to encode those additional data.
    """
    for c, inst in enumerate(d.X):
        inst.add_field("protected_label",
                       LabelField(int(d.protected_label[c]), skip_indexing=True, label_namespace="protected_labels"))
        inst.add_field("instance_weights", TensorField(np.array(d.instance_weights[c])))
        inst.add_field("adv_instance_weights", TensorField(np.array(d.adv_instance_weights[c])))
        inst.add_field("regression_label",
                       LabelField(int(d.regression_label[c]), skip_indexing=True, label_namespace="regression_labels"))


def get_dataloaders(args):
    """Initialize the torch dataloaders according to arguments.

    Args:
        args (namespace): arguments

    Raises:
        NotImplementedError: if correspoding components have not been implemented.

    Returns:
        tuple: dataloaders for training set, development set, and test set.
    """
    task_dataloader = name2loader(args)

    if args.encoder_architecture in ["Fixed", "MNIST", "EvidenceGRADEr"]:
        pass
    elif args.encoder_architecture == "BERT":
        # Init the encoder form text to idx.
        args.text_encoder = text2id(args)
    else:
        raise NotImplementedError

    train_data = task_dataloader(args=args, split="train")
    dev_data = task_dataloader(args=args, split="dev")
    test_data = task_dataloader(args=args, split="test")

    # DataLoader Parameters
    train_dataloader_params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers}

    eval_dataloader_params = {
        'batch_size': args.test_batch_size,
        'shuffle': False,
        'num_workers': args.num_workers}

    # init dataloader
    if args.encoder_architecture == "EvidenceGRADEr":
        vocab = get_vocab(args)
        # add other fairlib-related data to data.X so that we can use allennlp's data loader to facilitate model training
        expand_x(train_data)
        expand_x(dev_data)
        expand_x(test_data)
        training_generator = SimpleDataLoader(train_data.X, vocab=vocab,
                                              batch_size=train_dataloader_params["batch_size"],
                                              shuffle=train_dataloader_params["shuffle"])
        validation_generator = SimpleDataLoader(dev_data.X, vocab=vocab,
                                                batch_size=eval_dataloader_params["batch_size"],
                                                shuffle=eval_dataloader_params["shuffle"])
        test_generator = SimpleDataLoader(test_data.X, vocab=vocab, batch_size=eval_dataloader_params["batch_size"],
                                          shuffle=eval_dataloader_params["shuffle"])
    else:
        training_generator = torch.utils.data.DataLoader(train_data, **train_dataloader_params)
        validation_generator = torch.utils.data.DataLoader(dev_data, **eval_dataloader_params)
        test_generator = torch.utils.data.DataLoader(test_data, **eval_dataloader_params)

    return training_generator, validation_generator, test_generator
