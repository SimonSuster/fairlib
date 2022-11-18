from pathlib import Path

from allennlp.data import Vocabulary
from transformers import AutoTokenizer

from fairlib.src.dataloaders.loaders.EGBinaryGradeNum import get_dataset_reader


class text2id():
    """mapping natural language to numeric identifiers.
    """
    def __init__(self, args) -> None:
        if args.encoder_architecture == "Fixed":
            self.encoder = None
        elif args.encoder_architecture == "BERT":
            self.model_name = 'bert-base-cased'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        else:
            raise NotImplementedError
    
    def encoder(self, sample):
        encodings = self.tokenizer(sample, truncation=True, padding=True)
        return encodings["input_ids"]


def get_vocab(args):
    if Path(args.vocabulary_dir).exists():
        vocab = Vocabulary.from_files(args.vocabulary_dir)
    else:
        Path(args.vocabulary_dir).mkdir(parents=True)
        dataset_reader = get_dataset_reader(args.data_dir, args.fold_n, param_file=args.param_file)
        data_path = f"{args.data_dir}{args.fold_n}/train.csv"
        reader = dataset_reader._read(data_path)
        vocab = Vocabulary.from_instances(reader)
        vocab.save_to_files(args.vocabulary_dir)

    return vocab


class TextIndexer:
    def __init__(self, args):
        self.args = args
        self.vocab = get_vocab(self.args)

    def index(self, text_field_list):
        indexed_list = []
        for text_field in text_field_list:
            text_field.index(self.vocab)
            padding_lengths = text_field.get_padding_lengths()
            tensor_dict = text_field.as_tensor(padding_lengths)
            indexed_list.append(tensor_dict)

        return indexed_list



