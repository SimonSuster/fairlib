from pathlib import Path

from allennlp.data import Vocabulary
from transformers import AutoTokenizer

from fairlib.src.dataloaders.loaders.EGBinaryGrade import get_dataset_reader


class text2id():
    """mapping natural language to numeric identifiers.
    """

    def __init__(self, args) -> None:
        if args.encoder_architecture == "Fixed":
            self.encoder = None
        elif args.encoder_architecture == "BERT":
            self.model_name = 'bert-base-cased'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        elif args.encoder_architecture == "SciBERT":
            self.model_name = 'allenai/scibert_scivocab_uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=512)
        elif args.encoder_architecture == "BlueBERT":
            self.model_name = 'bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=512)
        elif args.encoder_architecture == "ClinicalBERT":
            self.model_name = 'emilyalsentzer/Bio_ClinicalBERT'
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, model_max_length=512)
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
        dataset_reader = get_dataset_reader(args.data_dir, args.fold_n, args.serialization_dir,
                                            args.scaler_training_path, param_file=args.param_file)
        data_path = f"{args.data_dir}{args.fold_n}/train.csv"
        reader = dataset_reader._read(data_path)
        vocab = Vocabulary.from_instances(reader)
        vocab.save_to_files(args.vocabulary_dir)

    return vocab
