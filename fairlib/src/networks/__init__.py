from . import DyBT
from . import FairCL
from . import INLP
from . import adv
from . import utils
from .classifier import MLP, BERTClassifier, ConvNet, EvidenceGRADEr


def get_main_model(args):
    if args.encoder_architecture == "Fixed":
        model = MLP(args)
    elif args.encoder_architecture == "BERT":
        model = BERTClassifier(args)
    elif args.encoder_architecture == "SciBERT":
        model = BERTClassifier(args, model_name="allenai/scibert_scivocab_uncased")
    elif args.encoder_architecture == "BlueBERT":
        model = BERTClassifier(args, model_name="bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16")
    elif args.encoder_architecture == "ClinicalBERT":
        model = BERTClassifier(args, model_name="emilyalsentzer/Bio_ClinicalBERT")
    elif args.encoder_architecture == "MNIST":
        model = ConvNet(args)
    elif args.encoder_architecture == "EvidenceGRADEr":
        model = EvidenceGRADEr(args)
    else:
        raise NotImplementedError

    return model
