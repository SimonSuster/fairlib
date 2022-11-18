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
    elif args.encoder_architecture == "MNIST":
        model = ConvNet(args)
    elif args.encoder_architecture == "EvidenceGRADEr":
        model = EvidenceGRADEr(args)
    else:
        raise NotImplementedError

    return model
