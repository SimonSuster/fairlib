from tutorial.binaryGRADE.eg_settings import args
from tutorial.evidencegrader_demo import train
from fairlib.__main__ import main

def train_debiasing_inlp():
    # prepare debiasing args
    debiasing_args = args.copy()
    # Update the experiment name
    debiasing_args["exp_id"] = f"INLP"
    debiasing_args["INLP_n"] = 300
    debiasing_args["INLP_by_class"] = True
    debiasing_args["INLP_discriminator_reweighting"] = "balanced"
    debiasing_args["INLP_min_acc"] = 0.1

    # train debiasing model
    main(debiasing_args)


train_debiasing_inlp()
