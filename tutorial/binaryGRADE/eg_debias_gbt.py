import sys

from tutorial.binaryGRADE.eg_settings import args
from tutorial.evidencegrader_demo import train


def train_debiasing_bt():
    """
    Carry out Balanced Training with different option combinations.
    """
    for BTObj_option in ["joint", "y", "g", "stratified_y", "stratified_g", "EO"]:
        # prepare debiasing args
        debiasing_args = args.copy()
        debiasing_args["GBT"] = True
        # Update the experiment name
        debiasing_args["exp_id"] = f"GBT{BTObj_option}"
        debiasing_args["BTObj"] = BTObj_option
        # train debiasing model
        train(debiasing_args)


train_debiasing_bt()
