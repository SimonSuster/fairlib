import sys

from tutorial.binaryGRADE.eg_settings import args
from tutorial.evidencegrader_demo import train


def train_debiasing_bt(BT_option, fold_n):
    """
    Carry out Balanced Training with different option combinations.
    """
    best_btobj_per_option = {"Reweighting": "EO", "Resampling": "g", "Downsampling": "y"}
    #for BTObj_option in ["joint", "y", "g", "stratified_y", "stratified_g", "EO"]:
    for BTObj_option in [best_btobj_per_option[BT_option]]:
        # prepare debiasing args
        debiasing_args = args.copy()
        debiasing_args["fold_n"] = fold_n
        # Update the experiment name
        debiasing_args["exp_id"] = f"BT{BT_option}{BTObj_option}_fold{fold_n}"
        # Specify the hyperparameters for Balanced Training
        debiasing_args["BT"] = BT_option
        debiasing_args["BTObj"] = BTObj_option
        # train debiasing model
        train(debiasing_args)


BT_option = sys.argv[1]
BT_options = ["Reweighting", "Resampling", "Downsampling"]
assert BT_option in BT_options, f"Should be one of {BT_options}"

for i in range(1, 10):
    train_debiasing_bt(BT_option, i)
