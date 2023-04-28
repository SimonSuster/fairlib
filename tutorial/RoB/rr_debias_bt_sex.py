import sys

from tutorial.RoB.rr_settings_sex import args
from tutorial.evidencegrader_demo import train


def train_debiasing_bt(BT_option):
    """
    Carry out Balanced Training with different option combinations.
    """
    # uncomment this to use the best-only hyperparams:
    #best_btobj_per_option = {"Reweighting": "EO", "Resampling": "g", "Downsampling": "y"}
    #for BTObj_option in [best_btobj_per_option[BT_option]]:

    # explore the hyperparam grid:
    for BTObj_option in ["joint", "y", "g", "stratified_y", "stratified_g", "EO"]:
        # prepare debiasing args
        debiasing_args = args.copy()
        # Update the experiment name
        debiasing_args["exp_id"] = f"RRSexBT{BT_option}{BTObj_option}"
        # Specify the hyperparameters for Balanced Training
        debiasing_args["BT"] = BT_option
        debiasing_args["BTObj"] = BTObj_option
        # train debiasing model
        train(debiasing_args)


BT_option = sys.argv[1]
BT_options = ["Reweighting", "Resampling", "Downsampling"]
assert BT_option in BT_options, f"Should be one of {BT_options}"

train_debiasing_bt(BT_option)
