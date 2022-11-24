from fairlib.src.utils import log_grid
from tutorial.binaryGRADE.eg_settings import args
from tutorial.evidencegrader_demo import train


def train_debiasing_dybt():
    for lambda_val in log_grid(-2, 0, 10):
        # prepare debiasing args
        debiasing_args = args.copy()
        # Update the experiment name
        debiasing_args["exp_id"] = f"DyBTFairbatch"
        # Specify the hyperparameters for Balanced Training
        debiasing_args["DyBT"] = "FairBatch"
        debiasing_args[
            "DyBTObj"] = "stratified_y"  # https://github.com/HanXudong/Systematic_Evaluation_of_Predictive_Fairness/blob/main/src/gen_exps_dist.py
        debiasing_args["DyBTalpha"] = lambda_val
        # train debiasing model
        train(debiasing_args)


train_debiasing_dybt()
