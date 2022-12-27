from fairlib.src.utils import log_grid
from tutorial.binaryGRADE.eg_settings import args
from tutorial.evidencegrader_demo import train


def train_debiasing_fcl():
    for i in log_grid(-3, 1, 10):
        lambda_g_val = float(i)
        lambda_y_val = float(i)
        # prepare debiasing args
        debiasing_args = args.copy()
        # Update the experiment name
        debiasing_args["exp_id"] = f"FCLg{lambda_g_val}_{lambda_y_val}"
        # Perform adversarial training if True
        debiasing_args["FCL"] = True
        debiasing_args["fcl_lambda_g"] = lambda_g_val
        debiasing_args["fcl_lambda_y"] = lambda_y_val
        # Specify the hyperparameters for Balanced Training
        # debiasing_args["BT"] = "Downsampling"
        # debiasing_args["BTObj"] = "EO"

        # train debiasing model
        train(debiasing_args)


train_debiasing_fcl()
