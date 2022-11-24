from tutorial.binaryGRADE.eg_settings import args
from tutorial.evidencegrader_demo import train


def train_debiasing_fcl():
    for i in range(11):
        lambda_val = i / 10
        # prepare debiasing args
        debiasing_args = args.copy()
        # Update the experiment name
        debiasing_args["exp_id"] = f"FCLg{lambda_val}"
        # Perform adversarial training if True
        debiasing_args["FCL"] = True
        debiasing_args["fcl_lambda_g"] = lambda_val
        # debiasing_args["fcl_lambda_y"] = lambda_val
        # Specify the hyperparameters for Balanced Training
        # debiasing_args["BT"] = "Downsampling"
        # debiasing_args["BTObj"] = "EO"

        # train debiasing model
        train(debiasing_args)


train_debiasing_fcl()
