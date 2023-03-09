from fairlib.src.utils import log_grid
from tutorial.Sepsis.sepsis_settings_sex import args
from tutorial.evidencegrader_demo import train


def train_debiasing_adv():
    for lambda_val in log_grid(-3, 3, 10):
        lambda_val = float(lambda_val)
        # prepare debiasing args
        debiasing_args = args.copy()
        # Update the experiment name
        debiasing_args["exp_id"] = f"SepsisSexAdv{lambda_val}"
        # Perform adversarial training if True
        debiasing_args["adv_debiasing"] = True
        #debiasing_args["adv_level"] = "input"
        debiasing_args["adv_lambda"] = lambda_val
        debiasing_args["adv_batch_size"] = args["batch_size"]
        debiasing_args["adv_test_batch_size"] = args["batch_size"]
        debiasing_args["epochs"] = 3
        debiasing_args["epochs_since_improvement"] = 1
        debiasing_args["adv_n_hidden"] = 1
        debiasing_args["adv_num_subDiscriminator"] = 1
        debiasing_args["adv_diverse_lambda"] = 0.
        # Specify the hyperparameters for Balanced Training
        #debiasing_args["BT"] = "Reweighting"
        #debiasing_args["BTObj"] = "y"

        # train debiasing model
        train(debiasing_args)


train_debiasing_adv()
