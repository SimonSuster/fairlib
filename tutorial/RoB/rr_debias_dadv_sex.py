from tutorial.RoB.rr_settings_sex import args
from tutorial.evidencegrader_demo import train


def train_debiasing_dadv():
    # explore the hyperparam grid:
    for diverse_lambda_val in [0.01, 0.1, 1, 10, 100]:
        adv_lambda = 0.251188643  # from adv hyper-tuning
        # prepare debiasing args
        debiasing_args = args.copy()
        # Update the experiment name
        debiasing_args["exp_id"] = f"RRSexDAdv{diverse_lambda_val}_{adv_lambda}"
        # Perform adversarial training if True
        debiasing_args["adv_debiasing"] = True
        #debiasing_args["adv_level"] = "input"
        debiasing_args["adv_lambda"] = adv_lambda
        debiasing_args["adv_batch_size"] = args["batch_size"]
        debiasing_args["adv_test_batch_size"] = args["batch_size"]
        debiasing_args["epochs"] = 3
        debiasing_args["epochs_since_improvement"] = 1
        debiasing_args["adv_n_hidden"] = 1
        debiasing_args["adv_num_subDiscriminator"] = 3
        debiasing_args["adv_diverse_lambda"] = diverse_lambda_val
        # Specify the hyperparameters for Balanced Training
        #debiasing_args["BT"] = "Reweighting"
        #debiasing_args["BTObj"] = "y"

        # train debiasing model
        train(debiasing_args)


train_debiasing_dadv()
