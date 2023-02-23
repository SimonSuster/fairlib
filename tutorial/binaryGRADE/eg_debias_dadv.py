from tutorial.binaryGRADE.eg_settings import args
from tutorial.evidencegrader_demo import train


def train_debiasing_dadv(fold_n):
    #for diverse_lambda_val in [0.01, 0.1, 1, 10, 100]:
    for diverse_lambda_val in [0.01]:
        adv_lambda = float(1.58489319e-02)  # from adv hyper-tuning
        # prepare debiasing args
        debiasing_args = args.copy()
        debiasing_args["fold_n"] = fold_n
        # Update the experiment name
        debiasing_args["exp_id"] = f"DAdv{diverse_lambda_val}_{adv_lambda}_fold{fold_n}"
        # Perform adversarial training if True
        debiasing_args["adv_debiasing"] = True
        debiasing_args["adv_lambda"] = adv_lambda
        debiasing_args["adv_batch_size"] = args["batch_size"]
        debiasing_args["adv_test_batch_size"] = args["batch_size"]
        debiasing_args["epochs"] = 3
        debiasing_args["epochs_since_improvement"] = 1
        debiasing_args["adv_n_hidden"] = 1
        debiasing_args["adv_num_subDiscriminator"] = 3
        debiasing_args["adv_diverse_lambda"] = diverse_lambda_val
        # Specify the hyperparameters for Balanced Training
        #debiasing_args["BT"] = "Downsampling"
        #debiasing_args["BTObj"] = "EO"

        # train debiasing model
        train(debiasing_args)


for i in range(1, 10):
    train_debiasing_dadv(i)
