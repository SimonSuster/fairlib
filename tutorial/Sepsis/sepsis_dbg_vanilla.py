from tutorial.Sepsis.sepsis_settings_sex import args
from tutorial.evidencegrader_demo import train


def train_vanilla():
    """
    Train two models to see if we get the same result.
    """
    # prepare debiasing args
    debiasing_args = args.copy()
    # Update the experiment name
    debiasing_args["exp_id"] = f"vanillaDbg"
    debiasing_args["epochs_since_improvement"] = 1
    debiasing_args["epochs"] = 2

    train(debiasing_args)


train_vanilla()
