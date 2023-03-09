from tutorial.Sepsis.sepsis_settings_sex import args
from tutorial.evidencegrader_demo import train


def train_vanilla():
    # prepare debiasing args
    debiasing_args = args.copy()
    # Update the experiment name
    debiasing_args["exp_id"] = f"SepsisSexVanilla"

    train(debiasing_args)


train_vanilla()
