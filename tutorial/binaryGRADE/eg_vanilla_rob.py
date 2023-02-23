from tutorial.binaryGRADE.eg_settings_rob import args
from tutorial.evidencegrader_demo import train


def train_vanilla(fold_n):
    args["fold_n"] = fold_n
    args["exp_id"] = f"vanilla_rob"
    train(args)


train_vanilla(0)
