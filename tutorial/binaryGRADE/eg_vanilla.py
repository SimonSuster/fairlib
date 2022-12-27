from tutorial.binaryGRADE.eg_settings import args
from tutorial.evidencegrader_demo import train


def train_vanilla(fold_n):
    args["fold_n"] = fold_n
    args["exp_id"] = f"vanilla_fold{fold_n}"
    train(args)


for i in range(1, 10):
    train_vanilla(i)
