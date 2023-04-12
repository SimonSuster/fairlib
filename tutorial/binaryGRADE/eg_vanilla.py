from tutorial.binaryGRADE.eg_settings_area import args
from tutorial.evidencegrader_demo import train


def train_vanilla(fold_n):
    args["fold_n"] = fold_n
    args["exp_id"] = f"dbg_vanilla_area_fold{fold_n}"
    train(args)


for i in range(1):
    train_vanilla(i)
