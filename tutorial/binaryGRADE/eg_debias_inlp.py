from fairlib.__main__ import main
from tutorial.binaryGRADE.eg_settings import args


def train_debiasing_inlp():
    #for inlp_by_class in [True, False]:
    #    for reweighting in ["balanced", None]:
    inlp_by_class = False
    reweighting = None

    # prepare debiasing args
    debiasing_args = args.copy()
    # Update the experiment name
    debiasing_args["exp_id"] = f"dbgINLP{inlp_by_class}_{reweighting}"
    debiasing_args["INLP"] = True
    debiasing_args["INLP_n"] = 300
    debiasing_args["INLP_by_class"] = inlp_by_class
    debiasing_args["INLP_discriminator_reweighting"] = reweighting
    debiasing_args["INLP_min_acc"] = 0.5
    debiasing_args["INLP_model_load_path"] = "/home/simon/Apps/fairlib/tutorial/binaryGRADE/dev/EGBinaryGradeNum/vanilla/models/"

    # train debiasing model
    main(debiasing_args)


train_debiasing_inlp()
