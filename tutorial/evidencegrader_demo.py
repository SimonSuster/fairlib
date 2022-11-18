import fairlib

from fairlib import analysis

from fairlib.src.dataloaders.loaders.EGBinaryGradeNum import TOPICS


# Init the argument
def train(args):
    options = fairlib.BaseOptions()
    state = options.get_state(args=args, silence=True)
    print(state.num_groups, state.num_classes, state.dataset)
    print(state.hidden_size, state.n_hidden, state.activation_function)
    fairlib.utils.seed_everything(2022)
    # Init Model
    model = fairlib.networks.get_main_model(state)
    model.train_self()


task = "bi_class2_nonaugm_all"
data_dir = "/home/simon/Apps/SysRevData/data/derivations/all/splits/"
serialization_dir = "/home/simon/Apps/SysRevData/data/modelling/saved/"
fold_n = 0
args = {
    # The name of the dataset, corresponding dataloader will be used,
    "dataset": "EGBinaryGradeNum",
    "encoder_architecture": "EvidenceGRADEr",
    # Specifiy the path to the input data
    "data_dir": data_dir,
    "scaler_training_path": f"{data_dir}FOLD/train.csv",
    "fold_n": fold_n,
    "serialization_dir": serialization_dir,
    "vocabulary_dir": f"{serialization_dir}{task}/{task}_{fold_n}/vocabulary",
    # if it does not exist, the vocabulary will be created from scratch
    "param_file": f"/home/simon/Apps/SysRev/sysrev/modelling/allennlp/training_config/{task}.jsonnet",
    # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
    "device_id": -1,
    # Give a name to the exp, which will be used in the path
    "exp_id": "vanilla",
    "num_groups": len(TOPICS),
    # "emb_size": len(NUM_TYPES),
    "emb_size": 3232,
    "n_hidden": 1,  # although in original EG, n_hidden=0, some de-biasing methods require n_hidden>0
    # "max_load": 10,
    "batch_size": 2,
    "group_agg_power": 2
}


def train_vanilla():
    train(args)


def train_debiasing_bt():
    # prepare debiasing args
    debiasing_args = args.copy()
    # Update the experiment name
    debiasing_args["exp_id"] = f"BT"
    # Specify the hyperparameters for Balanced Training
    debiasing_args["BT"] = "Downsampling"
    debiasing_args["BTObj"] = "EO"
    # train debiasing model
    train(debiasing_args)


def train_debiasing_adv():
    for i in range(11):
        lambda_val = i / 10
        # prepare debiasing args
        debiasing_args = args.copy()
        # Update the experiment name
        debiasing_args["exp_id"] = f"BTEOAdv{lambda_val}"
        # Perform adversarial training if True
        debiasing_args["adv_debiasing"] = True
        debiasing_args["adv_lambda"] = lambda_val
        # Specify the hyperparameters for Balanced Training
        debiasing_args["BT"] = "Downsampling"
        debiasing_args["BTObj"] = "EO"

        # train debiasing model
        train(debiasing_args)


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


train_vanilla()
# train_debiasing_bt()
# train_debiasing_adv()
# train_debiasing_fcl()

Shared_options = {
    # Random seed
    "seed": 2022,
    # The name of the dataset, corresponding dataloader will be used,
    "dataset": "EGBinaryGradeNum",
    # Specifiy the path to the input data
    "data_dir": f"{args['data_dir']}{args['fold_n']}",
    # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
    "device_id": -1,
    # The default path for saving experimental results
    "results_dir": "results",
    # Will be used for saving experimental results
    "project_dir": "dev",
    # We will focusing on TPR GAP, implying the Equalized Odds for binary classification.
    "GAP_metric_name": "TPR_GAP",
    # The overall performance will be measured as
    "Performance_metric_name": "macro_fscore",
    # Model selections are based on distance to optimum, see section 4 in our paper for more details
    "selection_criterion": "DTO",
    # Default dirs for saving checkpoints
    "checkpoint_dir": "models",
    "checkpoint_name": "checkpoint_epoch",
    # Loading experimental results
    "n_jobs": 1,
}
analysis.model_selection(
    # exp_id started with model_id will be treated as the same method, e.g, vanilla, and adv
    model_id=("vanilla"),
    # the tuned hyperparameters of a methods, which will be used to group multiple runs together.
    index_column_names=["BT", "BTObj"],
    # to convenient the further analysis, we will store the resulting DataFrame to the specified path
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/{Shared_options['dataset']}vanilla_df.pkl",
    # Follwoing options are predefined
    results_dir=Shared_options["results_dir"],
    project_dir=Shared_options["project_dir"] + "/" + Shared_options["dataset"],
    GAP_metric_name=Shared_options["GAP_metric_name"],
    Performance_metric_name=Shared_options["Performance_metric_name"],
    # We use DTO for epoch selection
    selection_criterion=Shared_options["selection_criterion"],
    checkpoint_dir=Shared_options["checkpoint_dir"],
    checkpoint_name=Shared_options["checkpoint_name"],
    # If retrive results in parallel
    n_jobs=Shared_options["n_jobs"],
)

analysis.model_selection(
    model_id=("BT"),
    # index_column_names = ["BT", "BTObj"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/{Shared_options['dataset']}BT_df.pkl",
    # Follwoing options are predefined
    results_dir=Shared_options["results_dir"],
    project_dir=Shared_options["project_dir"] + "/" + Shared_options["dataset"],
    GAP_metric_name=Shared_options["GAP_metric_name"],
    Performance_metric_name=Shared_options["Performance_metric_name"],
    selection_criterion=Shared_options["selection_criterion"],
    checkpoint_dir=Shared_options["checkpoint_dir"],
    checkpoint_name=Shared_options["checkpoint_name"],
    n_jobs=Shared_options["n_jobs"],
)
analysis.model_selection(
    model_id=("Adv"),
    # index_column_names = ["BT", "BTObj"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/{Shared_options['dataset']}ADV_df.pkl",
    # Follwoing options are predefined
    results_dir=Shared_options["results_dir"],
    project_dir=Shared_options["project_dir"] + "/" + Shared_options["dataset"],
    GAP_metric_name=Shared_options["GAP_metric_name"],
    Performance_metric_name=Shared_options["Performance_metric_name"],
    selection_criterion=Shared_options["selection_criterion"],
    checkpoint_dir=Shared_options["checkpoint_dir"],
    checkpoint_name=Shared_options["checkpoint_name"],
    n_jobs=Shared_options["n_jobs"],
)
analysis.model_selection(
    model_id=("BTEOAdv"),
    # index_column_names = ["BT", "BTObj"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/{Shared_options['dataset']}BTEOADV_df.pkl",
    # Follwoing options are predefined
    results_dir=Shared_options["results_dir"],
    project_dir=Shared_options["project_dir"] + "/" + Shared_options["dataset"],
    GAP_metric_name=Shared_options["GAP_metric_name"],
    Performance_metric_name=Shared_options["Performance_metric_name"],
    selection_criterion=Shared_options["selection_criterion"],
    checkpoint_dir=Shared_options["checkpoint_dir"],
    checkpoint_name=Shared_options["checkpoint_name"],
    n_jobs=Shared_options["n_jobs"],
)
analysis.model_selection(
    model_id=("FCL"),
    index_column_names=["fcl_lambda_g", "fcl_lambda_y"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/{Shared_options['dataset']}FCL_df.pkl",
    # Follwoing options are predefined
    results_dir=Shared_options["results_dir"],
    project_dir=Shared_options["project_dir"] + "/" + Shared_options["dataset"],
    GAP_metric_name=Shared_options["GAP_metric_name"],
    Performance_metric_name=Shared_options["Performance_metric_name"],
    selection_criterion=Shared_options["selection_criterion"],
    checkpoint_dir=Shared_options["checkpoint_dir"],
    checkpoint_name=Shared_options["checkpoint_name"],
    n_jobs=Shared_options["n_jobs"],
)
print()

EG_results = analysis.retrive_results("EGBinaryGradeNum", log_dir="results")
EG_main_results = analysis.final_results_df(
    results_dict=EG_results,
    pareto=True,
    pareto_selection="test",
    selection_criterion=None,
    # selection_criterion = "DTO",
    return_dev=True,
    return_conf=True,
    Fairness_threshold=-10
)

fairlib.analysis.tables_and_figures.make_plot(
    EG_main_results,
    figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/plot.png",
    performance_name=Shared_options["Performance_metric_name"]
)

# fairlib.analysis.tables_and_figures.make_zoom_plot(
#    EG_main_results, figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/zoom_plot.png", performance_name=Shared_options["Performance_metric_name"], dpi = 100,
#    zoom_xlim=(0.6, 0.78),
#    zoom_ylim=(0.8, 0.98)
#    )
print(EG_main_results)
