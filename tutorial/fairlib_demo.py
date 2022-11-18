import fairlib

from fairlib import analysis

# datasets.prepare_dataset("moji", "data/deepmoji")
# datasets.name2class.keys()

args = {
    # The name of the dataset, corresponding dataloader will be used,
    "dataset": "Moji",

    # Specifiy the path to the input data
    "data_dir": "data/deepmoji",

    # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
    "device_id": -1,

    # Give a name to the exp, which will be used in the path
    "exp_id": "vanilla",
    "batch_size": 32
}

# Init the argument
options = fairlib.BaseOptions()
state = options.get_state(args=args, silence=True)
print(state.num_groups, state.num_classes, state.dataset)
print(state.hidden_size, state.n_hidden, state.activation_function)

fairlib.utils.seed_everything(2022)

# Init Model
model = fairlib.networks.get_main_model(state)
model.train_self()
"""
debiasing_args = args.copy()

# Update the experiment name
debiasing_args["exp_id"] = "BTOnly"

# Perform adversarial training if True
debiasing_args["adv_debiasing"] = False

# Specify the hyperparameters for Balanced Training
debiasing_args["BT"] = "Downsampling"
debiasing_args["BTObj"] = "EO"

debias_options = fairlib.BaseOptions()
debias_state = debias_options.get_state(args=debiasing_args, silence=True)

fairlib.utils.seed_everything(2022)

debias_model = fairlib.networks.get_main_model(debias_state)
debias_model.train_self()
"""
Shared_options = {
    # Random seed
    "seed": 2022,

    # The name of the dataset, corresponding dataloader will be used,
    "dataset": "Moji",

    # Specifiy the path to the input data
    "data_dir": "data/deepmoji",

    # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
    "device_id": -1,

    # The default path for saving experimental results
    "results_dir": "results",

    # Will be used for saving experimental results
    "project_dir": "dev",

    # We will focusing on TPR GAP, implying the Equalized Odds for binary classification.
    "GAP_metric_name": "TPR_GAP",

    # The overall performance will be measured as accuracy
    "Performance_metric_name": "accuracy",

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
    index_column_names=["BT", "BTObj", "adv_debiasing"],

    # to convenient the further analysis, we will store the resulting DataFrame to the specified path
    # save_path = r"results/Vanilla_df.pkl",
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/{Shared_options['dataset']}BT_ADV_df.pkl",
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
    model_id=("BTOnly"),
    index_column_names=["BT", "BTObj", "adv_debiasing"],
    # save_path = r"results/BT_ADV_df.pkl",
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/{Shared_options['dataset']}BT_ADV_df.pkl",
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

Moji_results = analysis.retrive_results("MojiDbg", log_dir="results")
Moji_main_results = analysis.final_results_df(
    results_dict=Moji_results,
    pareto=False,
    pareto_selection="test",
    selection_criterion="DTO",
    return_dev=True,
    return_conf=True,
)
Moji_main_results
fairlib.analysis.tables_and_figures.make_plot(
    Moji_main_results, figure_name="/home/simon/Apps/fairlib/tutorial/results/dev/Moji/plot.png"
)
