from fairlib.src import analysis
from tutorial.binaryGRADE.eg_settings import args

Shared_options = {
    # Random seed
    "seed": 2022,
    # The name of the dataset, corresponding dataloader will be used,
    "dataset": "EGBinaryGrade",
    # Specifiy the path to the input data
    "data_dir": f"{args['data_dir']}{args['fold_n']}",
    # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
    "device_id": -1,
    # The default path for saving experimental results
    "results_dir": "/home/simon/Apps/fairlib/tutorial/binaryGRADE/",
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
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}vanilla_df.pkl",
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
    model_id=("BTDownsampling"),
    index_column_names = ["BT", "BTObj"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}BTDownsampling_df.pkl",
    # Follwoing options are predefined
    results_dir=Shared_options["results_dir"],
    project_dir=Shared_options["project_dir"] + "/" + Shared_options["dataset"],
    GAP_metric_name=Shared_options["GAP_metric_name"],
    Performance_metric_name=Shared_options["Performance_metric_name"],
    selection_criterion=Shared_options["selection_criterion"],
    checkpoint_dir=Shared_options["checkpoint_dir"],
    checkpoint_name=Shared_options["checkpoint_name"],
    n_jobs=Shared_options["n_jobs"]
)


analysis.model_selection(
    model_id=("BTResampling"),
    index_column_names = ["BT", "BTObj"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}BTResampling_df.pkl",
    # Follwoing options are predefined
    results_dir=Shared_options["results_dir"],
    project_dir=Shared_options["project_dir"] + "/" + Shared_options["dataset"],
    GAP_metric_name=Shared_options["GAP_metric_name"],
    Performance_metric_name=Shared_options["Performance_metric_name"],
    selection_criterion=Shared_options["selection_criterion"],
    checkpoint_dir=Shared_options["checkpoint_dir"],
    checkpoint_name=Shared_options["checkpoint_name"],
    n_jobs=Shared_options["n_jobs"]
)

analysis.model_selection(
    model_id=("BTReweighting"),
    index_column_names = ["BT", "BTObj"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}BTReweighting_df.pkl",
    # Follwoing options are predefined
    results_dir=Shared_options["results_dir"],
    project_dir=Shared_options["project_dir"] + "/" + Shared_options["dataset"],
    GAP_metric_name=Shared_options["GAP_metric_name"],
    Performance_metric_name=Shared_options["Performance_metric_name"],
    selection_criterion=Shared_options["selection_criterion"],
    checkpoint_dir=Shared_options["checkpoint_dir"],
    checkpoint_name=Shared_options["checkpoint_name"],
    n_jobs=Shared_options["n_jobs"]
)

analysis.model_selection(
    model_id=("Adv"),
    index_column_names=['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda'],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}ADV_df.pkl",
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

"""
analysis.model_selection(
    model_id=("BTEOAdv"),
    # index_column_names = ["BT", "BTObj"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}BTEOADV_df.pkl",
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
"""

analysis.model_selection(
    model_id=("FCL"),
    index_column_names=["fcl_lambda_g", "fcl_lambda_y"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}FCL_df.pkl",
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
    model_id=("INLP"),
    index_column_names=["INLP_by_class", "INLP_discriminator_reweighting"],
    save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}INLP_df.pkl",
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

EG_results = analysis.retrive_results(Shared_options["dataset"], log_dir=Shared_options["results_dir"])
EG_main_results = analysis.final_results_df(
    results_dict=EG_results,
    pareto=True,
    #pareto=False,
    pareto_selection="test",
    selection_criterion=None,
    #selection_criterion = "DTO",
    return_dev=True,
    return_conf=True,
    Fairness_threshold=-10
)

analysis.tables_and_figures.make_plot(
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
