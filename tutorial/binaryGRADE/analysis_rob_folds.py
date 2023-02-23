import numpy as np

from fairlib.src import analysis
from fairlib.src.analysis.utils import get_calib_scores_folds, analyse_selective_classification
from tutorial.binaryGRADE.eg_settings_rob import args

task = "binaryRob"
n_folds = 10
results_per_fold = {}
for fold_n in range(n_folds):
    Shared_options = {
        # Random seed
        "seed": 2022,
        # The name of the dataset, corresponding dataloader will be used,
        "dataset": f"EGBinaryGradeNumFolds_{fold_n}",
        # Specifiy the path to the input data
        "data_dir": f"{args['data_dir']}{fold_n}",
        # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
        "device_id": -1,
        # The default path for saving experimental results
        "results_dir": f"/home/simon/Apps/fairlib/tutorial/{task}/",
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
        "n_jobs": 4,
        "do_calib_eval": True,
        "calib_metric_name": "aurc",
        #"calib_selection_criterion": "DTO"
        "calib_selection_criterion": "performance"
        #"calib_selection_criterion": None
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
        do_calib_eval=Shared_options["do_calib_eval"],
        calib_metric_name=Shared_options["calib_metric_name"],
        calib_selection_criterion=Shared_options["calib_selection_criterion"]

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
        n_jobs=Shared_options["n_jobs"],
        do_calib_eval=Shared_options["do_calib_eval"],
        calib_metric_name=Shared_options["calib_metric_name"],
        calib_selection_criterion=Shared_options["calib_selection_criterion"]

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
        n_jobs=Shared_options["n_jobs"],
        do_calib_eval=Shared_options["do_calib_eval"],
        calib_metric_name=Shared_options["calib_metric_name"],
        calib_selection_criterion=Shared_options["calib_selection_criterion"]

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
        n_jobs=Shared_options["n_jobs"],
        do_calib_eval=Shared_options["do_calib_eval"],
        calib_metric_name=Shared_options["calib_metric_name"],
        calib_selection_criterion=Shared_options["calib_selection_criterion"]

    )

    analysis.model_selection(
        model_id=("Adv"),
        index_column_names=['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda', "BT"],
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
        do_calib_eval=Shared_options["do_calib_eval"],
        calib_metric_name=Shared_options["calib_metric_name"],
        calib_selection_criterion=Shared_options["calib_selection_criterion"]

    )

    analysis.model_selection(
        model_id=("DAdv"),
        index_column_names=['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda', "BT"],
        save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}DADV_df.pkl",
        # Follwoing options are predefined
        results_dir=Shared_options["results_dir"],
        project_dir=Shared_options["project_dir"] + "/" + Shared_options["dataset"],
        GAP_metric_name=Shared_options["GAP_metric_name"],
        Performance_metric_name=Shared_options["Performance_metric_name"],
        selection_criterion=Shared_options["selection_criterion"],
        checkpoint_dir=Shared_options["checkpoint_dir"],
        checkpoint_name=Shared_options["checkpoint_name"],
        n_jobs=Shared_options["n_jobs"],
        do_calib_eval=Shared_options["do_calib_eval"],
        calib_metric_name=Shared_options["calib_metric_name"],
        calib_selection_criterion=Shared_options["calib_selection_criterion"]

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
        do_calib_eval=Shared_options["do_calib_eval"],
        calib_metric_name=Shared_options["calib_metric_name"],
        calib_selection_criterion=Shared_options["calib_selection_criterion"]
    
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
        do_calib_eval=Shared_options["do_calib_eval"],
        calib_metric_name=Shared_options["calib_metric_name"],
        calib_selection_criterion=Shared_options["calib_selection_criterion"]
    )

    EG_results, EG_calib_results = analysis.retrive_results(Shared_options["dataset"], log_dir=Shared_options["results_dir"], do_calib_eval=Shared_options["do_calib_eval"])
    results_per_fold[fold_n] = {"EG_results": EG_results, "EG_calib_results": EG_calib_results}
    print()

"""
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
    #do_calib_eval=Shared_options["do_calib_eval"],
    do_calib_eval=False,
    calib_metric_name=Shared_options["calib_metric_name"],
    calib_selection_criterion=Shared_options["calib_selection_criterion"]
)
"""

EG_calib_results_folds = get_calib_scores_folds(results_per_fold)
"""
#pareto = False
pareto = True
selection_criterion = None
#selection_criterion = "DTO"
calib_selection_criterion=Shared_options["calib_selection_criterion"]
#calib_selection_criterion=None
EG_main_results, EG_calib_main_results = analysis.final_results_df(
    results_dict=EG_results,
    pareto=pareto,
    pareto_selection="test",
    selection_criterion=selection_criterion,
    return_dev=True,
    return_conf=True,
    Fairness_threshold=-10,
    calib_additional_metrics=["dev_performance", "test_performance"],
    do_calib_eval=Shared_options["do_calib_eval"],
    calib_results_dict=EG_calib_results,
    calib_metric_name=Shared_options["calib_metric_name"],
    calib_selection_criterion=calib_selection_criterion
)
"""
aurc_raw_out =True
if aurc_raw_out:
    #assert (not pareto) and (selection_criterion is None) and (calib_selection_criterion == "performance")
    with open(f"/home/simon/Apps/SysRevData/data/modelling/plots/evidencegrader/evidencegrader_aurc_raw_{task}_debias.csv", "w") as fh_out:
        fh_out.write("model\tcriterion\ttopic\tcoverage\trisk\n")
        for model, d in EG_calib_results_folds.items():
            all_results = d["aurc_raw"]
            for coverage, risk in all_results:
                fh_out.write("{}\t{}\t{}\t{}\t{}\n".format(model, task, "all", coverage, risk))
            #per_private_label_results = selected_row.dev_per_private_label.to_list()[0]
            per_private_label_results = d["per_private_label"]
            for protected_label, results in per_private_label_results.items():
                if not results:
                    continue
                for coverage, risk in results["aurc_raw"]:
                    fh_out.write("{}\t{}\t{}\t{}\t{}\n".format(model, task, protected_label, coverage, risk))
    with open(
            f"/home/simon/Apps/SysRevData/data/modelling/plots/evidencegrader/evidencegrader_aurc_{task}_debias.csv",
            "w") as fh_out:
        fh_out.write("model\tcriterion\ttopic\taurc\n")
        for model, d in EG_calib_results_folds.items():
            per_private_label_results = d["per_private_label"]
            for protected_label, results in per_private_label_results.items():
                if not results:
                    continue
                fh_out.write("{}\t{}\t{}\t{}\n".format(model, task, protected_label, results["aurc"]))

for result, d in analyse_selective_classification(EG_calib_results_folds).items():
    print(result)
    print(d)


"""
analysis.tables_and_figures.make_plot(
    EG_main_results,
    figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/plot.png",
    performance_name=Shared_options["Performance_metric_name"]
)

analysis.tables_and_figures.make_plot(
    EG_calib_main_results,
    figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/plot_calib.png",
    performance_name=Shared_options["calib_metric_name"]
)
"""
analysis.tables_and_figures.make_plot_folds(EG_calib_results_folds,
    figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/EGBinaryGradeNum/plot_folds.png",
    performance_name=Shared_options["Performance_metric_name"]
)
analysis.tables_and_figures.make_plot_folds(EG_calib_results_folds,
    figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/EGBinaryGradeNum/plot_calib_folds.png",
    performance_name=Shared_options["calib_metric_name"]
)



# fairlib.analysis.tables_and_figures.make_zoom_plot(
#    EG_main_results, figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/zoom_plot.png", performance_name=Shared_options["Performance_metric_name"], dpi = 100,
#    zoom_xlim=(0.6, 0.78),
#    zoom_ylim=(0.8, 0.98)
#    )
#print(EG_main_results)
#print(EG_calib_main_results)

