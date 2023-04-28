import argparse
import os

from fairlib.src import analysis
from fairlib.src.utils import write_raw_out_rr, shorten_debias_name
from tutorial.RoB.rr_settings_sex import args


def main(results_dir, calib_raw_out_f, perf_raw_out_f):
    task = "RoB"
    Shared_options = {
        # Random seed
        "seed": 2022,
        # The name of the dataset, corresponding dataloader will be used,
        "dataset": "RoB",
        # Specifiy the path to the input data
        "data_dir": f"{args['data_dir']}",
        # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
        "device_id": -1,
        # The default path for saving experimental results
        "results_dir": os.path.join(results_dir, task),
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
        "do_calib_eval": True,
        "calib_metric_name": "aurc",
        "calib_selection_criterion": "DTO"
        #"calib_selection_criterion": None
    }

    analysis.model_selection(
        # exp_id started with model_id will be treated as the same method, e.g, vanilla, and adv
        model_id=("RRSexVanilla"),
        # the tuned hyperparameters of a methods, which will be used to group multiple runs together.
        index_column_names=["BT", "BTObj"],
        # to convenient the further analysis, we will store the resulting DataFrame to the specified path
        save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}Sexvanilla_df.pkl",
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
        calib_metric_name=Shared_options["calib_metric_name"]
    )

    analysis.model_selection(
        model_id=("RRSexBTDownsampling"),
        index_column_names = ["BT", "BTObj"],
        save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}SexBTDownsampling_df.pkl",
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
        calib_metric_name=Shared_options["calib_metric_name"]

    )


    analysis.model_selection(
        model_id=("RRSexBTResampling"),
        index_column_names = ["BT", "BTObj"],
        save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}SexBTResampling_df.pkl",
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
        calib_metric_name=Shared_options["calib_metric_name"]
    )

    analysis.model_selection(
        model_id=("RRSexBTReweighting"),
        index_column_names = ["BT", "BTObj"],
        save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}SexBTReweighting_df.pkl",
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
        calib_metric_name=Shared_options["calib_metric_name"]
    )

    analysis.model_selection(
        model_id=("RRSexAdv"),
        index_column_names=['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda', "BT"],
        save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}SexADV_df.pkl",
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
        calib_metric_name=Shared_options["calib_metric_name"]
    )

    analysis.model_selection(
        model_id=("RRSexDAdv"),
        index_column_names=['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda', "BT"],
        save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}SexDADV_df.pkl",
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
        calib_metric_name=Shared_options["calib_metric_name"]
    )


    analysis.model_selection(
        model_id=("RRFCLSex"),
        index_column_names=["fcl_lambda_g", "fcl_lambda_y"],
        save_path=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}SexFCL_df.pkl",
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
        calib_metric_name=Shared_options["calib_metric_name"]
    )

    EG_results, EG_calib_results = analysis.retrive_results(f'{Shared_options["dataset"]}Sex', log_dir=Shared_options["results_dir"], do_calib_eval=Shared_options["do_calib_eval"])

    #pareto = False
    pareto = True
    #selection_criterion = None
    selection_criterion = "DTO"
    #selection_criterion = "fairness"
    #selection_criterion = "performance"
    #calib_selection_criterion=Shared_options["calib_selection_criterion"]
    #calib_selection_criterion=None
    EG_main_results, EG_calib_main_results = analysis.final_results_df(
        results_dict=EG_results,
        pareto=pareto,
        pareto_selection="test",
        selection_criterion=selection_criterion,
        return_dev=True,
        return_conf=True,
        Fairness_threshold=-10,
        do_calib_eval=Shared_options["do_calib_eval"],
        calib_results_dict=EG_calib_results,
        calib_metric_name=Shared_options["calib_metric_name"])

    EG_main_results.to_csv(os.path.join(Shared_options["results_dir"], Shared_options["project_dir"], "RRSex_results.csv"))
    EG_calib_main_results.to_csv(os.path.join(Shared_options["results_dir"], Shared_options["project_dir"], "RRSex_calib_results.csv"))

    aurc_raw_out =True
    if aurc_raw_out:
        #assert (not pareto) and (selection_criterion is None) and (calib_selection_criterion == "performance")
        with open(calib_raw_out_f, "w") as fh_out:
            fh_out.write("model\tcriterion\ttopic\tcoverage\trisk\n")
            for model, opt_dir in zip(EG_calib_main_results["Models"], EG_calib_main_results["opt_dir list"]):
                _model = shorten_debias_name(model.replace("Sex", ""))
                selected_row = EG_calib_results[model][EG_calib_results[model]["opt_dir"] == opt_dir[0]]
                #all_results = selected_row.dev_aurc_raw.to_list()[0]
                all_results = selected_row.test_aurc_raw.to_list()[0]
                for coverage, risk in all_results:
                    fh_out.write("{}\t{}\t{}\t{}\t{}\n".format(_model, task, "all", coverage, risk))
                #per_private_label_results = selected_row.dev_per_private_label.to_list()[0]
                per_private_label_results = selected_row.test_per_private_label.to_list()[0]
                for protected_label, results in per_private_label_results.items():
                    if not results:
                        continue
                    for coverage, risk in results["aurc_raw"]:
                        fh_out.write("{}\t{}\t{}\t{}\t{}\n".format(_model, task, protected_label, coverage, risk))

    perf_raw_out =True
    if perf_raw_out:
        write_raw_out_rr(EG_calib_results, EG_calib_main_results, task, "sex", out_f=perf_raw_out_f)


    analysis.tables_and_figures.make_plot(
        EG_main_results,
        figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/plot_sex.png",
        performance_name=Shared_options["Performance_metric_name"]
    )
    EG_calib_main_results["test_fairness mean"] = EG_main_results["test_fairness mean"]
    analysis.tables_and_figures.make_plot(
        EG_calib_main_results,
        figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/plot_calib_sex.png",
        performance_name=Shared_options["calib_metric_name"]
    )

    # fairlib.analysis.tables_and_figures.make_zoom_plot(
    #    EG_main_results, figure_name=f"{Shared_options['results_dir']}/{Shared_options['project_dir']}/{Shared_options['dataset']}/zoom_plot.png", performance_name=Shared_options["Performance_metric_name"], dpi = 100,
    #    zoom_xlim=(0.6, 0.78),
    #    zoom_ylim=(0.8, 0.98)
    #    )
    print(EG_main_results)
    print(EG_calib_main_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-results_dir', default="/home/simon/Apps/fairlib/tutorial/", help='Use your path to .../fairlib/tutorial/')
    parser.add_argument('-perf_raw_out_f', default="/home/simon/Apps/SysRevData/data/modelling/plots/robotreviewer/robotreviewer_perf_raw_RoB_sex_debias.csv")
    parser.add_argument('-calib_raw_out_f', default="/home/simon/Apps/SysRevData/data/modelling/plots/robotreviewer/robotreviewer_aurc_raw_RoB_debias_sex.csv")
    argums = parser.parse_args()

    main(argums.results_dir, argums.calib_raw_out_f, argums.perf_raw_out_f)
