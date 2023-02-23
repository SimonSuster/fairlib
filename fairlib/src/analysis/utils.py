import os
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import stats
from sklearn.metrics import f1_score
from sysrev.modelling.allennlp.analyse import get_prob_high_idx, get_entire_band, get_risk_coverage_auc_per_band, \
    calculate_mce, calculate_ece, calculate_brier
from sysrev.modelling.allennlp.util_stat import gap_eval_scores
from yaml.loader import SafeLoader

from fairlib.src.dataloaders.loaders.EGBinaryGrade import INV_PROTECTED_LABEL_MAP_AREA
from fairlib.src.dataloaders.loaders.RoB import get_protected_group, get_protected_label_map, \
    get_inv_protected_label_map

warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd


# import seaborn as sns


def mkdirs(paths):
    """make a set of new directories

    Args:
        paths (list): a list of directories
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """make a new directory

    Args:
        path (str): path to the directory
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def power_mean(series, p, axis=0):
    """calculate the generalized mean

    Args:
        series (np.array): a array of numbers.
        p (int): power of the generalized mean.
        axis (int, optional): axis to the aggregation. Defaults to 0.

    Returns:
        np.array: generalized mean of the inputs
    """
    if p > 50:
        return np.max(series, axis=axis)
    elif p < -50:
        return np.min(series, axis=axis)
    else:
        total = np.mean(np.power(series, p), axis=axis)
        return np.power(total, 1 / p)


def l2norm(matrix_1, matrix_2):
    """calculate Euclidean distance

    Args:
        matrix_1 (n*d np array): n is the number of instances, d is num of metric
        matrix_2 (n*d np array): same as matrix_1

    Returns:
        float: the row-wise Euclidean distance 
    """
    return np.power(np.sum(np.power(matrix_1 - matrix_2, 2), axis=1), 0.5)


def DTO(fairness_metric, performacne_metric, calibration_metric=None, utopia_fairness=None, utopia_performance=None):
    """calculate DTO for each condidate model

    Args:
        fairness_metric (List): fairness evaluation results (1-GAP)
        performacne_metric (List): performance evaluation results
        calibration_metric (List): calibration evaluation results, optional
    """

    fairness_metric, performacne_metric = np.array(fairness_metric), np.array(performacne_metric)
    if calibration_metric is not None:
        calibration_metric = np.array(calibration_metric)

    # Best metric
    if (utopia_performance is None):
        utopia_performance = np.max(performacne_metric)
    if (utopia_fairness is None):
        utopia_fairness = np.max(fairness_metric)

    # Normalize
    performacne_metric = performacne_metric / utopia_performance
    fairness_metric = fairness_metric / utopia_fairness
    if calibration_metric is not None:
        calibration_metric = calibration_metric / utopia_fairness

    # Reshape and concatnate
    performacne_metric = performacne_metric.reshape(-1, 1)
    fairness_metric = fairness_metric.reshape(-1, 1)
    if calibration_metric is not None:
        calibration_metric = calibration_metric.reshape(-1, 1)
        normalized_metric = np.concatenate([performacne_metric, fairness_metric, calibration_metric], axis=1)
    else:
        normalized_metric = np.concatenate([performacne_metric, fairness_metric], axis=1)

    # Calculate Euclidean distance
    return l2norm(normalized_metric, np.ones_like(normalized_metric))


# find folders for each run
def get_dir(results_dir, project_dir, checkpoint_dir, checkpoint_name, model_id):
    """retrive logs for experiments

    Args:
        results_dir (str): dir to the saved experimental results
        project_dir (str): experiment type identifier, e.g., final, hypertune, dev. Same as the arguments.
        checkpoint_dir (str): dir to checkpoints, `models` by default.
        checkpoint_name (str):  checkpoint_epoch{num_epoch}.ptr.gz
        model_id (str): read all experiment start with the same model_id. E.g., "Adv" when tuning hyperparameters for standard adversarial

    Returns:
        list: a list of dictionaries, where each dict contains the information for a experiment.
    """
    results_dir = Path(results_dir)
    project_dir = Path(project_dir)
    checkpoint_dir = Path(checkpoint_dir)

    exps = []
    for root, dirs, files in os.walk(results_dir / project_dir):
        # for file in files:
        # if file.endswith(".txt"):
        # print(os.path.join(root, file))
        for dir in dirs:
            if dir.startswith(model_id):
                _dirs = Path(os.path.join(root, dir))

                # Open the file and load the file
                with open(_dirs / 'opt.yaml') as f:
                    _opt = yaml.load(f, Loader=SafeLoader)

                _checkpoints_dirs = []
                for root2, dirs2, files2 in os.walk(_dirs / checkpoint_dir):
                    for file2 in files2:
                        if file2.startswith(checkpoint_name):
                            _dirs2 = os.path.join(root2, file2)
                            _checkpoints_dirs.append(_dirs2)
                exps.append(
                    {
                        "opt": _opt,
                        "opt_dir": str(os.path.join(root, dir, 'opt.yaml')),
                        "dir": _checkpoints_dirs,
                    }
                )
    return exps


def get_calib_scores_folds(results_per_fold):
    raw_data = {}
    calib_results = {}

    for fold_n, d in results_per_fold.items():
        for method, df in d["EG_calib_results"].items():
            method = method[3:]
            if method not in raw_data:
                raw_data[method] = {"probs": [], "labels": [], "preds": [], "private_labels": [], "vocab_f": None}
            raw_data[method]["probs"].extend(df["test_test_probs"].to_list()[0])
            raw_data[method]["labels"].extend(df["test_test_labels"].to_list()[0])
            raw_data[method]["preds"].extend(df["test_test_predictions"].to_list()[0])
            raw_data[method]["private_labels"].extend(df["test_test_private_labels"].to_list()[0])
            if raw_data[method]["vocab_f"] is None:
                raw_data[method]["vocab_f"] = df["test_vocab_f"].to_list()[0]
            else:
                assert raw_data[method]["vocab_f"] == df["test_vocab_f"].to_list()[0]

    for method, d in raw_data.items():
        results = calib_eval(d["probs"], d["labels"], d["preds"], d["private_labels"], d["vocab_f"], None, "area")
        _, _, perf_results_per_group = gap_eval_scores(d["preds"], d["labels"], d["private_labels"])
        perf_results_per_group = {INV_PROTECTED_LABEL_MAP_AREA[k]: v for k, v in perf_results_per_group.items() if k != "overall"}
        for g, d in results["per_private_label"].items():
            d.update(perf_results_per_group[g])
        for metric in ["ece", "mce", "aurc", "brier"]:
            results[f"{metric}_pos"] = abs(results[metric] - 1)
        calib_results[method] = results

    return calib_results


def calib_eval(probs, labels, preds, private_labels, vocab_f, dataset_id, protected_group, per_private_label=False):
    results = {}
    inv_protected_label_map = get_inv_protected_label_map(protected_group)

    # get index of the higher-quality-evidence class
    if vocab_f is None:
        prob_high_idx = 1
    else:
        try:
            prob_high_idx, _ = get_prob_high_idx(f"{vocab_f}/labels.txt")
        except FileNotFoundError:
            prob_high_idx, _ = get_prob_high_idx("/home/simon/Apps/SysRevData/data/modelling/saved/bi_class2_nonaugm_all/bi_class2_nonaugm_all_0/vocabulary/labels.txt")
    pos_golds = np.array(labels) == prob_high_idx
    pos_preds = np.array(preds) == prob_high_idx
    # probabilities that an instance is of higher-quality
    probs_high = [p[prob_high_idx] for p in probs]

    gap = gap_eval_scores(preds, labels, private_labels)
    results["Fairness"] = 1 - gap[0]["TPR_GAP"]
    results["macro_fscore"] = f1_score(labels, preds, average="macro")

    accs, confs, sizes = get_entire_band(probs_high, pos_golds, outfile=None)

    results["ece"] = calculate_ece(accs, confs, sizes)
    results["mce"] = calculate_mce(accs, confs)
    results["brier"] = calculate_brier(probs_high, labels, to_onehot=False)

    aurc_raw = []  # for plotting risk-coverage curves

    aurc, coverages, risks = get_risk_coverage_auc_per_band(probs_high, pos_golds, pos_preds, band="entire")
    for i in range(len(coverages)):
        aurc_raw.append((coverages[i], risks[i]))
    results["aurc"] = aurc
    results["aurc_raw"] = aurc_raw

    results["per_private_label"] = {}

    for private_label_idx in set(private_labels):
        subset_idxs = np.array(private_labels) == private_label_idx
        probs_high_subset = np.array(probs_high)[subset_idxs]
        pos_golds_subset = pos_golds[subset_idxs]
        labels_subset = np.array(labels)[subset_idxs]
        pos_preds_subset = pos_preds[subset_idxs]

        private_label = inv_protected_label_map[private_label_idx]
        if private_label not in results["per_private_label"]:
            results["per_private_label"][private_label] = {}

        try:
            accs, confs, sizes = get_entire_band(probs_high_subset, pos_golds_subset, outfile=None)
        except IndexError:
            continue
        ece = calculate_ece(accs, confs, sizes)
        mce = calculate_mce(accs, confs)
        brier = calculate_brier(probs_high_subset, labels_subset, to_onehot=False)

        try:
            aurc, coverages, risks = get_risk_coverage_auc_per_band(probs_high_subset, pos_golds_subset, pos_preds_subset, band="entire")
        except ValueError:
            continue

        results["per_private_label"][private_label]["ece"] = ece
        results["per_private_label"][private_label]["mce"] = mce
        results["per_private_label"][private_label]["brier"] = brier
        results["per_private_label"][private_label]["aurc"] = aurc

        aurc_raw = []  # for plotting risk-coverage curves
        for i in range(len(coverages)):
            aurc_raw.append((coverages[i], risks[i]))
        results["per_private_label"][private_label]["aurc_raw"] = aurc_raw

    return results


def get_calib_scores(exp):
    """
    Args:
        exp (str): get_dir output, includeing the options and path to checkpoints

    Returns:
        pd.DataFrame: a pandas df including dev and test calibration scores for each epoch
    """
    epoch_id = []
    epoch_scores_dev = {"ece": [], "brier": [], "mce": [], "aurc": [], "aurc_raw": [], "per_private_label": []}
    epoch_scores_test = {"ece": [], "brier": [], "mce": [], "aurc": [], "aurc_raw": [], "per_private_label": [],
                         "test_probs": [], "test_labels": [], "test_predictions": [], "test_private_labels": [],
                         "vocab_f": []}

    protected_group = get_protected_group(exp["opt"]["data_dir"])

    for epoch_result_dir in exp["dir"]:
        epoch_result = torch.load(epoch_result_dir)

        # Calculate calibration metrics
        if "dev_probs" not in epoch_result:
            print()
            continue

        # Track the epoch id
        epoch_id.append(epoch_result["epoch"])

        vocab_f = exp["opt"].get("vocabulary_dir", None)
        dev_calib_results = calib_eval(probs=epoch_result["dev_probs"],
                                       labels=epoch_result["dev_labels"],
                                       preds=epoch_result["dev_predictions"],
                                       private_labels=epoch_result["dev_private_labels"],
                                       vocab_f=vocab_f,
                                       dataset_id=exp["opt"]["dataset"],
                                       protected_group=protected_group)

        epoch_scores_dev["ece"].append(dev_calib_results["ece"])
        epoch_scores_dev["brier"].append(dev_calib_results["brier"])
        epoch_scores_dev["mce"].append(dev_calib_results["mce"])
        epoch_scores_dev["aurc"].append(dev_calib_results["aurc"])
        epoch_scores_dev["aurc_raw"].append(dev_calib_results["aurc_raw"])
        epoch_scores_dev["per_private_label"].append(dev_calib_results["per_private_label"])

        test_calib_results = calib_eval(probs=epoch_result["test_probs"],
                                        labels=epoch_result["test_labels"],
                                        preds=epoch_result["test_predictions"],
                                        private_labels=epoch_result["test_private_labels"],
                                        vocab_f=vocab_f,
                                        dataset_id=exp["opt"]["dataset"],
                                        protected_group=protected_group)
        epoch_scores_test["ece"].append(test_calib_results["ece"])
        epoch_scores_test["brier"].append(test_calib_results["brier"])
        epoch_scores_test["mce"].append(test_calib_results["mce"])
        epoch_scores_test["aurc"].append(test_calib_results["aurc"])
        epoch_scores_test["aurc_raw"].append(test_calib_results["aurc_raw"])
        epoch_scores_test["per_private_label"].append(test_calib_results["per_private_label"])
        # these are later concat'd across all folds for computing risk-coverage plots:
        epoch_scores_test["test_probs"].append(epoch_result["test_probs"])
        epoch_scores_test["test_labels"].append(epoch_result["test_labels"])
        epoch_scores_test["test_predictions"].append(epoch_result["test_predictions"])
        epoch_scores_test["test_private_labels"].append(epoch_result["test_private_labels"])
        epoch_scores_test["vocab_f"].append(vocab_f)

    epoch_results_dict = {"epoch": epoch_id}

    for _dev_metric_keys in epoch_scores_dev.keys():
        epoch_results_dict["dev_{}".format(_dev_metric_keys)] = epoch_scores_dev[_dev_metric_keys]
    for _test_metric_keys in epoch_scores_test.keys():
        epoch_results_dict["test_{}".format(_test_metric_keys)] = epoch_scores_test[_test_metric_keys]

    # include positive interpretation for ECE, MCE, AURC and Brier
    for split in ["dev", "test"]:
        for metric in ["ece", "mce", "aurc", "brier"]:
            epoch_results_dict[f"{split}_{metric}_pos"] = []
            for score_per_epoch in epoch_results_dict[f"{split}_{metric}"]:
                epoch_results_dict[f"{split}_{metric}_pos"].append(abs(score_per_epoch-1))

    epoch_scores = pd.DataFrame(epoch_results_dict)

    return epoch_scores


def get_model_scores(exp, GAP_metric, Performance_metric, keep_original_metrics=False):
    """given the log path for a exp, read log and return the dev&test performacne, fairness, and DTO

    Args:
        exp (str): get_dir output, includeing the options and path to checkpoints
        GAP_metric (str): the target GAP metric name
        Performance_metric (str): the target performance metric name, e.g., F1, Acc.

    Returns:
        pd.DataFrame: a pandas df including dev and test scores for each epoch
    """

    epoch_id = []
    epoch_scores_dev = {"performance": [], "fairness": []}
    epoch_scores_test = {"performance": [], "fairness": []}
    for epoch_result_dir in exp["dir"]:
        #print(epoch_result_dir)
        epoch_result = torch.load(epoch_result_dir)

        # Track the epoch id
        epoch_id.append(epoch_result["epoch"])

        # Get fairness evaluation scores, 1-GAP, the larger the better
        epoch_scores_dev["fairness"].append(1 - epoch_result["dev_evaluations"][GAP_metric])
        epoch_scores_test["fairness"].append(1 - epoch_result["test_evaluations"][GAP_metric])

        epoch_scores_dev["performance"].append(epoch_result["dev_evaluations"][Performance_metric])
        epoch_scores_test["performance"].append(epoch_result["test_evaluations"][Performance_metric])

        if keep_original_metrics:
            for _dev_keys in epoch_result["dev_evaluations"].keys():
                epoch_scores_dev[_dev_keys] = (
                        epoch_scores_dev.get(_dev_keys, []) + [epoch_result["dev_evaluations"][_dev_keys]])

            for _test_keys in epoch_result["test_evaluations"].keys():
                epoch_scores_test[_test_keys] = (
                        epoch_scores_test.get(_test_keys, []) + [epoch_result["test_evaluations"][_test_keys]])

    # Calculate the DTO for dev and test 
    dev_DTO = DTO(fairness_metric=epoch_scores_dev["fairness"], performacne_metric=epoch_scores_dev["performance"])
    test_DTO = DTO(fairness_metric=epoch_scores_test["fairness"], performacne_metric=epoch_scores_test["performance"])

    epoch_results_dict = {
        "epoch": epoch_id,
        "dev_DTO": dev_DTO,
        # "test_fairness":epoch_scores_test["fairness"],
        # "test_performance":epoch_scores_test["performance"],
        "test_DTO": test_DTO,
    }

    for _dev_metric_keys in epoch_scores_dev.keys():
        epoch_results_dict["dev_{}".format(_dev_metric_keys)] = epoch_scores_dev[_dev_metric_keys]
    for _test_metric_keys in epoch_scores_test.keys():
        epoch_results_dict["test_{}".format(_test_metric_keys)] = epoch_scores_test[_test_metric_keys]

    epoch_scores = pd.DataFrame(epoch_results_dict)

    return epoch_scores


def retrive_all_exp_results(exp, GAP_metric_name, Performance_metric_name, index_column_names,
                            keep_original_metrics=False):
    """retrive experimental results according to the input list of experiments.

    Args:
        exp (list): a list of experiment info.
        GAP_metric_name (str): gap metric name, such as TPR_GAP.
        Performance_metric_name (str): performance metric name, such as accuracy.
        index_column_names (_type_): a list of hyperparameters that will be used to differentiate runs within a single method.
        keep_original_metrics (bool): whether or not keep all experimental results in the checkpoint. Default to False. 

    Returns:
        dict: retrived results.
    """
    # Get scores
    epoch_scores = get_model_scores(
        exp=exp, GAP_metric=GAP_metric_name,
        Performance_metric=Performance_metric_name,
        keep_original_metrics=keep_original_metrics,
    )
    _exp_results = epoch_scores

    _exp_opt = exp["opt"]
    # Get hyperparameters for this epoch
    for hyperparam_key in index_column_names:
        _exp_results[hyperparam_key] = [_exp_opt[hyperparam_key]] * len(_exp_results)

    _exp_results["opt_dir"] = [exp["opt_dir"]] * len(_exp_results)

    return _exp_results


def retrive_exp_results(
        exp, GAP_metric_name, Performance_metric_name,
        selection_criterion, index_column_names, keep_original_metrics=False, calib_metric_name=None, calib_selection_criterion=None, do_calib_eval=False):
    """Retrive experimental results of a epoch from the saved checkpoint.

    Args:
        exp (_type_): _description_
        GAP_metric_name (_type_): _description_
        Performance_metric_name (_type_): _description_
        selection_criterion (_type_): _description_
        index_column_names (_type_): _description_
        keep_original_metrics (bool, optional): besides selected performance and fairness, show original metrics. Defaults to False.
        calib_metric_name (_type_, optional): _description_
        calib_selection_criterion (_type_, optional): _description_
        do_calib_eval (bool, optional): whether to perform calibration evaluation

    Returns:
        dict: retrived results.
    """

    # Get scores
    epoch_scores = get_model_scores(
        exp=exp, GAP_metric=GAP_metric_name,
        Performance_metric=Performance_metric_name,
        keep_original_metrics=keep_original_metrics,
    )

    # selection for predictive performance
    if selection_criterion == "DTO":
        selected_epoch_id = np.argmin(epoch_scores["dev_{}".format(selection_criterion)])
    else:
        selected_epoch_id = np.argmax(epoch_scores["dev_{}".format(selection_criterion)])
    selected_epoch_scores = epoch_scores.iloc[selected_epoch_id]

    _exp_opt = exp["opt"]

    # Get hyperparameters for this epoch
    _exp_results = {}
    for hyperparam_key in index_column_names:
        _exp_results[hyperparam_key] = _exp_opt[hyperparam_key]

    # Merge opt with scores
    for key in selected_epoch_scores.keys():
        _exp_results[key] = selected_epoch_scores[key]

    _exp_results["opt_dir"] = exp["opt_dir"]
    _exp_results["epoch"] = selected_epoch_id

    if do_calib_eval:
        epoch_calib_scores = get_calib_scores(exp)

        _test_selected_score = epoch_calib_scores[f"test_{calib_metric_name}"]

        if calib_metric_name in ["ece", "mce", "aurc", "brier"]:
            # select positive interpretation of the lower-is-better metric
            calib_metric_name_select = f"{calib_metric_name}_pos"
        else:
            calib_metric_name_select = calib_metric_name

        dev_selected_score = epoch_calib_scores[f"dev_{calib_metric_name_select}"]
        test_selected_score = epoch_calib_scores[f"test_{calib_metric_name_select}"]

        if len(epoch_scores.dev_fairness) != len(dev_selected_score):
            dev_fairness = epoch_scores.dev_fairness[epoch_calib_scores.epoch]
        else:
            dev_fairness = epoch_scores.dev_fairness
        if len(epoch_scores.test_fairness) != len(test_selected_score):
            test_fairness = epoch_scores.test_fairness[epoch_calib_scores.epoch]
        else:
            test_fairness = epoch_scores.test_fairness

        dev_calib_DTO = DTO(fairness_metric=dev_fairness, performacne_metric=dev_selected_score)
        test_calib_DTO = DTO(fairness_metric=test_fairness, performacne_metric=test_selected_score)

        # selection for calibration performance
        if calib_selection_criterion == "DTO":
            calib_selected_epoch_id = np.argmin(dev_calib_DTO)
        elif calib_selection_criterion in ["ece", "mce", "aurc", "brier", "fairness", "performance"]:
            # we use positive-interpretation of all these metrics, so just argmax
            calib_selected_epoch_id = np.argmax(dev_selected_score)
        else:
            raise NotImplementedError

        calib_selected_epoch_scores = epoch_calib_scores.iloc[calib_selected_epoch_id]
        # this is potentially different from selected_epoch_scores above as we are selecting based on best calib epoch
        # need this to get fairness result to include in the calib results
        _selected_epoch_scores = epoch_scores.iloc[calib_selected_epoch_id]
        # Get hyperparameters for this epoch
        _calib_exp_results = {}
        for hyperparam_key in index_column_names:
            _calib_exp_results[hyperparam_key] = _exp_opt[hyperparam_key]

        # Merge opt with scores
        for key in calib_selected_epoch_scores.keys():
            _calib_exp_results[key] = calib_selected_epoch_scores[key]
        for key in ["dev_fairness", "test_fairness", "dev_performance", "test_performance"]:
            _calib_exp_results[key] = _selected_epoch_scores[key]

        _calib_exp_results["opt_dir"] = exp["opt_dir"]
        _calib_exp_results["epoch"] = calib_selected_epoch_id

    return (_exp_results, _calib_exp_results if do_calib_eval else None)


def is_pareto_efficient(costs, return_mask=True):
    """Find the pareto-efficient points

    If return_mask is True, this will be an (n_points, ) boolean array
    Otherwise it will be a (n_efficient_points, ) integer array of indices.

    Args:
        costs (np.array): An (n_points, n_costs) array
        return_mask (bool, optional): True to return a mask. Defaults to True.

    Returns:
        np.array: An array of indices of pareto-efficient points.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


def auc_performance_fairness_tradeoff(
        pareto_df,
        random_performance=0,
        pareto_selection="test",
        fairness_metric_name="fairness",
        performance_metric_name="performance",
        interpolation="linear",
        performance_threshold=None,
        normalization=False,
):
    """calculate the area under the performance--fairness trade-off curve.

    Args:
        pareto_df (DataFrame): A data frame of pareto frontiers
        random_performance (float, optional): the lowest performance, which leads to the 1 fairness. Defaults to 0.
        pareto_selection (str, optional): which split is used to select the frontiers. Defaults to "test".
        fairness_metric_name (str, optional):. the metric name for fairness evaluation. Defaults to "fairness".
        performance_metric_name (str, optional): the metric name for performance evaluation. Defaults to "performance".
        interpolation (str, optional): interpolation method for the threshold fairness. Defaults to "linear".
        performance_threshold (float, optional): the performance threshold for the method. Defaults to None.
        normalization (bool, optional): if normalize the auc score with the maximum auc that can be achieved. Defaults to False.

    Returns:
        tuple: (AUC score, AUC DataFrame)
    """
    fairness_col_name = "{}_{} mean".format(pareto_selection, fairness_metric_name)
    performance_col_name = "{}_{} mean".format(pareto_selection, performance_metric_name)

    # Filter the df with only performance and fairness scores
    results_df = pareto_df[[fairness_col_name, performance_col_name]]

    # Add the worst performed model
    results_df = results_df.append({
        fairness_col_name: 1,
        performance_col_name: random_performance,
    }, ignore_index=True)

    sorted_results_df = results_df.sort_values(by=[fairness_col_name])

    if performance_threshold is not None:
        if performance_threshold > sorted_results_df.values[0][1]:
            return 0, None
        if performance_threshold < sorted_results_df.values[-1][1]:
            performance_threshold = sorted_results_df.values[-1][1]

        # Find the closest performed points to the threshold
        closest_worser_performed_point = \
            sorted_results_df[sorted_results_df[performance_col_name] <= performance_threshold].values[0]
        closest_better_performed_point = \
            sorted_results_df[sorted_results_df[performance_col_name] >= performance_threshold].values[-1]

        # Interpolation
        assert interpolation in ["linear", "constant"]
        if interpolation == "constant":
            if (performance_threshold - closest_worser_performed_point[1]) <= (
                    closest_better_performed_point[1] - performance_threshold):
                interpolation_fairness = closest_worser_performed_point[0]
            else:
                interpolation_fairness = closest_better_performed_point[0]
        elif interpolation == "linear":
            _ya, _xa = closest_worser_performed_point[0], closest_worser_performed_point[1]
            _yb, _xb = closest_better_performed_point[0], closest_better_performed_point[1]
            interpolation_fairness = _ya + (_yb - _ya) * ((performance_threshold - _xa) / (_xb - _xa))

        interpolated_point = {
            fairness_col_name: interpolation_fairness,
            performance_col_name: performance_threshold,
        }

        sorted_results_df = sorted_results_df[sorted_results_df[performance_col_name] >= performance_threshold]
        sorted_results_df = sorted_results_df.append(
            interpolated_point, ignore_index=True,
        )

    filtered_curve = sorted_results_df.sort_values(by=[performance_col_name])
    auc_filtered_curve = np.trapz(
        filtered_curve[fairness_col_name],
        x=filtered_curve[performance_col_name], )

    # Calculate the normalization term
    if normalization:
        normalization_term = (1 - min(filtered_curve[performance_col_name]))
        auc_filtered_curve = auc_filtered_curve / normalization_term

    return auc_filtered_curve, filtered_curve


def analyse_increases(aurc_raw):
    n_increases = 0
    sum_increases = 0
    old_err = None
    for th, err in aurc_raw:
        if old_err is None:
            old_err = err
        else:
            if err > old_err:
                n_increases += 1
                sum_increases += err - old_err

    return n_increases, sum_increases


def analyse_worse_than_full_coverage(aurc_raw):
    """
    is there, at any reduced coverage, an error rate that is higher than that at full coverage?
    """
    errs = np.array([err for th, err in aurc_raw])

    return np.max(errs) > errs[0]


def get_stats_aurcs(results, fun=stats.describe):
    out = {}
    for method, d in results.items():
        out[method] = fun([xd["aurc"] for area, xd in d["per_private_label"].items()])

    return out


def analyse_selective_classification(results):
    out_n_increases, out_sum_increases = {}, {}
    out_worse_than_full_coverage = {}

    stats_aurcs = get_stats_aurcs(results)
    median_aurcs = get_stats_aurcs(results, fun=np.median)

    for model, d in results.items():
        # general (all):
        n_increases, sum_increases = analyse_increases(d["aurc_raw"])
        worse_than_full_coverage = analyse_worse_than_full_coverage(d["aurc_raw"])
        if model not in out_n_increases:
            out_n_increases[model] = {}
        if model not in out_sum_increases:
            out_sum_increases[model] = {}
        if model not in out_worse_than_full_coverage:
            out_worse_than_full_coverage[model] = {}
        out_n_increases[model]["all"] = n_increases
        out_sum_increases[model]["all"] = sum_increases
        out_worse_than_full_coverage[model]["all"] = worse_than_full_coverage

        per_private_label_results = d["per_private_label"]
        for protected_label, xd in per_private_label_results.items():
            if not xd:
                continue
            n_increases, sum_increases = analyse_increases(xd["aurc_raw"])
            worse_than_full_coverage = analyse_worse_than_full_coverage(xd["aurc_raw"])

            out_n_increases[model][protected_label] = n_increases
            out_sum_increases[model][protected_label] = sum_increases
            out_worse_than_full_coverage[model][protected_label] = worse_than_full_coverage

    return {"out_n_increases": out_n_increases,
            "total_n_increases": {model: [sum(d.values())] for model, d in out_n_increases.items()},
            "out_sum_increases": out_sum_increases,
            "total_sum_increases": {model: [sum(d.values())] for model, d in out_sum_increases.items()},
            "out_worse_than_full_coverage": {model: sum(d.values()) for model, d in out_worse_than_full_coverage.items()},
            "stats_aurcs": (stats_aurcs, {model: np.sqrt(stats.variance) for model, stats in stats_aurcs.items()}),
            "median_aurcs": median_aurcs
            }
