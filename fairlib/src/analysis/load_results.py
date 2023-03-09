import pickle
import warnings

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import get_dir, retrive_exp_results, retrive_all_exp_results

warnings.simplefilter(action='ignore', category=FutureWarning)


def model_selection_parallel(
        results_dir,
        project_dir,
        model_id,
        GAP_metric_name,
        Performance_metric_name,
        selection_criterion,
        checkpoint_dir="models",
        checkpoint_name="checkpoint_epoch",
        index_column_names=['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda'],
        n_jobs=20,
        save_path=None,
        return_all=False,
        keep_original_metrics=False,
        calib_metric_name="ece",
        do_calib_eval=False
):
    """perform model selection over different runs wrt different hyperparameters

    Args:
        results_dir (str): dir to the saved experimental results
        project_dir (str): experiment type identifier, e.g., final, hypertune, dev. Same as the arguments.
        checkpoint_dir (str): dir to checkpoints, `models` by default.
        checkpoint_name (str):  checkpoint_epoch{num_epoch}.ptr.gz
        model_id (str): read all experiment start with the same model_id. E.g., "Adv" when tuning hyperparameters for standard adversarial
        GAP_metric_name (str): fairness metric in the log
        Performance_metric_name (str): performance metric name in the log
        selection_criterion (str): {GAP_metric_name | Performance_metric_name | "DTO"}
        index_column_names (list): tuned hyperparameters, ['adv_lambda', 'adv_num_subDiscriminator', 'adv_diverse_lambda'] by default.
        n_jobs (nonnegative int): 0 for non-parallel, positive integer refers to the number of parallel processes
        calib_metric_name (str): calibration performance metric to use when do_calib_eval is True
        do_calib_eval (bool, optional): whether to perform calibration evaluation
    Returns:
        pd.DataFrame: loaded results
    """
    if save_path is not None:
        try:
            return pd.read_pickle(save_path)
        except:
            pass

    exps = get_dir(
        results_dir=results_dir,
        project_dir=project_dir,
        checkpoint_dir=checkpoint_dir,
        checkpoint_name=checkpoint_name,
        model_id=model_id)

    exp_results = []
    calib_exp_results = []

    if return_all:
        for exp in tqdm(exps):
            _exp_results = retrive_all_exp_results(exp, GAP_metric_name, Performance_metric_name, index_column_names,
                                                   keep_original_metrics)
            exp_results.append(_exp_results)

        result_df = pd.concat(exp_results)
        result_df["index_epoch"] = result_df["epoch"].copy()
        result_df = result_df.set_index(index_column_names + ["index_epoch"])
    else:
        if n_jobs == 0:
            for exp in tqdm(exps):
                # Get scores
                _exp_results, _calib_exp_results = retrive_exp_results(exp, GAP_metric_name, Performance_metric_name, selection_criterion,
                                                   index_column_names, keep_original_metrics,
                                                   calib_metric_name=calib_metric_name,
                                                   do_calib_eval=do_calib_eval)

                exp_results.append(_exp_results)
                calib_exp_results.append(_calib_exp_results)
        else:
            exp_results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(retrive_exp_results)
                                                             (exp, GAP_metric_name, Performance_metric_name,
                                                              selection_criterion, index_column_names,
                                                              keep_original_metrics,
                                                              calib_metric_name=calib_metric_name,
                                                              do_calib_eval=do_calib_eval)
                                                             for exp in exps)
            # separate pred-perf and calib-perf results
            calib_exp_results = [i[1] for i in exp_results]
            exp_results = [i[0] for i in exp_results]

        result_df = pd.DataFrame(exp_results).set_index(index_column_names)
        if do_calib_eval:
            calib_result_df = pd.DataFrame(calib_exp_results).set_index(index_column_names)
        else:
            calib_result_df = None

    if save_path is not None:
        if do_calib_eval:
            all_result_df = {"result": result_df, "calib_result": calib_result_df}
            pickle.dump(all_result_df, open(save_path, "wb"))
        else:
            result_df.to_pickle(save_path)

    return (result_df, calib_result_df)
