import csv
import difflib
import json
import os
import random
from collections import OrderedDict, Counter
from os import makedirs
from os.path import exists, join, realpath

import numpy as np
import pandas as pd
import torch
from sysrev.modelling.allennlp.util_stat import gap_eval_scores

from fairlib.src.dataloaders.utils import get_inv_protected_label_map


class PandasUtils:
    @staticmethod
    def load_csv(fname, dir_in, dtype=None):
        return pd.read_csv(realpath(join(dir_in, fname)), dtype=dtype)


class FileUtils:
    @staticmethod
    def write_json(obj_dict, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            json.dump(obj_dict, f)

    @staticmethod
    def read_json(fname, dir_in):
        with open(realpath(join(dir_in, fname))) as f:
            return json.load(f)

    @staticmethod
    def write_list(data_list, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            for term in data_list:
                f.write(term + '\n')

    @staticmethod
    def read_list(fname, dir_in):

        data = list()
        with open(realpath(join(dir_in, fname))) as f:
            for line in f:
                data.append(line.strip())

        return data

    @staticmethod
    def write_txt(string, fname, dir_out):

        if not exists(dir_out):
            makedirs(dir_out)

        with open(realpath(join(dir_out, fname)), 'w') as f:
            f.write(string)

    @staticmethod
    def write_numpy(array, fname, dir_in):
        fpath = realpath(join(dir_in, fname))
        np.save(fpath, array)

    @staticmethod
    def read_numpy(fname, dir_in):
        fpath = realpath(join(dir_in, fname))
        return np.load(fpath)


def get_top_items_dict(data_dict, k, order=False):
    """Get top k items in the dictionary by score.
    Returns a dictionary or an `OrderedDict` if `order` is true.
    """
    top = sorted(data_dict.items(), key=lambda x: x[1], reverse=True)[:k]
    if order:
        return OrderedDict(top)
    return dict(top)


def get_most_freq_items(items, k):
    """
    :param items: 2D list of items
    :param k: number of items to retain
    :return: set of most frequent items
    """
    # count the number of instances the term occurs in
    term2freq = Counter(x for xs in items for x in set(xs))
    reduced_set = dict(term2freq.most_common(k)).keys()  # frequency filter.
    return reduced_set


def log_grid(start, stop, number_trials):
    assert stop >= start
    step = (stop - start) / number_trials
    step_index = [i for i in range(number_trials + 1)]
    return np.power(10, np.array([start + i * step for i in step_index]))


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def diff_str(first, second):
    firstlines = first.splitlines(keepends=True)
    secondlines = second.splitlines(keepends=True)
    if len(firstlines) == 1 and first.strip('\r\n') == first:
        firstlines = [first + '\n']
        secondlines = [second + '\n']
    return ''.join(difflib.unified_diff(firstlines, secondlines))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


def get_file_list(topdir, identifiers=None, all_levels=False, ignore_hidden=False):
    """
    :param identifiers: a list of strings, any of which should be in the filename
    :param all_levels: get filenames recursively
    """
    if identifiers is None:
        identifiers = [""]
    filelist = []
    for root, dirs, files in os.walk(topdir):
        if not all_levels and (root != topdir):  # don't go deeper
            continue
        for filename in files:
            get = False
            for i in identifiers:
                if i in filename:
                    get = True
                if ignore_hidden and filename[0] == ".":
                    get = False
            if get:
                fullname = os.path.join(root, filename)
                filelist.append(fullname)

    return filelist


def get_dir_list(topdir, identifiers=None, all_levels=False, ignore_hidden=False):
    """
    :param identifiers: a list of strings, any of which should be in the filename
    :param all_levels: get filenames recursively
    """
    if identifiers is None:
        identifiers = [""]
    dirlist = []
    for root, dirs, files in os.walk(topdir):
        if not all_levels and (root != topdir):  # don't go deeper
            continue
        for dir in dirs:
            get = False
            for i in identifiers:
                if i in dir:
                    get = True
                if ignore_hidden and dir[0] == ".":
                    get = False
            if get:
                fullname = os.path.join(root, dir)
                dirlist.append(fullname)

    return dirlist


def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'), indent=2, sort_keys=True)


def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)


def read_csv(f, keys=None, delimiter=","):
    with open(f) as csvfile:
        if keys is not None:
            reader = csv.DictReader(csvfile, fieldnames=keys, delimiter=delimiter)
        else:
            reader = csv.DictReader(csvfile, delimiter=delimiter)
        for row in reader:
            yield row


def print_tsv(data_dist, inv_protected_label_map, protected_group, inv_label_mapping):
    counts = (data_dist["joint_dist"] * data_dist["N"]).astype("int").transpose()
    row_str = f"\t{inv_label_mapping[0]}\t{inv_label_mapping[1]}\tTotal"
    print(row_str)
    for g_idx, row in enumerate(counts):
        g_name = inv_protected_label_map[g_idx]
        if protected_group == "area":
            g_name = f"{g_name[:10]}..."
        row_str = f"{g_name}\t"
        row_str += "\t".join(row.astype(str))
        row_str += f"\t{sum(row)}"
        print(row_str)

    y_totals = (data_dist["y_dist"] * data_dist["N"]).astype("int")
    row_str = "Total\t" + "\t".join(y_totals.astype(str)) + f"\t{int(data_dist['N'])}"
    print(row_str)


def extract_name(s):
    if "vanilla" in s:
        return "vanilla"
    elif "Resampling" in s:
        return "Resampling"
    elif "Reweighting" in s:
        return "Reweighting"
    elif "Downsampling" in s:
        return "Downsampling"
    elif "DADV" in s:
        return "DAdv"
    elif "ADV" in s:
        return "Adv"
    elif "FCL" in s:
        return "FCL"
    else:
        raise KeyError


def csv_results_to_tex_table(f, f_calib):
    """
    :param f: csv file with results
    :param f_calib: csv file with calibration results
    """
    def pretty_str(i):
        if i > 0:
            return f"\\textcolor{{green}}{{+{i}}}"
        else:
            if i == 0:
                return f"\\textcolor{{red}}{{-{i}}}"
            else:
                return f"\\textcolor{{red}}{{{i}}}"

    r = read_csv(f)
    r_calib = read_csv(f_calib)
    vanilla_results = {}
    for row, row_calib in zip(r, r_calib):
        model_name = extract_name(row["Models"])
        if model_name == "vanilla":
            vanilla_results["test_performance mean"] = float(row["test_performance mean"])
            vanilla_results["test_fairness mean"] = float(row["test_fairness mean"])
            vanilla_results["test_aurc_pos mean"] = float(row_calib["test_aurc_pos mean"])
            vanilla_results["test_mce_pos mean"] = float(row_calib["test_mce_pos mean"])
            vanilla_results["test_ece_pos mean"] = float(row_calib["test_ece_pos mean"])

    r = read_csv(f)
    r_calib = read_csv(f_calib)
    d = {}
    for row, row_calib in zip(r, r_calib):
        model_name = extract_name(row["Models"])
        if model_name == "vanilla":
            test_performance = round(float(row["test_performance mean"]), 3)
            test_fairness = round(float(row["test_fairness mean"]), 3)
            test_aurc_pos = round(float(row_calib["test_aurc_pos mean"]), 3)
            test_mce_pos = round(float(row_calib["test_mce_pos mean"]), 3)
            test_ece_pos = round(float(row_calib["test_ece_pos mean"]), 3)
            d[model_name] = f'{model_name} & {test_performance} & {test_fairness} & {test_aurc_pos} & {test_mce_pos} & {test_ece_pos} \\\\'
            d[model_name] = d[model_name].replace("0.", ".")
        else:
            test_performance = round(float(row["test_performance mean"]) - vanilla_results["test_performance mean"], 3)
            test_fairness = round(float(row["test_fairness mean"]) - vanilla_results["test_fairness mean"], 3)
            test_aurc_pos = round(float(row_calib["test_aurc_pos mean"]) - vanilla_results["test_aurc_pos mean"], 3)
            test_mce_pos = round(float(row_calib["test_mce_pos mean"]) - vanilla_results["test_mce_pos mean"], 3)
            test_ece_pos = round(float(row_calib["test_ece_pos mean"]) - vanilla_results["test_ece_pos mean"], 3)
            d[model_name] = f'{model_name} & {pretty_str(test_performance)} & {pretty_str(test_fairness)} & {pretty_str(test_aurc_pos)} & {pretty_str(test_mce_pos)} & {pretty_str(test_ece_pos)} \\\\'
            d[model_name] = d[model_name].replace("0.", ".")
    for model_name in ["vanilla", "Downsampling", "Resampling", "Reweighting", "Adv", "DAdv", "FCL"]:
        print(d[model_name])
        if model_name == "vanilla":
            print("\\cmidrule(lr){1-6}")


def write_raw_out_rr(calib_results, calib_main_results, task, protected_attribute, out_f):
    inv_protected_label_map = get_inv_protected_label_map(protected_attribute)
    with open(out_f, "w") as fh_out:
        writer = csv.writer(fh_out)
        writer.writerow(("model", "criterion", "topic", "precision", "recall", "f_macro"))
        for model, d in calib_results.items():
            best_d = calib_results[model].loc[calib_results[model]["opt_dir"] ==
                                           calib_main_results.loc[calib_main_results["Models"] == model][
                                               "opt_dir list"].tolist()[0][0]]
            _, _, perf_results_per_group = gap_eval_scores(best_d["test_test_predictions"].tolist()[0],
                                                           best_d["test_test_labels"].tolist()[0],
                                                           best_d["test_test_private_labels"].tolist()[0])
            perf_results_per_group = {inv_protected_label_map[k]: v for k, v in perf_results_per_group.items() if
                                      k != "overall"}
            for protected_label, results in perf_results_per_group.items():
                if not results:
                    continue
                writer.writerow((model, task, protected_label, results["precision_macro"], results["recall_macro"],
                                 results["macro_fscore"]))


def write_raw_out(results, task, out_f):
    with open(out_f, "w") as fh_out:
        writer = csv.writer(fh_out)
        writer.writerow(("model", "criterion", "topic", "precision", "recall", "f_macro"))
        for model, d in results.items():
            per_private_label_results = d["per_private_label"]
            for protected_label, results in per_private_label_results.items():
                if not results:
                    continue
                writer.writerow((model, task, protected_label, results["precision_macro"],
                                 results["recall_macro"],
                                 results["macro_fscore"]))


def folds_results_to_csv(results, f, f_calib):
    with open(f, "w") as fh_out:
        writer = csv.writer(fh_out)
        writer.writerow(("Models", "test_performance mean", "test_fairness mean"))
        for model, d in results.items():
            writer.writerow((model, d["macro_fscore"], d["Fairness"]))

    with open(f_calib, "w") as fh_calib_out:
        writer_calib = csv.writer(fh_calib_out)
        writer_calib.writerow(("Models", "test_aurc_pos mean", "test_mce_pos mean", "test_ece_pos mean"))
        for model, d in results.items():
            writer_calib.writerow((model, d["aurc_pos"], d["mce_pos"], d["ece_pos"]))
