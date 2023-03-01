import csv
import difflib
import os
import random

import torch

from os.path import exists, join, realpath
from os import makedirs
import json
from collections import OrderedDict, Counter

import numpy as np
import pandas as pd


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
                f.write(term+'\n')

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

