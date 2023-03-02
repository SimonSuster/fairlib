import argparse
from collections import Counter

import numpy as np
import scipy
from sklearn.metrics import precision_score, recall_score, f1_score

from data.src.Sepsis.util import SEPSIS_LABEL_MAP, PROTECTED_LABEL_MAP_ETHN, INV_PROTECTED_LABEL_MAP_ETHN, \
    PROTECTED_LABEL_MAP_SEX, INV_PROTECTED_LABEL_MAP_SEX, get_protected_label_map, get_inv_protected_label_map
from fairlib.src.dataloaders.generalized_BT import get_data_distribution
from fairlib.src.utils import load_json


def random_baseline(f):
    test_set = load_json(f)
    y_true = []

    for i in test_set:
        y_true.append(SEPSIS_LABEL_MAP[i["y"]])

    y_true = np.array(y_true)
    y_pred = np.random.randint(0, 2, len(y_true))

    return {"precision_macro": round(precision_score(y_true, y_pred, average="macro"), 3),
            "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 3),
            "macro_fscore": round(f1_score(y_true, y_pred, average="macro"), 3),
            "fscore_class_pos": round(f1_score(y_true, y_pred, average="binary", pos_label=1), 3),
            "fscore_class_neg": round(f1_score(y_true, y_pred, average="binary", pos_label=0), 3)}


def freq_baseline(f):
    test_set = load_json(f)
    y_true = []

    for i in test_set:
        y_true.append(SEPSIS_LABEL_MAP[i["y"]])

    y_true = np.array(y_true)
    pred_label = Counter(y_true).most_common(1)[0][0]
    y_pred = [pred_label] * len(y_true)

    return {"precision_macro": round(precision_score(y_true, y_pred, average="macro"), 3),
            "recall_macro": round(recall_score(y_true, y_pred, average="macro"), 3),
            "macro_fscore": round(f1_score(y_true, y_pred, average="macro"), 3),
            "fscore_class_pos": round(f1_score(y_true, y_pred, average="binary", pos_label=1), 3),
            "fscore_class_neg": round(f1_score(y_true, y_pred, average="binary", pos_label=0), 3)}


def describe(input_dir, protected_attribute):
    protected_label_map = get_protected_label_map(protected_attribute)
    inv_protected_label_map = get_inv_protected_label_map(protected_attribute)

    train_set = load_json(f"{input_dir}/train.json")
    dev_set = load_json(f"{input_dir}/dev.json")
    test_set = load_json(f"{input_dir}/test.json")
    for dataset in (train_set, dev_set, test_set):
        xs = []
        labels = []
        protected_labels = []

        for c, i in enumerate(dataset):
            labels.append(SEPSIS_LABEL_MAP[i["y"]])
            protected_labels.append(protected_label_map[i["protected_label"]])

            xs.append(len(i["x"].split()))

        data_dist = get_data_distribution(np.array(labels), np.array(protected_labels))
        print(f"Label dist: {data_dist['y_dist']}")
        protected_labels_dist = [(inv_protected_label_map[idx], round(data_dist["g_dist"][idx], 2)) for idx in
                                 data_dist["g_dist"].argsort()]
        print(f"Protected label dist ({protected_attribute}): {protected_labels_dist}")
        txt_desc = scipy.stats.describe(xs)
        print(f"Text distribution {txt_desc}")
        print(f"SD: {np.sqrt(txt_desc.variance)}")
        print(f"N instances w/ len>512: {sum(np.array(xs) > 512)}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir',
                        help="directory containing prepared data")
    args = parser.parse_args()

    describe(f"{args.input_dir}/ethnicity/", protected_attribute="Ethnicity")
    describe(f"{args.input_dir}/sex/", protected_attribute="Sex")

    random_baseline_results = random_baseline(f"{args.input_dir}/ethnicity/test.json")
    freq_baseline_results = freq_baseline(f"{args.input_dir}/ethnicity/test.json")

    print(random_baseline_results)
    print(freq_baseline_results)
