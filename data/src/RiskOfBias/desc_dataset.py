import argparse
from collections import Counter

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from fairlib.src.dataloaders.generalized_BT import get_data_distribution
from fairlib.src.dataloaders.utils import PROTECTED_LABEL_MAP_AREA, INV_PROTECTED_LABEL_MAP_AREA, \
    PROTECTED_LABEL_MAP_AGE, INV_PROTECTED_LABEL_MAP_AGE, PROTECTED_LABEL_MAP_SEX, INV_PROTECTED_LABEL_MAP_SEX
from fairlib.src.dataloaders.loaders.RoB import label_map
from fairlib.src.utils import load_json, print_tsv, INV_ROB_MAPPING


def random_baseline(f):
    test_set = load_json(f)
    y_true = []

    for i in test_set:
        y_true.append(label_map(i["labels"]))

    y_true = np.array(y_true).astype(int)
    y_pred = np.random.randint(0, 2, len(y_true))

    return {"precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "macro_fscore": f1_score(y_true, y_pred, average="macro")}


def freq_baseline(f):
    test_set = load_json(f)
    y_true = []

    for i in test_set:
        y_true.append(label_map(i["labels"]))

    y_true = np.array(y_true).astype(int)
    pred_label = Counter(y_true).most_common(1)[0][0]
    y_pred = [pred_label] * len(y_true)

    return {"precision_macro": precision_score(y_true, y_pred, average="macro"),
            "recall_macro": recall_score(y_true, y_pred, average="macro"),
            "macro_fscore": f1_score(y_true, y_pred, average="macro")}


def describe(input_dir, protected_group):
    train_set = load_json(f"{input_dir}/train.json")
    dev_set = load_json(f"{input_dir}/dev.json")
    test_set = load_json(f"{input_dir}/test.json")

    protected_label_map = {"area": PROTECTED_LABEL_MAP_AREA,
                           "age": PROTECTED_LABEL_MAP_AGE,
                           "sex": PROTECTED_LABEL_MAP_SEX}[protected_group]

    inv_protected_label_map = {"area": INV_PROTECTED_LABEL_MAP_AREA,
                               "age": INV_PROTECTED_LABEL_MAP_AGE,
                               "sex": INV_PROTECTED_LABEL_MAP_SEX}[protected_group]
    all_d = []

    for dataset in (train_set, dev_set, test_set):
        abstracts = []
        pmids = []
        labels = []
        protected_labels = []
        protected_labels_idx = []
        for c, i in enumerate(dataset):
            i_protected_labels = i["protected_labels"]
            if not isinstance(i_protected_labels, list):
                i_protected_labels = [i_protected_labels]

            # when multiple protected labels per single protected group exist, create one instance for each label:
            for protected_label in i_protected_labels:
                p_label = protected_label_map.get(protected_label.replace("_", " "), None)
                if p_label is None:
                    continue

                abstracts.append(i["abstract"])
                pmids.append(i["pmid"])
                labels.append(label_map(i["labels"]))
                protected_labels.append(protected_label)
                protected_labels_idx.append(p_label)
        all_d.append((pmids, labels, protected_labels))
        labels = np.array(labels).astype(int)
        """
        print(f"Size: {len(labels)}")
        abstract_desc = scipy.stats.describe([len(a.split()) for a in abstracts])
        print(f"Abstracts stats: {abstract_desc}")
        """
        data_dist = get_data_distribution(np.array(labels), np.array(protected_labels_idx), y_size=2, g_size=len(protected_label_map))
        """
        print(f"Label dist: {data_dist['y_dist']}")
        protected_labels_dist = [(inv_protected_label_map[idx], round(data_dist["g_dist"][idx], 2)) for idx in
                                 data_dist["g_dist"].argsort()]
        print(f"Protected label dist: {protected_labels_dist}")
        print()
        """
        print_tsv(data_dist, inv_protected_label_map, protected_group, inv_label_mapping=INV_ROB_MAPPING)

    return all_d


def intersectional_desc(d_sex, d_area, split_set="train", out_f="/home/simon/Apps/SysRevData/data/modelling/plots/robotreviewer/intersectional.csv"):
    import csv
    from scipy.stats.contingency import crosstab

    set_idx = {"train": 0, "dev": 1, "test": 2}[split_set]
    # only train sets
    joint_pmids = set(d_area[set_idx][0]) & set(d_sex[set_idx][0])

    d = {}
    for pmid, label, protected_label in zip(d_sex[set_idx][0], d_sex[set_idx][1], d_sex[set_idx][2]):
        if pmid not in joint_pmids:
            continue
        if pmid not in d:
            d[pmid] = {}
        d[pmid]["label"] = label
        d[pmid]["protected_label_sex"] = protected_label

    for pmid, label, protected_label in zip(d_area[set_idx][0], d_area[set_idx][1], d_area[set_idx][2]):
        if pmid not in joint_pmids:
            continue
        if pmid not in d:
            d[pmid] = {}
        d[pmid]["label"] = label
        d[pmid]["protected_label_area"] = protected_label

    sexes = [i["protected_label_sex"] for i in d.values()]
    areas = [i["protected_label_area"] for i in d.values()]
    results = crosstab(sexes, areas)

    n_sex, n_area = results[1].shape

    with open(out_f, "w") as fh_out:
        writer = csv.writer(fh_out)
        writer.writerow(("Area", "Sex", "Count"))
        for i in range(n_sex):
            for j in range(n_area):
                sex = results[0][0][i]
                area = results[0][1][j]
                count = results[1][i][j]
                writer.writerow((area, sex, count))

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input_dir', default="/home/simon/Apps/robotreviewer/rob_abstract_dataset_sex/",
                        help="directory containing prepared data")
    parser.add_argument('-protected_group',
                        help="name of the protected group: e.g. area, age, sex")
    args = parser.parse_args()

    describe(args.input_dir, args.protected_group)
    random_baseline_results = random_baseline(f"{args.input_dir}/test.json")
    freq_baseline_results = freq_baseline(f"{args.input_dir}/test.json")
    print(random_baseline_results)
    print(freq_baseline_results)

    #d_sex = describe(args.input_dir, "sex")
    #d_area = describe(args.input_dir.replace("sex", "area"), "area")
    #intersectional_desc(d_sex, d_area, split_set="dev")
