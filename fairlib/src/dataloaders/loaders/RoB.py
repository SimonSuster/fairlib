from pathlib import Path

from sysrev.dataset_construction.util.util import load_json
from sysrev.modelling.allennlp.util import ROB_MAPPING

from .EGBinaryGrade import PROTECTED_LABEL_MAP_AREA, PROTECTED_LABEL_MAP_AGE, PROTECTED_LABEL_MAP_SEX, \
    INV_PROTECTED_LABEL_MAP_AREA, INV_PROTECTED_LABEL_MAP_AGE, INV_PROTECTED_LABEL_MAP_SEX
from ..utils import BaseDataset


def binary_mapping_criteria(label):
    if label == "other":
        return 1
    else:
        return 0


def label_map(labels):
    valid_tasks = {"rand_seq_gen", "allo_conceal", "blinding_part", "outcome_blinding"}  # RSG, AC, BPP, BOA
    mapped = [ROB_MAPPING[label] for task, label in labels.items() if task in valid_tasks]

    # positive only when we have low risk on all tasks
    label = sum(mapped) == len(mapped)

    return label


def get_protected_group(data_dir):
    if Path(data_dir).parts[-1].endswith("area"):
        protected_group = "area"
    elif Path(data_dir).parts[-1].endswith("age"):
        protected_group = "age"
    elif Path(data_dir).parts[-1].endswith("sex"):
        protected_group = "sex"
    else:
        raise ValueError

    return protected_group


def get_protected_label_map(protected_group):
    return {
        "area": PROTECTED_LABEL_MAP_AREA,
        "age": PROTECTED_LABEL_MAP_AGE,
        "sex": PROTECTED_LABEL_MAP_SEX}[protected_group]


def get_inv_protected_label_map(protected_group):
    return {
        "area": INV_PROTECTED_LABEL_MAP_AREA,
        "age": INV_PROTECTED_LABEL_MAP_AGE,
        "sex": INV_PROTECTED_LABEL_MAP_SEX}[protected_group]


class RoB(BaseDataset):
    """
    Data reader for  risk of bias assessment at study level
    """
    def load_data(self):
        self.data_dir = self.args.data_dir
        self.serialization_dir = self.args.serialization_dir

        protected_group = get_protected_group(self.data_dir)

        protected_label_map = get_protected_label_map(protected_group)

        data_path = f"{self.data_dir}/{self.split}.json"
        for c, i in enumerate(load_json(data_path)):
            if self.args.max_load is not None:
                if c > self.args.max_load:
                    break
            i_protected_labels = i["protected_labels"]
            if not isinstance(i_protected_labels, list):
                i_protected_labels = [i_protected_labels]

            # when multiple topics exist, create one instance per each topic:
            for protected_label in i_protected_labels:
                p_label = protected_label_map.get(protected_label.replace("_", " "), None)
                if p_label is None:
                    continue
                self.protected_label.append(p_label)
                abstract = i["abstract"]

                self.X.append(abstract)
                self.y.append(label_map(i["labels"]))

        if self.args.encoder_architecture in {"BERT", "SciBERT"}:
            self.X = self.args.text_encoder.encoder(self.X)
        else:
            raise NotImplementedError
        print()
