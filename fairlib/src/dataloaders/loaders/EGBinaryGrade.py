from allennlp.common import Params
from sysrev.dataset_construction.dataset.robotreviewer_train_prepare import get_protected_attribute, load_pico_anno
from sysrev.modelling.allennlp.my_project.dataset_reader import *
from sysrev.modelling.allennlp.util import BINARY_MAPPING2, INV_BINARY_MAPPING2

from ..generalized_BT import get_data_distribution
from ..utils import BaseDataset, get_protected_label_map, get_inv_protected_label_map, CAT_TYPES
from ...utils import print_tsv


def binary_mapping_criteria(label):
    if label == "other":
        return 1
    else:
        return 0


def get_params(data_dir, fold_n, serialization_dir, scaler_training_path,
               param_file="/home/simon/Apps/SysRev/sysrev/modelling/allennlp/training_config/bi_class2_nonaugm_all.jsonnet"):
    params = Params.from_file(param_file)
    params["train_data_path"] = f"{data_dir}/{fold_n}/train.csv"
    params["validation_data_path"] = f"{data_dir}/{fold_n}/dev.csv"
    params["test_data_path"] = f"{data_dir}/{fold_n}/test.csv"
    params["dataset_reader"][
        "serialization_dir"] = f"{serialization_dir}{Path(param_file).stem}/"
    if not Path(params["dataset_reader"]["serialization_dir"]).exists():
        Path(params["dataset_reader"]["serialization_dir"]).mkdir()
    params["dataset_reader"][
        "scaler_training_path"] = scaler_training_path

    return params


def get_dataset_reader(data_dir, fold_n, serialization_dir, scaler_training_path, param_file):
    # read experiment params from EvidenceGRADEr allennlp config file:
    params = get_params(data_dir, fold_n, serialization_dir, scaler_training_path, param_file=param_file)

    # instantiate a data reader
    dataset_reader = DatasetReader.from_params(params["dataset_reader"])

    return dataset_reader


class EGBinaryGrade(BaseDataset):
    """
    Data reader for EvidenceGRADEr's binary grading task.
    """

    def load_data(self):
        self.data_dir = self.args.data_dir
        self.fold_n = self.args.fold_n
        self.serialization_dir = self.args.serialization_dir
        self.scaler_training_path = self.args.scaler_training_path
        self.pico_anno = load_pico_anno(self.args.pico_anno_f) if self.args.protected_attribute in {"Sex",
                                                                                                    "Age"} else None
        dataset_reader = get_dataset_reader(self.data_dir, self.fold_n, self.serialization_dir,
                                            self.scaler_training_path, param_file=self.args.param_file)

        if "bi_class2" in self.args.opt.param_file:
            label_map = lambda x: BINARY_MAPPING2[x]
        else:
            label_map = binary_mapping_criteria

        data_path = f"{self.data_dir}{self.fold_n}/{self.split}.csv"
        for c, i in enumerate(dataset_reader._read(data_path)):
            if self.args.max_load is not None:
                if c > self.args.max_load:
                    break
            # for area, we get the info from the data instance
            if self.args.protected_attribute == "Area":
                protected_labels = i.fields["cat_feats_list"].field_list[CAT_TYPES.index("topics")].tokens
            # for sex, we get the info from a separate meta file
            else:
                protected_labels = get_protected_attribute(i.fields["metadata"], self.pico_anno,
                                                           self.args.protected_attribute)
                if not isinstance(protected_labels, list):
                    protected_labels = [protected_labels]
            n_cat_types = len(CAT_TYPES)
            # when multiple areas exist, create one instance per each topic:
            for protected_label in protected_labels:
                if self.args.protected_attribute == "Area":
                    protected_label = protected_label.text.replace("_", " ")
                _i = i.duplicate()
                p_label = get_protected_label_map(self.args.protected_attribute.lower()).get(protected_label, None)
                if p_label is None:
                    continue
                self.protected_label.append(p_label)

                # erase area from the features
                assert CAT_TYPES[-1] == "topics"
                assert len(_i.fields["cat_feats_list"].field_list) == n_cat_types
                _i.fields["cat_feats_list"].field_list[-1].tokens = []
                self.X.append(_i)
                self.y.append(label_map(_i.fields["label"].label))

        print()
        data_dist = get_data_distribution(np.array(self.y), np.array(self.protected_label), y_size=2, g_size=self.args.num_groups)
        print_tsv(data_dist, get_inv_protected_label_map(self.args.protected_attribute.lower()),
                  self.args.protected_attribute.lower(), inv_label_mapping=INV_BINARY_MAPPING2)
