from allennlp.common import Params
from sysrev.modelling.allennlp.my_project.dataset_reader import *
from sysrev.modelling.allennlp.util import BINARY_MAPPING2

from ..utils import BaseDataset

# sys.path.append("/home/simon/Apps/SysRev/")

CAT_TYPES = ["review_type", "type_effect", "topics"]
TXT_TYPES = ["full_abstract_txt", "abstract_conclusion_txt", "plain_language_summary_txt", "authors_conclusions_txt"]
NUM_TYPES = NUMERICAL_FEATS + NUMERICAL_FEATS_PER_OUTCOME
LABEL_MAP = BINARY_MAPPING2
TOPICS = ["Allergy & intolerance", "Blood disorders", "Cancer", "Child health", "Complementary & alternative medicine",
          "Consumer & communication strategies", "Dentistry & oral health",
          "Developmental, psychosocial & learning problems",
          "Ear, nose & throat", "Effective practice & health systems", "Endocrine & metabolic", "Eyes & vision",
          "Gastroenterology & hepatology", "Genetic disorders", "Gynaecology", "Health & safety at work",
          "Heart & circulation",
          "Infectious disease", "Insurance medicine", "Kidney disease", "Lungs & airways", "Mental health",
          "Neonatal care",
          "Neurology", "Orthopaedics & trauma", "Pain & anaesthesia", "Pregnancy & childbirth", "Public health",
          "Rheumatology", "Skin disorders", "Tobacco, drugs & alcohol", "Urology", "Wounds"]
PROTECTED_LABEL_MAP = dict(zip(TOPICS, list(range(len(TOPICS)))))
INV_PROTECTED_LABEL_MAP = {v: k for k, v in PROTECTED_LABEL_MAP.items()}


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

        dataset_reader = get_dataset_reader(self.data_dir, self.fold_n, self.serialization_dir,
                                            self.scaler_training_path, param_file=self.args.param_file)

        data_path = f"{self.data_dir}{self.fold_n}/{self.split}.csv"
        for c, i in enumerate(dataset_reader._read(data_path)):
            if self.args.max_load is not None:
                if c > self.args.max_load:
                    break
            # when multiple topics exist, create one instance per each topic:
            protected_labels = i.fields["cat_feats_list"].field_list[CAT_TYPES.index("topics")].tokens
            n_cat_types = len(CAT_TYPES)
            for protected_label in protected_labels:
                _i = i.duplicate()
                p_label = PROTECTED_LABEL_MAP.get(protected_label.text.replace("_", " "), None)
                if p_label is None:
                    continue
                self.protected_label.append(p_label)

                # erase protected label from the features
                assert CAT_TYPES[-1] == "topics"
                assert len(_i.fields["cat_feats_list"].field_list) == n_cat_types
                _i.fields["cat_feats_list"].field_list[-1].tokens = []
                self.X.append(_i)
                self.y.append(LABEL_MAP[_i.fields["label"].label])

        print()
