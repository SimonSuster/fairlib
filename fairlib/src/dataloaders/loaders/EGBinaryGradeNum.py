from sysrev.modelling.allennlp.util import BINARY_MAPPING2

from ..utils import BaseDataset
from allennlp.common import Params
from sysrev.modelling.allennlp.my_project.dataset_reader import *
#sys.path.append("/home/simon/Apps/SysRev/")

CAT_TYPES = ["review_type", "type_effect", "topics"]
TXT_TYPES =  ["full_abstract_txt", "abstract_conclusion_txt", "plain_language_summary_txt", "authors_conclusions_txt"]
NUM_TYPES = NUMERICAL_FEATS + NUMERICAL_FEATS_PER_OUTCOME
LABEL_MAP = BINARY_MAPPING2
TOPICS = ["Allergy & intolerance", "Blood disorders", "Cancer", "Child health", "Complementary & alternative medicine",
           "Consumer & communication strategies", "Dentistry & oral health", "Developmental, psychosocial & learning problems",
           "Ear, nose & throat", "Effective practice & health systems", "Endocrine & metabolic", "Eyes & vision",
           "Gastroenterology & hepatology", "Genetic disorders", "Gynaecology", "Health & safety at work", "Heart & circulation",
           "Infectious disease", "Insurance medicine", "Kidney disease", "Lungs & airways", "Mental health", "Neonatal care",
           "Neurology", "Orthopaedics & trauma", "Pain & anaesthesia", "Pregnancy & childbirth", "Public health",
           "Rheumatology", "Skin disorders", "Tobacco, drugs & alcohol", "Urology", "Wounds"]
PROTECTED_LABEL_MAP = dict(zip(TOPICS, list(range(len(TOPICS)))))

def get_params(data_dir, param_file= "/home/simon/Apps/SysRev/sysrev/modelling/allennlp/training_config/bi_class2_nonaugm_all.jsonnet"):
    params = Params.from_file(param_file)
    params["train_data_path"] = f"{data_dir}/train.csv"
    params["validation_data_path"] = f"{data_dir}/dev.csv"
    params["test_data_path"] = f"{data_dir}/test.csv"
    params["dataset_reader"][
        "serialization_dir"] = f"/home/simon/Apps/SysRevData/data/modelling/saved/{Path(param_file).stem}/"
    if not Path(params["dataset_reader"]["serialization_dir"]).exists():
        Path(params["dataset_reader"]["serialization_dir"]).mkdir()
    params["dataset_reader"][
        "scaler_training_path"] = "/home/simon/Apps/SysRevData/data/derivations/all/splits/FOLD/train.csv"

    return params


class EGBinaryGradeNum(BaseDataset):
    """
    Data reader for EvidenceGRADEr's binary grading task.
    Reads only numerical features.
    """
    def load_data(self):
        self.data_dir = self.args.data_dir
        # read experiment params from EvidenceGRADEr allennlp config file:
        params = get_params(self.data_dir, param_file="/home/simon/Apps/SysRev/sysrev/modelling/allennlp/training_config/bi_class2_nonaugm_all.jsonnet")

        # instantiate a data reader
        dataset_reader = DatasetReader.from_params(params["dataset_reader"])
        data_path = f"{self.data_dir}{self.split}.csv"

        for c, i in enumerate(dataset_reader._read(data_path)):
            #if c > 100:
            #    break
            # when multiple topics exist, create one instance per each topic:
            protected_labels = i.fields["cat_feats_list"].field_list[CAT_TYPES.index("topics")].tokens
            for protected_label in protected_labels:
                p_label = PROTECTED_LABEL_MAP.get(protected_label.text.replace("_", " "), None)
                if p_label is None:
                    continue
                self.protected_label.append(p_label)
                self.X.append(i.fields["feat"].array)
                self.y.append(LABEL_MAP[i.fields["label"].label])

        # data_loader = MultiProcessDataLoader(reader=dataset_reader, data_path=data_path, batch_size=params["data_loader"]["batch_size"], shuffle=params["data_loader"]["shuffle"])

        print()
