from sysrev.dataset_construction.util.util import load_json

from data.src.Sepsis.util import get_protected_group, get_protected_label_map, label_map
from ..utils import BaseDataset


class Sepsis(BaseDataset):
    """
    Data reader for sepsis identification from clinical notes
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

            p_label = protected_label_map.get(i["protected_label"])
            if p_label is None:
                continue
            self.protected_label.append(p_label)

            self.X.append(i["x"])
            self.y.append(label_map(i["y"]))

        if self.args.encoder_architecture in {"BERT", "SciBERT", "BlueBERT", "ClinicalBERT"}:
            self.X = self.args.text_encoder.encoder(self.X)
        else:
            raise NotImplementedError
        print()
