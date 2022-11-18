import os

import numpy as np
import pandas as pd

from ..utils import BaseDataset


class COMPASDataset(BaseDataset):

    def load_data(self):

        self.data_dir = os.path.join(self.args.data_dir, "COMPAS_{}.pkl".format(self.split))

        data = pd.read_pickle(self.data_dir)

        self.X = data.drop(['sex', 'race', 'is_recid'], axis=1).to_numpy().astype(np.float32)
        self.y = list(data["is_recid"])

        if self.args.protected_task == "gender":
            self.protected_label = np.array(list(data["sex"])).astype(np.int32)  # Gender
        elif self.args.protected_task == "race":
            self.protected_label = np.array(list(data["race"])).astype(np.int32)  # Race
