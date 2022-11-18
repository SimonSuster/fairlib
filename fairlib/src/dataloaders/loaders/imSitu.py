import os

import numpy as np
import pandas as pd

from ..utils import BaseDataset


class imSituDataset(BaseDataset):

    def load_data(self):
        self.data_dir = os.path.join(self.args.data_dir, "imsitu_{}_df.pkl".format(self.split))

        data = pd.read_pickle(self.data_dir)

        self.X = list(data["transformed_images"])
        self.y = data["verb"].astype(np.int32)
        self.protected_label = data["gender"].astype(np.int32)
