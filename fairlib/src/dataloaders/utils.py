import numpy as np
import pandas as pd
import torch

from .BT import get_weights, get_sampled_indices
from .generalized_BT import get_data_distribution, manipulate_data_distribution

CAT_TYPES = ["review_type", "type_effect", "topics"]
TXT_TYPES = ["full_abstract_txt", "abstract_conclusion_txt", "plain_language_summary_txt", "authors_conclusions_txt"]
NUMERICAL_FEATS = ["n_participants", "n_studies", "year", "n_sofs", "n_outcomes",
                   "ci_lower", "ci_upper", "relative_effect", "n_included_studies", "n_excluded_studies",
                   "n_ongoing_studies",
                   "n_additional_studies", "n_other_studies"
                   ]
NUMERICAL_FEATS_PER_OUTCOME = ["val_chi2", "val_df", "val_i2", "val_i2_q", "val_p_chi2", "val_p_q", "val_p_z", "val_q",
                               "val_tau2", "val_z", 'val_n_high_risk_alloc', 'val_n_high_risk_incom',
                               'val_n_high_risk_other', 'val_n_high_risk_outco', 'val_n_high_risk_partb',
                               'val_n_high_risk_blind', 'val_n_high_risk_rands', 'val_n_high_risk_selec',
                               'val_n_low_risk_alloc', 'val_n_low_risk_incom', 'val_n_low_risk_other',
                               'val_n_low_risk_outco', 'val_n_low_risk_partb', 'val_n_low_risk_blind',
                               'val_n_low_risk_rands', 'val_n_low_risk_selec', 'val_n_unclear_risk_alloc',
                               'val_n_unclear_risk_incom', 'val_n_unclear_risk_other', 'val_n_unclear_risk_outco',
                               'val_n_unclear_risk_partb', 'val_n_unclear_risk_blind', 'val_n_unclear_risk_rands',
                               'val_n_unclear_risk_selec', 'val_prop_high_risk_alloc', 'val_prop_high_risk_incom',
                               'val_prop_high_risk_other', 'val_prop_high_risk_outco', 'val_prop_high_risk_partb',
                               'val_prop_high_risk_blind', 'val_prop_high_risk_rands', 'val_prop_high_risk_selec']
NUM_TYPES = NUMERICAL_FEATS + NUMERICAL_FEATS_PER_OUTCOME
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
PROTECTED_LABEL_MAP_AREA = dict(zip(TOPICS, list(range(len(TOPICS)))))
INV_PROTECTED_LABEL_MAP_AREA = {v: k for k, v in PROTECTED_LABEL_MAP_AREA.items()}
AGE = ['Child 6-12 years', 'Aged 65-79 years', 'Middle Aged 45-64 years', 'Adolescent 13-18 years', 'Adult 19-44 years',
       'Child, Preschool 2-5 years', 'Birth to 1 mo', 'Young Adult 19-24 years', 'Aged 80 and over 80+ years', 'Infant 1 to 23 mo']
PROTECTED_LABEL_MAP_AGE = dict(zip(AGE, list(range(len(AGE)))))
INV_PROTECTED_LABEL_MAP_AGE = {v: k for k, v in PROTECTED_LABEL_MAP_AGE.items()}
SEX = ["Male", "Female", "Male and Female"]
PROTECTED_LABEL_MAP_SEX = dict(zip(SEX, list(range(len(SEX)))))
INV_PROTECTED_LABEL_MAP_SEX = {v: k for k, v in PROTECTED_LABEL_MAP_SEX.items()}
INV_ROB_MAPPING = {1: "Low risk", 0: "High/unclear risk"}



def full_label_data(df, tasks):
    """filter the instances with all required labels

    Args:
        df (pd.DataFrame): a DataFrame containing data instances
        tasks (list): a list of names of target columns

    Returns:
        np.array: an array of boolean values indicating whether or not each row meets the requirement.
    """
    selected_rows = np.array([True] * len(df))
    for task in tasks:
        selected_rows = selected_rows & df[task].notnull().to_numpy()
    return selected_rows


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split

        self.X = []
        self.y = []
        self.protected_label = []
        self.instance_weights = []
        self.adv_instance_weights = []
        self.regression_label = []

        self.load_data()

        self.regression_init()

        if self.args.encoder_architecture != "EvidenceGRADEr":
            self.X = np.array(self.X)
            if len(self.X.shape) == 3:
                self.X = np.concatenate(list(self.X), axis=0)
        self.y = np.array(self.y).astype(int)
        self.protected_label = np.array(self.protected_label).astype(int)

        self.manipulate_data_distribution()

        self.balanced_training()

        self.adv_balanced_training()

        if self.split == "train":
            self.adv_decoupling()

        print("Loaded data shapes {}: {}, {}, {}".format(self.split,
                                                         self.X.shape if self.args.encoder_architecture != "EvidenceGRADEr" else len(
                                                             self.X), self.y.shape,
                                                         self.protected_label.shape))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.y)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.X[index], self.y[index], self.protected_label[index], self.instance_weights[index], \
               self.adv_instance_weights[index], self.regression_label[index]

    def load_data(self):
        pass

    def manipulate_data_distribution(self):
        if self.args.GBT and self.split == "train":
            # Get data distribution
            distribution_dict = get_data_distribution(y_data=self.y, g_data=self.protected_label, y_size=2, g_size=self.args.num_groups)

            selected_index = manipulate_data_distribution(
                default_distribution_dict=distribution_dict,
                N=self.args.GBT_N,
                GBTObj=self.args.GBTObj,
                alpha=self.args.GBT_alpha)

            self.X = self.X[selected_index]
            self.y = self.y[selected_index]
            self.protected_label = self.protected_label[selected_index]

    def balanced_training(self):
        if (self.args.BT is None) or (self.split != "train"):
            # Without balanced training
            self.instance_weights = np.array([1 for _ in range(len(self.protected_label))])
        else:
            assert self.args.BT in ["Reweighting", "Resampling", "Downsampling"], "not implemented"

            assert self.args.BTObj in ["joint", "y", "g", "stratified_y", "stratified_g", "EO"], "not implemented"
            """
            reweighting each training instance 
                joint:          y,g combination, p(g,y)
                y:              main task label y only, p(y)
                g:              protected label g only, p(g)
                stratified_y:   balancing the g for each y, p(g|y), while keeping the y distribution
                stratified_g:   balancing the y for each g, p(y|g)
                EO:             balancing the g for each y, p(g|y)
            """

            if self.args.BT == "Reweighting":
                self.instance_weights = get_weights(self.args.BTObj, self.y, self.protected_label)

            elif self.args.BT in ["Resampling", "Downsampling"]:

                selected_index = get_sampled_indices(self.args.BTObj, self.y, self.protected_label, method=self.args.BT)

                X = [self.X[index] for index in selected_index]
                if self.args.encoder_architecture != "EvidenceGRADEr":
                    self.X = np.array(X)
                else:
                    self.X = X
                y = [self.y[index] for index in selected_index]
                self.y = np.array(y)
                _protected_label = [self.protected_label[index] for index in selected_index]
                self.protected_label = np.array(_protected_label)
                self.instance_weights = np.array([1 for _ in range(len(self.protected_label))])

            else:
                raise NotImplementedError
        return None

    def adv_balanced_training(self):
        if (self.args.adv_BT is None) or (self.split != "train"):
            # Without balanced training
            self.adv_instance_weights = np.array([1 for _ in range(len(self.protected_label))])
        else:
            assert self.args.adv_BT in ["Reweighting"], "not implemented"

            assert self.args.adv_BTObj in ["joint", "y", "g", "stratified_y", "stratified_g", "EO"], "not implemented"
            """
            reweighting each training instance 
                joint:          y,g combination, p(g,y)
                y:              main task label y only, p(y)
                g:              protected label g only, p(g)
                stratified_y:   balancing the g for each y, p(g|y)
                stratified_g:   balancing the y for each g, p(y|g)
            """

            if self.args.adv_BT == "Reweighting":
                self.adv_instance_weights = get_weights(self.args.adv_BTObj, self.y, self.protected_label)
            else:
                raise NotImplementedError
        return None

    def adv_decoupling(self):
        """Simulating unlabelled protected labels through assigning -1 to instances.

        Returns:
            None
        """
        if self.args.adv_decoupling and self.args.adv_decoupling_labelled_proportion < 1:
            self.adv_instance_weights[
                np.random.rand(len(self.protected_label)) > self.args.adv_decoupling_labelled_proportion
                ] = -1
        else:
            pass
        return None

    def regression_init(self):
        if not self.args.regression:
            self.regression_label = np.array([0 for _ in range(len(self.protected_label))])
        else:
            # Discretize variable into equal-sized buckets
            if self.split == "train":
                bin_labels, bins = pd.qcut(self.y, q=self.args.n_bins, labels=False, duplicates="drop", retbins=True)
                self.args.regression_bins = bins
            else:
                bin_labels = pd.cut(self.y, bins=self.args.regression_bins, labels=False, duplicates="drop",
                                    include_lowest=True)
            bin_labels = np.nan_to_num(bin_labels, nan=0)

            # Reassign labels
            self.regression_label, self.y = np.array(self.y), bin_labels


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


def get_num_groups(attribute):
    return len({"area": TOPICS, "sex": SEX, "age": AGE}[attribute])