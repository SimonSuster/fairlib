from pathlib import Path

SEPSIS_LABEL_MAP = {"non_septic": 0, "septic": 1}
INV_SEPSIS_LABEL_MAP = {v: k for k, v in SEPSIS_LABEL_MAP.items()}

PROTECTED_LABEL_MAP_SEX = {"M": 0, "F": 1}
INV_PROTECTED_LABEL_MAP_SEX = {v: k for k, v in PROTECTED_LABEL_MAP_SEX.items()}

PROTECTED_LABEL_MAP_ETHN = {"White": 0, "Black": 1, "Asian": 2, "Hispanic": 3, "Other": 4}
INV_PROTECTED_LABEL_MAP_ETHN = {v: k for k, v in PROTECTED_LABEL_MAP_ETHN.items()}


def label_map(label):
    return SEPSIS_LABEL_MAP[label]


def get_protected_group(data_dir):
    if Path(data_dir).parts[-1].endswith("ethnicity"):
        protected_group = "ethnicity"
    elif Path(data_dir).parts[-1].endswith("sex"):
        protected_group = "sex"
    else:
        raise ValueError

    return protected_group


def get_protected_label_map(protected_group):
    return {
        "ethnicity": PROTECTED_LABEL_MAP_ETHN,
        "sex": PROTECTED_LABEL_MAP_SEX}[protected_group.lower()]


def get_inv_protected_label_map(protected_group):
    return {
        "ethnicity": INV_PROTECTED_LABEL_MAP_ETHN,
        "sex": INV_PROTECTED_LABEL_MAP_SEX}[protected_group.lower()]
