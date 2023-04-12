from fairlib.src.dataloaders.utils import get_num_groups

task = "bi_class2_nonaugm_all"
protected_attribute = "Area"
num_groups = get_num_groups(protected_attribute.lower())
pico_anno_f = "/home/simon/Apps/SysRevData/data/PICO_annotation_per_review_2020-11-30.csv"
data_dir = "/home/simon/Apps/SysRevData/data/derivations/all/splits/"
results_dir = ""  # dev/...
serialization_dir = "/home/simon/Apps/SysRevData/data/modelling/saved/"
fold_n = 0
batch_size = 2
lr = 2e-5
args = {
    # The name of the dataset, corresponding dataloader will be used,
    "dataset": "EGBinaryGrade",
    "encoder_architecture": "EvidenceGRADEr",
    "protected_attribute": protected_attribute,
    "pico_anno_f": pico_anno_f,
    # Specifiy the path to the input data
    "data_dir": data_dir,
    "results_dir": results_dir,
    "scaler_training_path": f"{data_dir}FOLD/train.csv",
    "fold_n": fold_n,
    "serialization_dir": serialization_dir,
    "vocabulary_dir": f"{serialization_dir}{task}/{task}_{fold_n}/vocabulary",
    # if it does not exist, the vocabulary will be created from scratch
    "param_file": f"/home/simon/Apps/SysRev/sysrev/modelling/allennlp/training_config/{task}.jsonnet",
    # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
    "device_id": -1,
    "num_groups": num_groups,
    # "emb_size": len(NUM_TYPES),
    "emb_size": 3232,
    "n_hidden": 1,  # although in original EG, n_hidden=0, some de-biasing methods require n_hidden>0
    "max_load": 30,
    "batch_size": batch_size,
    "test_batch_size": batch_size,
    "lr": lr,
    "epochs": 1,
    "epochs_since_improvement": 1,
    "group_agg_power": 2
}
