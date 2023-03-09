from data.src.Sepsis.util import PROTECTED_LABEL_MAP_SEX

task = "sepsis"
data_dir = "/home/simon/Datasets/sepsis_mimic_discharge/output/sex/"
results_dir = ""  # dev/...
serialization_dir = "/home/simon/Apps/SysRevData/data/modelling/saved/"
batch_size = 32
lr = 2e-5
args = {
    # The name of the dataset, corresponding dataloader will be used,
    "dataset": "Sepsis",
    "encoder_architecture": "BlueBERT",
    # Specifiy the path to the input data
    "data_dir": data_dir,
    "results_dir": results_dir,
    "serialization_dir": serialization_dir,
    # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
    "device_id": -1,
    "num_groups": len(PROTECTED_LABEL_MAP_SEX),
    # "emb_size": len(NUM_TYPES),
    #"emb_size": 768,
    "emb_size": 1024,
    "n_hidden": 2,
    #"max_load": 30,
    "batch_size": batch_size,
    "test_batch_size": batch_size,
    "lr": lr,
    "epochs": 3,
    "epochs_since_improvement": 1,
    "group_agg_power": 2
}
