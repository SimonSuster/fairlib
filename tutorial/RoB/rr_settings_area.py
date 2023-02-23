from fairlib.src.dataloaders.loaders.EGBinaryGrade import TOPICS

task = "risk-of-bias"
data_dir = "/home/simon/Apps/robotreviewer/rob_abstract_dataset_area/"
results_dir = ""  # dev/...
serialization_dir = "/home/simon/Apps/SysRevData/data/modelling/saved/"
batch_size = 2
lr = 2e-5
args = {
    # The name of the dataset, corresponding dataloader will be used,
    "dataset": "RoB",
    "encoder_architecture": "SciBERT",
    # Specifiy the path to the input data
    "data_dir": data_dir,
    "results_dir": results_dir,
    "serialization_dir": serialization_dir,
    # Device for computing, -1 is the cpu; non-negative numbers indicate GPU id.
    "device_id": -1,
    "num_groups": len(TOPICS),
    # "emb_size": len(NUM_TYPES),
    "emb_size": 768,
    "n_hidden": 2,
    "max_load": 30,
    "batch_size": batch_size,
    "test_batch_size": batch_size,
    "lr": lr,
    "epochs": 1,
    "epochs_since_improvement": 1,
    "group_agg_power": 2
}
