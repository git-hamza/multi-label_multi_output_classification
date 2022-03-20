import configparser


config_path = "../configuration.ini"
config = configparser.ConfigParser()
config.read(config_path)

hyperparameters_config = config["HYPER_PARAMETERS"]
paths_config = config["PATHS"]

MEAN = eval(hyperparameters_config.get("mean"))
STD = eval(hyperparameters_config.get("std"))
LEARNING_RATE = eval(hyperparameters_config.get("learning_rate"))
TRAINING_BATCH_SIZE = eval(hyperparameters_config.get("training_batch_size"))
VALIDATION_BATCH_SIZE = eval(hyperparameters_config.get("validation_batch_size"))
N_EPOCH = eval(hyperparameters_config.get("n_epoch"))
WANDB_FLAG = eval(hyperparameters_config.get("wandb_flag"))

CHECKPOINT_PATH = paths_config.get("checkpoint_path")
DATASET_PATH = paths_config.get("dataset_path")