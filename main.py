"""
This is the main file and you can run the code from here by executing the following command.
python3 main.py --mode train
"""
import argparse
import warnings
from sklearn.metrics import classification_report
import torch
from torch.utils.data import DataLoader
from dataset.data_attributes import AttributeDataset
from dataset.data_split import split_data
from model import MultiOutputModel
from test import test
from train import train
from utils.helper_functions import zip_dataset, checkpoint_load
import utils.config as cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Running Pipeline")
    parser.add_argument("--mode", type=str, default="train")
    args = parser.parse_args()

    All_files = zip_dataset(cfg.DATASET_PATH)
    # loading data attributes
    print("Loading attributes....")
    data_attrib = AttributeDataset(All_files)

    # splitting the dataset and applying transformation
    print("Splitting Dataset.....")
    train_dataset, val_dataset, test_dataset = split_data(All_files, data_attrib)

    # Dataloader
    print("Calling Dataloader....")
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.VALIDATION_BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset)

    # setting up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # loading model
    print("Loading Model....")
    model = MultiOutputModel(
        n_color_classes=data_attrib.num_colors, n_state_classes=data_attrib.num_states
    ).to(device)

    if args.mode == "train":
        train(model, train_loader, val_loader, device)

    if args.mode == "test":
        model = checkpoint_load(model)
        results = test(model, test_dataset)
        with warnings.catch_warnings():
            # to ignore sklearn warnings
            warnings.simplefilter("ignore")
            print("Color Classification Report")
            print(classification_report(results[0], results[1], target_names=data_attrib.color_classes))

            print("___________________________________")
            print("State Classification Report")
            print(classification_report(results[2], results[3], target_names=data_attrib.state_classes))
