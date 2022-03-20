from dataset.data_loading import AssetsDataset
from torchvision import transforms
import utils.config as cfg


def split_data(dataset, attrib, split_ratio=(.8, .1, .1)):
    """
    This function is used to apply transformation on the dataset as well as split them
    into train, val and test.
    """
    train_range = 0, int(split_ratio[0]*len(dataset))
    val_range = train_range[1], int(split_ratio[1]*len(dataset)) + train_range[1]
    test_range = val_range[1], int(split_ratio[2]*len(dataset)) + val_range[1]

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1,), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.MEAN, cfg.STD)
    ])

    train_dataset = AssetsDataset(attrib, train_range, train_transform)
    val_dataset = AssetsDataset(attrib, val_range, train_transform)
    test_dataset = AssetsDataset(attrib, test_range, test_transform)

    return train_dataset, val_dataset, test_dataset
