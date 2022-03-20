from torch.utils.data import Dataset
from PIL import Image
from utils.helper_functions import encode_label


class AssetsDataset(Dataset):
    """
    Purpose of this class is to load the assets dataset.
    As we are treating the problem as multi output, multiclass classification, that is why
    the output is an image and the labels are color and state. Moreoever, color and state also have
    different classes inside them.
    """

    def __init__(self, attrib, data_range, transform=None):
        super().__init__()
        self.transform = transform
        self.attrib = attrib
        # initialize the arrays to store the ground truth labels and paths to the images
        self.data = []
        self.color_labels = []
        self.state_labels = []

        for i in range(data_range[0], data_range[1]):
            self.data.append(attrib.imgs_path[i])
            self.color_labels.append(self.attrib.color_labels[i])
            self.state_labels.append(self.attrib.state_labels[i])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx])

        # for preprocessing i.e normalizing, augmentation, resizing
        if self.transform:
            img = self.transform(img)

        dict_data = {
            "img": img,
            "labels": {
                "color_labels": encode_label(self.color_labels[idx], self.attrib.color_classes),
                "state_labels": encode_label(self.state_labels[idx], self.attrib.state_classes),
            },
        }
        return dict_data
