from itertools import chain
import json
from random import shuffle


class AttributeDataset:
    """
    Purpose of this class is to read data directory and find different attribute of the dataset
    i.e total number of classes, total number of images, instance of each class.
    """

    def __init__(self, dataset_paths):
        self.color_labels = []
        self.state_labels = []
        self.imgs_path = []

        # shuffle the dataset_paths as data will be split based path list
        shuffle(dataset_paths)
        for img_path, anno in dataset_paths:
            with open(anno, "r") as f:
                anno_dict = json.load(f)

            # assign "no_color" label to those which have no color specified in json.
            if "color" not in anno_dict.keys():
                self.color_labels.append(["no_color"])
            else:
                # lower() is used to tackle the cases sensitivity among labels e.g Green and GREEN.
                self.color_labels.append(list(map(lambda x: x.lower(), anno_dict["color"])))
            self.state_labels.append(list(map(lambda x: x.lower(), anno_dict["state"])))
            self.imgs_path.append(img_path)

        self.color_list = list(chain(*self.color_labels))
        # total color classes i.e (unique colors in json)
        self.color_classes = list(set(self.color_list))

        self.state_list = list(chain(*self.state_labels))
        # total state classes i.e (unique states in json)
        self.state_classes = list(set(self.state_list))

        self.num_colors = len(self.color_classes)
        self.num_states = len(self.state_classes)

    def class_occurence_count(self):
        """
        This function can be used to see each label occurence in the dataset
        """
        for i in self.color_classes:
            print(f"{i}: {self.color_list.count(i)}")

        for i in self.state_classes:
            print(f"{i}: {self.state_list.count(i)}")
