import glob
import os
import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import balanced_accuracy_score
from torchvision.utils import make_grid
from utils import config as cfg


def encode_label(label, classes_list):
    """
    Encoding the classes into a tensor of shape (classes_list) with 0 and 1s i.e one-hot-encoding
    """
    target = torch.zeros(len(classes_list))
    for val in label:
        idx = classes_list.index(val)
        target[idx] = 1
    return target


def denorm_tensor(img_tensors):
    """
    denomalizing the tensor, if needed
    """
    return img_tensors * cfg.STD[0] + cfg.MEAN[0]


def calculate_metrics(output, target, thresh=0.6):
    """
    This function is used inside training loop to find accuracy of each class i.e color, state
    after each epoch.
    P.S. Not that accuracy should matter much in this case, reason is accuracy of each color should
    matter rather than the sum of whole color class. Moreover, the dataset is also imbalance and low.
    """
    predicted_color = output["color"].cpu().detach()
    predicted_color = predicted_color.numpy()
    gt_color = target["color_labels"].cpu()
    gt_color = gt_color.numpy()

    predicted_state = output["state"].cpu().detach()
    predicted_state = predicted_state.numpy()
    gt_state = target["state_labels"].cpu()
    gt_state = gt_state.numpy()

    accuracy_color = 0
    accuracy_state = 0
    with warnings.catch_warnings():
        # to ignore sklearn warnings
        warnings.simplefilter("ignore")
        for i in range(len(gt_color)):
            pred_color = predicted_color[i]
            pred_color = np.where(pred_color <= thresh, 0.0, 1.0)
            pred_state = predicted_state[i]
            pred_state = np.where(pred_state <= thresh, 0.0, 1.0)
            accuracy_color += balanced_accuracy_score(y_true=gt_color[i], y_pred=pred_color)
            accuracy_state += balanced_accuracy_score(y_true=gt_state[i], y_pred=pred_state)
        accuracy_color = accuracy_color / len(gt_color) * 100
        accuracy_state = accuracy_state / len(gt_state) * 100

    return accuracy_color, accuracy_state


def zip_dataset(path_):
    """
    This function reads the data and return list of tuple as [(image_path, json_path)].
    It is expecting a directory which contains a folder inside which there should be an image and json.
    """
    images_path = glob.glob(f"{path_}/*/*.jpg")
    json_path = glob.glob(f"{path_}/*/*.json")
    file_ = list(zip(images_path, json_path))
    return file_


def show_batch(dl, nmax=16):
    """
    To visualize a batch from dataloader
    """
    for d_val in dl:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(denorm_tensor(d_val["img"][:nmax]), nrow=4).permute(1, 2, 0))


def checkpoint_save(model):
    """
    save model checkpoint
    """
    os.makedirs(cfg.CHECKPOINT_PATH, exist_ok=True)
    path_ = os.path.join(cfg.CHECKPOINT_PATH, "best_checkpoint.pth")
    torch.save(model.state_dict(), path_)


def checkpoint_load(model):
    """
    load model checkpoint
    """
    path_ = os.path.join(cfg.CHECKPOINT_PATH, "best_checkpoint.pth")
    if os.path.isfile(path_):
        model.load_state_dict(torch.load(path_))
        return model
    else:
        print("No Checkpoint available")


def decode_pred(pred, attrib, thresh=0.6):
    """
    decode numeric prediction to the required output.
    """
    pred_color = pred["color"].sigmoid().cpu().detach().numpy()
    pred_color = np.where(pred_color <= thresh, False, True)
    pred_color = pred_color[0].tolist()
    colors = [attrib.color_classes[i] for i in range(len(attrib.color_classes)) if pred_color[i]]

    pred_state = pred["state"].sigmoid().cpu().detach().numpy()
    pred_state = np.where(pred_state <= thresh, False, True)
    pred_state = pred_state[0].tolist()
    state = [attrib.state_classes[i] for i in range(len(attrib.state_classes)) if pred_state[i]]
    return colors, state
