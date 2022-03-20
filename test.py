import numpy as np
from utils.helper_functions import decode_pred


def test(model, test_dataset, thresh=0.6):
    """
    This function is used for inference, it takes input, test_dataset and thresh as an input.
    It output list of numpy arrays prediciton as well as ground trurth for the provided test_dataset
    for each class i.e color and state.
    This data is used in main to provide classification report i.e precision, recall and f1 score.
    """
    prediction_color = []
    prediction_state = []
    gt_color = []
    gt_state = []
    for batch in test_dataset:
        pred = model(batch["img"].unsqueeze(0))
        pred_color = pred["color"].cpu().detach().numpy()
        pred_color = np.where(pred_color <= thresh, 0., 1.)
        pred_state = pred["state"].cpu().detach().numpy()
        pred_state = np.where(pred_state <= thresh, 0., 1.)

        prediction_color.append(pred_color[0])
        gt_color.append(batch["labels"]["color_labels"].cpu().numpy())
        prediction_state.append(pred_state[0])
        gt_state.append(batch["labels"]["state_labels"].cpu().numpy())

    return prediction_color, gt_color, prediction_state, gt_state


def infer_single_image(model, data_attrib, img):
    """
    This function infers on a single image and returns decoded prediction i.e in the same form as
    mentioned in json.
    """
    img = img.unsqueeze(0)
    model.eval()
    pred = model(img)
    print(decode_pred(pred,data_attrib))

