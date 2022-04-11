import torch
import torch.nn as nn
import torch.nn.functional as func
import torchvision.models as models


class MultiOutputModel(nn.Module):
    """
    Purpose of this classes is to build the model. We treated the problem as multi-output
    multi-class classification, so we output two final classification layers i.e one for the state,
    one for the colors.

    We have chosen mobilenetv2 and its pretrained weights. Mobilenetv2 is a light weight architecture
    and works better on simple problems. We are restricted by the dataset size, so we went for
    pretrained imagenet weight and also restricted ourself from going towards bigger architectures
    e.g ResNet, Inception etc.
    """

    def __init__(self, n_color_classes, n_state_classes):
        super().__init__()
        self.base_model = models.mobilenet_v2(pretrained=True).features
        last_channel = models.mobilenet_v2().last_channel

        # the input for the classifier should be two-dimensional, but we will have
        # [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.color = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=n_color_classes)
        )
        self.state = nn.Sequential(
            nn.Dropout(p=0.2), nn.Linear(in_features=last_channel, out_features=n_state_classes)
        )
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        x = self.base_model(x)
        x = self.pool(x)

        # reshape from [batch, channels, 1, 1] to [batch, channels] to put it into classifier
        x = torch.flatten(x, 1)

        return {"color": self.sigm(self.color(x)), "state": self.sigm(self.state(x))}

    def get_loss(self, net_output, ground_truth):
        """
        Choosing loss function and final layer activation was important part.
        We can have multiple color and state at once, so we opted for sigmoid activation function
        rather than softmax because softmax predicted probabilities sum is 1 and
        there couldn't be several correct outputs.

        For loss we choose BCS instead of CCE because we wanted each prediction inside color and state
        to be treated independently.
        """
        color_loss = func.binary_cross_entropy(net_output["color"], ground_truth["color_labels"])
        state_loss = func.binary_cross_entropy(net_output["state"], ground_truth["state_labels"])
        loss = color_loss + state_loss
        return loss, {"color": color_loss, "state": state_loss}
