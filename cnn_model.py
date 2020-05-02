from torch.nn.functional import relu
from utils import *
import torch.nn as nn


class FashionConvNet(nn.Module):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - conv - relu - 2x2 max pool - affine - softmax (cross-entropy)

    The network is trained on FashionMNIST dataset to predict the right class for each image.
    """

    def __init__(self, in_channel, channel_1, channel_2, num_classes, act):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=channel_1, kernel_size=5, padding=2)
        nn.init.kaiming_normal_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(num_features=channel_1, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=channel_1, out_channels=channel_2, kernel_size=3, padding=1)
        nn.init.kaiming_normal_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(num_features=channel_2, eps=1e-05, momentum=0.1, affine=True,
                                  track_running_stats=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc = nn.Linear(channel_2 * 7 * 7, num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

        self.activation = act

    def forward(self, x):
        scores = None

        x = self.maxpool1(self.activation(self.bn1(self.conv1(x))))
        x = self.maxpool2(self.activation(self.bn2(self.conv2(x))))
        x = flatten(x)
        scores = self.fc(x)

        return scores
