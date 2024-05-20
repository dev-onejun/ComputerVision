from torchvision.models import (
    resnet152,
    ResNet152_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
)
from torch import nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 10),
        )

    def forward(self, x):
        return self.model(x)


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()

        self.model = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
        self.model.classifier[1] = nn.Linear(1280, 10)

    def forward(self, x):
        return self.model(x)
