import torch.nn as nn
from torchvision import models

def build_resnet50_4ch(num_classes: int = 4):
    model = models.resnet50(weights=None)
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model