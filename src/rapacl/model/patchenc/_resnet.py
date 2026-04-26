import torch.nn as nn
import torchvision.models as models


def build_resnet50(pretrained=True):
    model = models.resnet50(pretrained=pretrained)

    in_dim = model.fc.in_features   # 2048
    model.fc = nn.Identity()

    return model, in_dim
