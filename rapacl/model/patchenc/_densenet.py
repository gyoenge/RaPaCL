import torch.nn as nn
import torchvision.models as models


def build_densenet121(pretrained=True):
    model = models.densenet121(pretrained=pretrained)

    in_dim = model.classifier.in_features
    model.classifier = nn.Identity()

    return model, in_dim
