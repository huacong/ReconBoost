import torch
import torchvision
import torch.nn as nn
import random
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self, output_dim = 40, pretrained=False):
        super(FeatureNet, self).__init__()
        self.base_model = torchvision.models.resnet18(pretrained=pretrained)
        self.feature_len = 512
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])
        self.fc = nn.Linear(self.feature_len, output_dim)

    def forward(self, x):
        # feature maps
        x = self.features(x)
        # flatten
        gf = x.view(x.size(0), -1)
        pred = self.fc(gf)
        return pred 


