import torch
import torchvision
import torch.nn as nn
import random
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self, pretrained=False):
        super(FeatureNet, self).__init__()
        self.base_model = torchvision.models.resnet18(pretrained=pretrained)
        self.feature_len = 512
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

    def forward(self, x):
        # feature maps
        x = self.features(x)
        # flatten
        x = x.view(x.size(0), -1)
        return x

class MVCNN(nn.Module):
    def __init__(self, n_class, n_view, pretrained=False):
        super(MVCNN, self).__init__()
        self.n_view = n_view
        self.ft_net = FeatureNet(pretrained=pretrained)
        ##### fts fusion
        self.middle_fusion = nn.Linear(1024,512)
        self.middle_bn = nn.BatchNorm1d(512)

        self.cls_net = nn.Linear(self.ft_net.feature_len, n_class)

    def forward(self, view_batch, global_ft=False, local_ft=False,middle_fts = None):
        assert view_batch.size(1) == self.n_view
        view_batch = view_batch.view(-1, view_batch.size(2), view_batch.size(3), view_batch.size(4))
        view_fts = self.ft_net(view_batch)
        local_view_fts = view_fts.view(-1, self.n_view, view_fts.size(-1))
        global_view_fts, _ = local_view_fts.max(dim=1)
        ### global_view_fts 512  middle_fts 512
        #### feature fusion
        if middle_fts is not None:
            global_view_fts = F.relu(self.middle_bn(self.middle_fusion(torch.cat([global_view_fts,middle_fts],dim=1))))
        ####
        outputs = self.cls_net(global_view_fts)
        if global_ft and local_ft:
            return outputs, global_view_fts, local_view_fts
        elif global_ft:
            return outputs, global_view_fts
        elif local_ft:
            return outputs, local_view_fts
        else:
            return outputs
