import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import resnet18


class AudioNet(nn.Module):
    def __init__(self,dataset,latent_in = 512):
        super(AudioNet, self).__init__()
        self.audio_net = resnet18(modality="audio")
        if dataset == 'CREMAD':
            n_classes = 6
        elif dataset == 'AVE':
            n_classes = 28
        output_dim = n_classes

        self.fc = nn.Linear(latent_in, output_dim)
        
    def forward(self, audio, global_ft = False):
        a = self.audio_net(audio)
        a = F.adaptive_avg_pool2d(a,1)
        gf = torch.flatten(a,1)
        out = self.fc(gf)
        if global_ft:
            return out, gf
        return out
        
        
class VisualNet(nn.Module):
    def __init__(self, dataset,latent_in = 512):
        super(VisualNet, self).__init__()
        self.visual_net = resnet18(modality="visual")
        if dataset == 'CREMAD':
            n_classes  = 6
        elif dataset == 'AVE':
            n_classes = 28
        output_dim = n_classes
        self.fc = nn.Linear(latent_in, output_dim)
        
    def forward(self, visual, global_ft = False):
        v = self.visual_net(visual)
        (_, C, H, W) = v.size()
        v = v.view(visual.size(0), -1, C, H, W)
        v = v.permute(0, 2, 1, 3, 4)
        v = F.adaptive_avg_pool3d(v, 1)
        gf = torch.flatten(v, 1)
        out = self.fc(gf)
        if global_ft:
            return out, gf
        return out
            
        