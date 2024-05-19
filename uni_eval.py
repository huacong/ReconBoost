import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch.optim as optim
import time

from torch.utils.data import DataLoader
from models.CREMA.basic_model import AudioNet, VisualNet

from loaders.CramedDataset import CramedDataset
from loaders.AVEDataset import AVEDataset
import time
from utils import res2tab, acc_score, map_score

os.environ["CUDA_VISIBLE_DEVICES"] = '6'
device = torch.device("cuda")


def parse_args():
    parser = argparse.ArgumentParser(description="Uni-modal-evaluation")
    parser.add_argument('--dataset', type=str, default='AVE')
    parser.add_argument('--dataset_path',type=str, default='/data/huacong/AVE_Dataset')
    parser.add_argument('--modality',type=str,default='visual') # visual or audio
    parser.add_argument('--n_class',type=int, default=28)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--max_epochs',type=int,default=100)
    parser.add_argument('--emb',type=int,default=512)
    parser.add_argument('--uni_ckpt_path',type=str,default='/data/huacong/MN40/grownet/cache/Trained_ckpt/Trained_Common_Space_AVE_boost_4_4_fts_common_space_fixed_mask_lr_0.01_0.01_br__GA_mse_4.0_1.0_2/uni_encoder_of_best_model_stage_34_acc_0.7135416666666666.pth')
    args = parser.parse_args()

    return args


class Classifier(nn.Module):
    def __init__(self, input_dim=512, output_dim=40):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, fts):
        output = self.fc(fts)
        return output
    
def test(args, model, cls, test_dataloader, epoch):
    print(f"Epoch {epoch}, Testing...")
    all_lbls, all_preds = [], []
    st = time.time()
    
    for i, (spec, image, label) in enumerate(test_dataloader):
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            if args.modality == "audio":
                _, gf = model(spec.unsqueeze(1).float(), global_ft = True)            
            else:
                _, gf = model(image.float(), global_ft = True)
        out = cls(gf)
        out = torch.as_tensor(out, dtype=torch.float32).cuda()
        _ , preds = torch.max(out, 1)

        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(label.squeeze().detach().cpu().numpy().tolist())
        
    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")
    print(f"Stage: {epoch}, Time: {time.time()-st:.4f}s")
    res = {
            "overall acc": acc_mi,
            "meanclass acc": acc_ma,
        }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Stage Done!\n")   
    return acc_mi          
    

def main(args):
    if args.dataset == 'CREMAD':
        train_data = CramedDataset(dataset_dir = args.dataset_path, mode="train")
        test_data = CramedDataset(dataset_dir = args.dataset_path, mode="test")
    elif args.dataset == 'AVE':
        train_data = AVEDataset(dataset_dir = args.dataset_path, mode="train")
        test_data = AVEDataset(dataset_dir = args.dataset_path, mode='test')
    else:
        raise NotImplementedError('Incorrect dataset name {}! '
                                  'Only support VGGSound and CREMA-D for now!'.format(args.dataset))

    train_dataloader = DataLoader(train_data, batch_size = args.batch_size,
                                  shuffle=True, num_workers=32, pin_memory=True, drop_last=True)

    test_dataloader = DataLoader(test_data, batch_size = args.batch_size,
                                 shuffle=False, num_workers=32, pin_memory=True, drop_last=True)

    ##### uni-modality
    uni_ckpt = torch.load(args.uni_ckpt_path)
    if args.modality == "audio":
        model = AudioNet(dataset = args.dataset)
        pretrained_dict = uni_ckpt['model_audio']
        model_dict = model.state_dict()
        pretrained_dict = { k: v for k, v in pretrained_dict.items() if not 'fc' in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        model = VisualNet(dataset = args.dataset)
        pretrained_dict = uni_ckpt['model_visual']
        model_dict = model.state_dict()
        pretrained_dict = { k: v for k, v in pretrained_dict.items() if not 'fc' in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model.cuda()
    model.eval()

    cls = Classifier(input_dim = args.emb, output_dim = args.n_class)
    cls.cuda()
    optimizer = optim.SGD(cls.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
    ce_criterion = nn.CrossEntropyLoss()

    for epoch in range(args.max_epochs):
        cls.train()
        for i, data_label in enumerate(train_dataloader):
            spec, image, lbl = data_label
            spec = spec.cuda()
            spec = spec.unsqueeze(1).float()
            image = image.cuda()
            image = image.float()
            lbl = lbl.cuda()
            with torch.no_grad():
                if args.modality == "audio":
                    _, gf = model(spec,global_ft = True)            
                else:
                    _, gf = model(image,global_ft = True)
            out = cls(gf)    
            loss = ce_criterion(out,lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Results from epoch = {epoch}" + '\n')
        model.eval()
        cls.eval()
        test(args, model, cls, test_dataloader, epoch)

if __name__ == "__main__":
    args = parse_args()
    main(args)
