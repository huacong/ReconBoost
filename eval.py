## shared package
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import warnings
from pathlib import Path

## models
### CREMA-D,AVE
from models.CREMA.basic_model import AudioNet, VisualNet

## dataloader
from loaders.CramedDataset import CramedDataset
from loaders.AVEDataset import AVEDataset


from models.dynamic_net import DynamicNet

from utils import res2tab, acc_score, map_score
from utils import check_status
import argparse

# os environment
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
device = torch.device("cuda")

def setup_seed():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")

def test(args, data_loader, net_ensemble, stage = 0):
    print(f"Stage {stage}, Testing...")
    all_lbls, all_preds = [], []

    for i, data_label in enumerate(data_loader):
        spec, image, lbl = data_label
        spec = spec.cuda()
        spec = spec.unsqueeze(1).float()
        image = image.to(device)
        image = image.float()
        lbl = lbl.cuda()
        data = (spec,image)

        out_join = net_ensemble.forward(data = data)
        out_join = F.softmax(torch.as_tensor(out_join, dtype=torch.float32).cuda(),dim=1)
       
        _ , preds = torch.max(out_join, 1)
      

        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
     
  
    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")

    res = {
            "overall acc": acc_mi,
            "meanclass acc": acc_ma
        }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Stage Done!\n")     
    return acc_mi

def parse_args():
    parser = argparse.ArgumentParser(description="ICML-2024-ReconBoost")
    parser.add_argument('--dataset', type=str, default='AVE')
    parser.add_argument('--dataset_path',type=str, default='/data/huacong/CREMA/data')
    parser.add_argument('--n_class',type=int, default=28)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--n_worker',type=int,default=8)
    parser.add_argument('--ensemble_ckpt_path',type=str,default='/data/huacong/MN40/grownet/cache/Trained_ckpt/Trained_Common_Space_CREMAD_boost_4_4_fts_common_space_fixed_mask_lr_0.01_0.01_br__GA_mse_5.0_1.0_2/best_ensemble_net_stage_99_acc_0.8110795454545454.path')
    parser.add_argument('--uni_ckpt_path',type=str,default='/data/huacong/MN40/grownet/cache/Trained_ckpt/Trained_Common_Space_CREMAD_boost_4_4_fts_common_space_fixed_mask_lr_0.01_0.01_br__GA_mse_5.0_1.0_2/uni_encoder_of_best_model_stage_99_acc_0.8110795454545454.pth')

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    setup_seed()
    
    d = torch.load(args.ensemble_ckpt_path)
    uni = torch.load(args.uni_ckpt_path)

    net = DynamicNet(
        c0 = d['c0'],
        lr = d['lr'],
        common_space = d['common_space'],
        dataset = args.dataset,
        n_class = args.n_class)

    net.head.load_state_dict(d['head']) # d['common_space']=True
    
    model_audio = AudioNet(dataset=args.dataset)
    model_audio.cuda()
    model_audio.fc = net.head
    model_visual = VisualNet(dataset=args.dataset)
    model_visual.cuda()
    model_visual.fc = net.head

    for _, m_name in enumerate(d['models_name']):
        if m_name == 'audio':
            model_audio.load_state_dict(uni['model_audio'])
            net.add(model=model_audio,model_name='audio')
        elif m_name == 'visual':
            model_visual.load_state_dict(uni['model_visual'])
            net.add(model=model_visual,model_name='visual')

    if args.dataset == "CREMAD":
        test_data = CramedDataset(dataset_dir=args.dataset_path,mode="test")
    elif args.dataset == 'AVE':
        test_data = AVEDataset(dataset_dir=args.dataset_path,mode='test')
        
    ## dataset
    test_loader = DataLoader(dataset=test_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_worker,
                            drop_last=True)
    
    net.to_eval()
    acc = test(args, test_loader,net,0)
 