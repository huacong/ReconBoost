## shared package
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
import json
import scipy
import random
from torch.utils.data import DataLoader
import sys
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import warnings
from pathlib import Path

## models
### CREMA-D,AVE
from models.CREMA.basic_model import AudioNet, VisualNet
### ModelNet40
from models.MView40.image import FeatureNet

## dataloader
from loaders.CramedDataset import CramedDataset
from loaders.AVEDataset import AVEDataset
from loaders.MView40.train_dataset import MView_train
from loaders.MView40.test_dataset import MView_test

#from loaders.MSADataset import MSADataset

##### Shared import 
from models.dynamic_net import DynamicNet

from schedule import schedule_model
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, res2tab, acc_score, map_score
from utils import check_status

import argparse
# os environment
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda")

def parse_args():
    parser = argparse.ArgumentParser(description="ICML-2024-ReconBoost")
    parser.add_argument('--dataset', type=str, default='CREMAD')
    parser.add_argument('--dataset_path',type=str, default='/data/huacong/CREMA/data')
    parser.add_argument('--n_class',type=int, default=6)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--boost_rate',type=float,default=1.0)
    parser.add_argument('--n_worker',type=int,default=8)
    parser.add_argument('--epochs_per_stage',type=int,default=4)
    parser.add_argument('--correct_epoch',type=int,default=4)
    parser.add_argument('--common_space',type=bool,default=True)

    parser.add_argument('--use_lr',type=bool,default=True) ##是否调整learning rate
    parser.add_argument('--m_lr',type=float,default=0.01)
    parser.add_argument('--e_lr',type=float,default=0.01)

    parser.add_argument('--use_br',type=bool,default=True)
    
    parser.add_argument('--use_pretrain',action="store_true")
    parser.add_argument('--m1ckpt', type=str, default = '/data/huacong/CREMA/OGM-GE/ckpt/CREMAD_audio_encoder_of_best_model_epoch_21_acc_0.5667613636363636.pth')
    parser.add_argument('--m2ckpt', type=str, default = '/data/huacong/CREMA/OGM-GE/ckpt/CREMAD_visual_encoder_of_best_model_epoch_87_acc_0.5198863636363636.pth')

    parser.add_argument('--use_ga',type=bool,default=True)

    parser.add_argument('--weight1',type=float,default=5.0)
    parser.add_argument('--weight2',type=float,default=1.0)
    parser.add_argument('--alpha',type=float,default=0.5)

    #### save dir & tensorboard dir
    parser.add_argument('--ckpt_dir',type=str,default='/data/huacong/MN40/grownet/cache/ckpt')
    parser.add_argument('--use_tensorboard',type=bool,default=True)
    parser.add_argument('--tensorboard_dir',type=str,default='/data/huacong/MN40/grownet/cache/tensorboard') 

    ################# test_mode ########################
    parser.add_argument('--ensemble_ckpt_path',type=str,default='')
    parser.add_argument('--uni_ckpt_path',type=str,default='')

    args = parser.parse_args()

    return args


def setup_seed():
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print(f"random seed: {seed}")

def init_model():
    return random.random()

def init_pretrain(model,ckpt):
    model_ckpt = torch.load(ckpt)
    pretrained_dict = model_ckpt['model']
    model.load_state_dict(pretrained_dict)
    return model

def test(args,data_loader, net_ensemble, stage = 0):
    print(f"Stage {stage}, Testing...")
    all_lbls, all_preds = [], []
    st = time.time()

    for i, data_label in enumerate(data_loader):
        if args.dataset == 'CREMAD' or args.dataset == 'AVE':
            spec, image, lbl = data_label
            spec = spec.cuda()
            spec = spec.unsqueeze(1).float()
            image = image.to(device)
            image = image.float()
            lbl = lbl.cuda()
            data = (spec,image)
        elif args.dataset == 'MView40':
            img_1, img_2, lbl = data_label
            img_1 = img_1.cuda()
            img_2 = img_2.cuda()
            lbl = torch.squeeze(lbl.cuda())
            data = (img_1, img_2)
            
       
        out_join = net_ensemble.forward(data)
        out_join = F.softmax(torch.as_tensor(out_join, dtype=torch.float32).cuda(),dim=1)

        _ , preds = torch.max(out_join, 1)
        
        
        all_preds.extend(preds.squeeze().detach().cpu().numpy().tolist())
        all_lbls.extend(lbl.squeeze().detach().cpu().numpy().tolist())
    
    print(len(all_lbls), len(all_preds))
    acc_mi = acc_score(all_lbls, all_preds, average="micro")
    acc_ma = acc_score(all_lbls, all_preds, average="macro")

    print(f"Stage: {stage}, Time: {time.time()-st:.4f}s")
    res = {
            "overall acc": acc_mi,
            "meanclass acc": acc_ma
        }
    tab_head, tab_data = res2tab(res)
    print(tab_head)
    print(tab_data)
    print("This Stage Done!\n")     
    return acc_mi


def main(args,this_task):
    setup_seed()

    ### tensorboard
    writer_path = os.path.join(args.tensorboard_dir,this_task)
    if not os.path.exists(writer_path):
        os.makedirs(writer_path)
    writer = SummaryWriter(writer_path)

    ckpt_path = os.path.join(args.ckpt_dir,this_task)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    ## dataset and model
    if args.dataset == 'CREMAD': # audio + visual
        train_data = CramedDataset(dataset_dir=args.dataset_path,mode="train")
        test_data = CramedDataset(dataset_dir=args.dataset_path,mode="test")
        #### define models
        model_audio = AudioNet(dataset='CREMAD')
        model_visual = VisualNet(dataset='CREMAD')

        audio_ckpt_path = args.m1ckpt
        visual_ckpt_path = args.m2ckpt
        # pre-trained uni-modal model can help converge
        if args.use_pretrain:
            model_audio = init_pretrain(model_audio,audio_ckpt_path)
            model_visual = init_pretrain(model_visual,visual_ckpt_path)
        
        model_audio.cuda()    
        model_visual.cuda()

    elif args.dataset == 'AVE':
        train_data = AVEDataset(dataset_dir=args.dataset_path,mode="train")
        test_data = AVEDataset(dataset_dir=args.dataset_path,mode='test')
        
        model_audio = AudioNet(dataset='AVE')
        model_visual = VisualNet(dataset='AVE')

        audio_ckpt_path = args.m1ckpt
        visual_ckpt_path = args.m2ckpt

        if args.use_pretrain:
            model_audio = init_pretrain(model_audio,audio_ckpt_path)
            model_visual = init_pretrain(model_visual,visual_ckpt_path)

        model_audio.cuda()
        model_visual.cuda()

    elif args.dataset == 'MView40':
        train_data = MView_train(dataset_dir=args.dataset_path, phase="train")
        test_data = MView_test(dataset_dir=args.dataset_path, phase="test")
        
        model_img1 = FeatureNet(output_dim=args.n_class)
        model_img1 = nn.DataParallel(model_img1)
        model_img2 = FeatureNet(output_dim=args.n_class)
        model_img2 = nn.DataParallel(model_img2)
        model_img1.cuda()
        model_img2.cuda()

        img_ckpt_path = args.m1ckpt

        
        if args.use_pretrain:
            model_img1.load_state_dict(torch.load(img_ckpt_path))
            model_img2.load_state_dict(torch.load(img_ckpt_path))


    train_loader = DataLoader(dataset=train_data,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.n_worker,
                            drop_last=True)
    test_loader = DataLoader(dataset=test_data,
                            batch_size=args.batch_size,
                            shuffle=False,
                            num_workers=args.n_worker,
                            drop_last=True)
    

    net_ensemble = DynamicNet(
        c0 = init_model(), 
        lr = args.boost_rate, 
        common_space = args.common_space,
        dataset = args.dataset,
        n_class= args.n_class)
    

    if args.use_pretrain:
        if args.dataset == 'AVE' or args.dataset == 'CREMAD':
            net_ensemble.add(model = model_audio,model_name='audio')
            net_ensemble.add(model = model_visual,model_name='visual')
        elif args.dataset == 'MView40':
            net_ensemble.add(model = model_img1, model_name='img_1')
            net_ensemble.add(model = model_img2, model_name='img_2')

    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    stage = 0
    models_name = []
    
    best_acc = 0.0
    
    while check_status(stage):
        model_name = schedule_model(stage=stage,dataset=args.dataset)        
        print(f"Stage {stage}, pick {model_name} modality........")
        
        if stage == 0:
            modality_lr = args.m_lr
            ensemble_lr = args.e_lr
       
        if args.dataset == 'CREMAD' or args.dataset == 'AVE':
            if model_name == 'audio':
                model = model_audio ## current model
                pre_model = model_visual
                pre_model_name = 'visual'
            else: #model_name == 'visual':
                model = model_visual
                pre_model = model_audio
                pre_model_name = 'audio'
        elif args.dataset == 'MView40':
            if model_name == 'img_1':
                model = model_img1
                pre_model = model_img2
                pre_model_name = 'img_2'
            elif model_name == 'img_2':
                model = model_img2
                pre_model = model_img1
                pre_model_name = 'img_1'

        
        if args.common_space:
            model.fc = net_ensemble.head


        net_ensemble.to_train()
        ce_loss = []


        # phase 1
        optimizer = optim.SGD(model.parameters(), lr=modality_lr, momentum=0.9, weight_decay=5e-4)
        
        net_ensemble.to_train()

        stage_loss = []
        stage_ga_loss = []
        ce_loss = []

        for epoch in range(args.epochs_per_stage):
            for i, data_label in enumerate(train_loader):
                if args.dataset == 'CREMAD' or args.dataset == 'AVE':
                    spec, image, lbl = data_label
                    spec = spec.cuda()
                    spec = spec.unsqueeze(1).float()
                    image = image.cuda()
                    image = image.float()
                    lbl = lbl.cuda()
                    data = (spec, image)
                    model_input_map = {
                        'audio': spec,
                        'visual': image
                    }
                elif args.dataset == 'MView40':
                    img_1, img_2, lbl = data_label
                    img_1 = img_1.cuda()
                    img_2 = img_2.cuda()
                    lbl = torch.squeeze(lbl.cuda())
                    data = (img_1, img_2)
                    model_input_map = {
                        'img_1': img_1,
                        'img_2': img_2
                    }
                
                out_join = net_ensemble.forward(data=data, mask_model=model_name) ## mask_model = model_name global_ft: true or false
               
                if not args.use_pretrain and stage == 0: # initial
                    out_join = torch.as_tensor(out_join, dtype=torch.float32).cuda().view(-1, 1).expand(args.batch_size,1)
                else:
                    out_join = torch.as_tensor(out_join, dtype=torch.float32).cuda()

                out_obj = model(model_input_map[model_name])
                target = torch.zeros(args.batch_size,args.n_class).cuda().scatter_(1,lbl.view(-1,1),1)

                boosting_loss = - args.weight1 *  (target * out_obj.log_softmax(1)).mean(-1) \
                            + args.weight2 * (target*out_join.detach().softmax(1) * out_obj.log_softmax(1)).mean(-1)
                
                model.zero_grad()

                if args.use_ga:
                    if stage == 0:
                        loss = boosting_loss
                    else:
                        pre_out_obj = pre_model(model_input_map[pre_model_name])
                        ga_loss = mse_criterion(out_obj.detach().softmax(1), pre_out_obj.detach().softmax(1)) ## ga loss
                        stage_ga_loss.append(ga_loss.mean().item())
                        loss = boosting_loss + args.alpha * ga_loss
                    loss.mean().backward()
                else:
                    boosting_loss.mean().backward()
                
                optimizer.step()  
                stage_loss.append(boosting_loss.mean().item())
             
        stage_mean_loss = np.mean(stage_loss).item()
        stage_mean_ga_loss = np.mean(stage_ga_loss).item()

        models_name = net_ensemble.get_model_name()
        print(f"There are {len(models_name)} modality(ies),{models_name}",)
        print(f"Adding {model_name} modality.....")

        net_ensemble.add(model,model_name)
        

        ## phase 2
        if stage >= 0: 
            optimizer_correct = optim.SGD(net_ensemble.parameters(), ensemble_lr, momentum=0.9, weight_decay=5e-4)
            for epoch in range(args.correct_epoch):
                for i, data_label in enumerate(train_loader):
                    if args.dataset == 'CREMAD' or args.dataset == 'AVE':
                        spec, image, lbl = data_label
                        spec = spec.cuda()
                        spec = spec.unsqueeze(1).float()
                        image = image.cuda()
                        image = image.float()
                        lbl = lbl.cuda()
                        data = (spec, image)
                        model_input_map = {
                            'audio': spec,
                            'visual': image
                        }
                    elif args.dataset == 'MView40':
                        img_1, img_2, lbl = data_label
                        img_1 = img_1.cuda()
                        img_2 = img_2.cuda()
                        lbl = torch.squeeze(lbl.cuda())
                        data = (img_1, img_2)
                        model_input_map = {
                            'img_1': img_1,
                            'img_2': img_2
                        }
                    
                    out = net_ensemble.forward_grad(data)
                    loss = ce_criterion(out, lbl)
                    optimizer_correct.zero_grad()
                    loss.backward()
                    ce_loss.append(loss.item())
                    optimizer_correct.step()
            
        ce_mean_loss = np.mean(ce_loss).item()
        print('Results from stage := ' + str(stage) + '\n')
        
        net_ensemble.to_eval()
        acc = test(args, test_loader, net_ensemble, stage)

        ###################### write in tensorboard
        if args.use_tensorboard:
            writer.add_scalar('Train/Boosting Loss', stage_mean_loss, stage)
            writer.add_scalar('Train/GA Loss', stage_mean_ga_loss, stage)
            writer.add_scalar('Train/Net Loss', ce_mean_loss, stage)
            writer.add_scalar('Evaluation/ACC', acc, stage)
            writer.add_scalar('Train/Modality_lr',modality_lr,stage)
            writer.add_scalar('Train/Ensemble_lr',ensemble_lr,stage)
        
        ########## save model    
        if stage >= 3 and acc > best_acc: 
            best_acc = float(acc)
            uni_model_name = 'uni_encoder_of_best_model_stage_{}_acc_{}.pth'.format(stage, acc)
            if args.dataset == 'CREMAD' or args.dataset == 'AVE':
                saved_dict = {'saved_stage':stage,
                        'acc':acc,
                        'model_audio':model_audio.state_dict(),
                        'model_visual':model_visual.state_dict()
                        }  
            elif args.dataset == 'MView40':
                saved_dict =  {'saved_stage':stage,
                        'acc':acc,
                        'model_audio':model_img1.state_dict(),
                        'model_visual':model_img2.state_dict()
                        }
            
            uni_save_path = os.path.join(ckpt_path, uni_model_name)
            torch.save(saved_dict, uni_save_path)
            print('The uni encoder of the best model has been saved at {}'.format(uni_model_name))
            ensemble_net_name = 'best_ensemble_net_stage_{}_acc_{}.path'.format(stage,acc)
            ensemble_save_path = os.path.join(ckpt_path, ensemble_net_name)
            net_ensemble.to_file(ensemble_save_path)

        stage = stage + 1

        if args.use_lr:
            if args.dataset == 'AVE' and stage == 40:
                modality_lr = modality_lr * 0.5
                ensemble_lr = ensemble_lr * 0.5
            if args.dataset == 'CREMAD' and stage !=0 and stage % 30 == 0:
                modality_lr = modality_lr * 0.1
                ensemble_lr = ensemble_lr * 0.1
    

if __name__ == "__main__":
    args = parse_args()
    print(args)
   
    all_st = time.time()
    this_task = f'{args.dataset}'
    main(args=args,this_task=this_task)
    all_sec = time.time()-all_st
    print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
    