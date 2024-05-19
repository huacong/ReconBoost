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
import random
from torch.utils.data import DataLoader
import sys
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
import warnings
from pathlib import Path
### MOSEI, MOSI, SIMS
from models.MOSEI.LF_DNN import MAudioNet, MVisualNet, MTextNet

## dataloader
from loaders.MOSEIDataset import MMDataLoader

#from loaders.MSADataset import MSADataset

##### Shared import 
from models.dynamic_net import DynamicNet

from schedule import schedule_model
from torch.utils.tensorboard import SummaryWriter

from utils import res2tab, acc_score, map_score
from utils import check_status

import argparse
# os environment
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
device = torch.device("cuda")


def parse_args():
    parser = argparse.ArgumentParser(description="ICML-2024-ReconBoost")
    parser.add_argument('--dataset', type=str, default='MSA')
    
    parser.add_argument('--dataset_name', type=str, default='mosei')
    parser.add_argument('--featurePath', type=str, default='/data/huacong/MSA/MOSEI/Processed/unaligned_50.pkl')
    parser.add_argument('--seq_lens', type=int, nargs='+', default=[50, 1, 1])
    parser.add_argument('--feature_dims', type=int, nargs='+', default=[768, 74, 35])
    parser.add_argument('--need_data_aligned', type=bool, default=False)


    parser.add_argument('--dataset_path',type=str, default='/data/huacong/CREMA/data')
    parser.add_argument('--n_class',type=int, default=6)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--boost_rate',type=float,default=1.0)
    parser.add_argument('--n_worker',type=int,default=8)
    parser.add_argument('--epochs_per_stage',type=int,default=4)
    parser.add_argument('--correct_epoch',type=int,default=4)
    parser.add_argument('--global_ft',type=bool,default=True)
    parser.add_argument('--fixed',type=bool,default=True)
    parser.add_argument('--common_space',type=bool,default=True)
    parser.add_argument('--train_mode',type=bool,default=True)
    parser.add_argument('--MASK',type=bool,default=True)
    parser.add_argument('--use_val',type=bool,default=False)

    parser.add_argument('--use_lr',type=bool,default=True) ##调整learning rate
    parser.add_argument('--m_lr',type=float,default=0.01)
    parser.add_argument('--e_lr',type=float,default=0.01)

    parser.add_argument('--use_br',type=bool,default=True)
    parser.add_argument('--use_pretrain',type=bool,default=False)
    parser.add_argument('--use_ga',type=bool,default=True)
    parser.add_argument('--ga_mse',type=bool,default=True)
    parser.add_argument('--ga_cos',type=bool,default=False)
    parser.add_argument('--use_fusion',type=bool,default=False)
    parser.add_argument('--weight1',type=float,default=1.0)
    parser.add_argument('--weight2',type=float,default=0.25)
    parser.add_argument('--alpha',type=float,default=0.5)


    ################# test_mode ########################
    parser.add_argument('--ensemble_ckpt_path',type=str,default='')
    parser.add_argument('--uni_ckpt_path',type=str,default='')

    args = parser.parse_args()

    return args
################SAVE Dir checkpoint and tensorboard################################
this_task = f''
out_dir = Path('cache')
save_dir = out_dir/'Schedule'/this_task
save_dir.mkdir(parents=True, exist_ok=True)
tensorboard_dir = out_dir/'Schedule'/this_task
tensorboard_dir.mkdir(parents=True, exist_ok=True)
use_tensorboard = True
################# ModelNet config###############################

def setup_seed():
    seed = 0 # 594216.2640838623
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

def test(args, data_loader, net_ensemble, stage = 0):
    print(f"Stage {stage}, Testing...")
    all_lbls, all_preds = [], []
    
    st = time.time()

    for i, data_label in enumerate(data_loader):
        
        vision = data_label['vision'].cuda()
        audio = data_label['audio'].cuda()
        text = data_label['text'].cuda()
        lbl = data_label['labels']['M'].cuda()
        lbl = lbl.view(-1).long()
        data = (vision, audio, text)
            

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


def main(args):
    setup_seed()
    writer_path = os.path.join(tensorboard_dir,this_task)
    writer = SummaryWriter(writer_path)
    
    mosei_dataloader = MMDataLoader(args=args)
    train_loader = mosei_dataloader['train']
    test_loader = mosei_dataloader['test']

    model_text = MTextNet()
    model_audio = MAudioNet()
    model_visual = MVisualNet()

    model_audio.cuda()
    model_visual.cuda()
    model_text.cuda()

    ckpt_text = '/data/huacong/MMSA_Code/cache/ckpt/text/mosei_text_encoder_of_best_model_epoch_3_acc_0.6654195617316943.pth'
    ckpt_audio = '/data/huacong/MMSA_Code/cache/ckpt/audio/mosei_audio_encoder_of_best_model_epoch_22_acc_0.5269909139497595.pth'
    ckpt_visual = '/data/huacong/MMSA_Code/cache/ckpt/visual/mosei_visual_encoder_of_best_model_epoch_2_acc_0.518439337252806.pth'

    net_ensemble = DynamicNet(
        c0 = init_model(), 
        lr = args.boost_rate, 
        common_space = args.common_space,
        dataset=args.dataset,
        n_class=args.n_class)

    if args.use_pretrain:
        model_text.load_state_dict(torch.load(ckpt_text))
        model_audio.load_state_dict(torch.load(ckpt_audio))        
        model_visual.load_state_dict(torch.load(ckpt_visual))    
        net_ensemble.add(model = model_text, model_name='text')
        net_ensemble.add(model = model_visual, model_name='visual')
        net_ensemble.add(model = model_audio, model_name='audio')

    ce_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()

    stage = 0
    models_name = []

    best_acc = 0.0

    
    while check_status(stage):
        model_name = schedule_model(stage = stage, dataset = args.dataset)        
        print(f"Stage {stage}, choose {model_name} modality........")
        
        if stage == 0:
            modality_lr = args.m_lr
            ensemble_lr = args.e_lr
        

        if model_name == 'text':
            model = model_text
            pre_model = model_audio
            pre_model_name = 'audio'
        elif model_name == 'visual':
            model = model_visual
            pre_model = model_text
            pre_model_name = 'text'
        elif model_name == 'audio':
            model = model_audio
            pre_model = model_visual
            pre_model_name = 'visual'

        
        if args.common_space:
            model.fc = net_ensemble.head

        net_ensemble.to_train()
        ce_loss = []
        # alternating stage
        optimizer = optim.SGD(model.parameters(), modality_lr, momentum=0.9, weight_decay=5e-4)
        
        net_ensemble.to_train()
        stage_loss = []
        stage_ga_loss = []
        ce_loss = []

        for epoch in range(args.epochs_per_stage):
            for i, data_label in enumerate(train_loader):
                vision = data_label['vision'].cuda()
                audio = data_label['audio'].cuda()
                text = data_label['text'].cuda()
                lbl = data_label['labels']['M'].cuda()
                lbl = lbl.view(-1).long()
                data = (vision, audio, text)
                model_input_map = {
                    'text': text,
                    'audio': audio,
                    'visual': vision
                }
                out_join = net_ensemble.forward(data, mask_model=model_name) ## mask_model = model_name global_ft: true or false

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

        net_ensemble.add(model, model_name)
        
        # global rectification scheme 
        if stage >= 0: # if stage ! = 0
            optimizer_correct = optim.SGD(net_ensemble.parameters(), ensemble_lr, momentum=0.9, weight_decay=5e-4)
            for epoch in range(args.correct_epoch):
                for i, data_label in enumerate(train_loader):
                    vision = data_label['vision'].cuda()
                    audio = data_label['audio'].cuda()
                    text = data_label['text'].cuda()
                    lbl = data_label['labels']['M'].cuda()
                    lbl = lbl.view(-1).long()
                    data = (vision, audio, text)
                    model_input_map = {
                        'text': text,
                        'audio': audio,
                        'visual': vision
                    }
                    out = net_ensemble.forward_grad(data)
                    loss = ce_criterion(out, lbl)
                    optimizer_correct.zero_grad()
                    loss.backward()
                    ce_loss.append(loss.item())
                    optimizer_correct.step()
            
        ce_mean_loss = np.mean(ce_loss).item()
        print('Results from stage := ' + str(stage) + '\n')
        
        
        #### phase 2
        
        net_ensemble.to_eval()
       
        acc = test(args, test_loader, net_ensemble, stage)

        ######################
        if use_tensorboard:
            writer.add_scalar('Train/Boosting Loss', stage_mean_loss, stage)
            writer.add_scalar('Train/GA Loss', stage_mean_ga_loss, stage)
            writer.add_scalar('Train/Net Loss', ce_mean_loss, stage)
            writer.add_scalar('Evaluation/ACC', acc, stage)
            writer.add_scalar('Train/Modality_lr',modality_lr,stage)
            writer.add_scalar('Train/Ensemble_lr',ensemble_lr,stage)
            
        if stage >= 3 and acc > best_acc: # and acc > 0.91:
            best_acc = float(acc)
            uni_model_name = 'uni_encoder_of_best_model_stage_{}_acc_{}.pth'.format(stage, acc)
            saved_dict = {
                'saved_stage':stage,
                'acc':acc,
                'model_audio':model_audio.state_dict(),
                'model_visual':model_visual.state_dict(),
                'model_text':model_text.state_dict()
            }
            uni_save_path = os.path.join(save_dir, uni_model_name)
            torch.save(saved_dict, uni_save_path)
            print('The uni encoder of the best model has been saved at {}'.format(uni_model_name))
            ensemble_net_name = 'best_ensemble_net_stage_{}_acc_{}.path'.format(stage,acc)
            ensemble_save_path = os.path.join(save_dir,ensemble_net_name)
            net_ensemble.to_file(ensemble_save_path)

        stage = stage + 1

if __name__ == "__main__":

    args = parse_args()
    print(args)
    if args.train_mode:
        print("train_mode")   
        all_st = time.time()
        main(args=args)
        all_sec = time.time()-all_st
        print(f"Time cost: {all_sec//60//60} hours {all_sec//60%60} minutes!")
    else:
        print("test_mode")
        setup_seed()
        all_st = time.time()
        ### model
        ensemble_ckpt_path = ''
        d = torch.load(ensemble_ckpt_path)
        
        net = DynamicNet(
            c0 = d['c0'],
            lr = d['lr'],
            common_space = d['common_space'],
            dataset = args.dataset,
            n_class = args.n_class)
        
        if d['common_space']:
            net.head.load_state_dict(d['head'])
        
        uni_ckpt_path = ''

        uni = torch.load(uni_ckpt_path)
        

        model_text = MTextNet()
        model_text.cuda()
        model_text.fc = net.head
        model_audio = MAudioNet()
        model_audio.cuda()
        model_audio.fc = net.head
        model_visual = MVisualNet()
        model_visual.cuda()
        model_visual.fc = net.head
        for _, m_name in enumerate(d['models_name']):
            if m_name == 'audio':
                model_audio.load_state_dict(uni['model_audio'])
                net.add(model=model_audio,model_name='audio')
            elif m_name == 'visual':
                model_visual.load_state_dict(uni['model_visual'])
                net.add(model=model_visual,model_name='visual')
            elif m_name == 'text':
                model_text.load_state_dict(uni['model_text'])
                net.add(model=model_text,model_name='text')
        mosei_dataloader = MMDataLoader(args=args)
        test_loader = mosei_dataloader['test']

        net.to_eval()
        acc = test(test_loader,net,0)
