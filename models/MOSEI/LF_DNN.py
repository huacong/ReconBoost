"""
paper1: Benchmarking Multimodal Sentiment Analysis
paper2: Recognizing Emotions in Video Using Multimodal DNN Feature Fusion
From: https://github.com/rhoposit/MultimodalDNN
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .subNets.FeatureNets import SubNet, TextSubNet

__all__ = ['LF_DNN']
class LF_DNN(nn.Module):
    """
    late fusion using DNN
    """
    def __init__(self, args):
        super(LF_DNN, self).__init__()
        print("model:")
        print(args)
        '''
        {'model_name': 'lf_dnn', 'dataset_name': 'mosei', 'featurePath': '/data/huacong/MSA_Datasets/MOSEI/Processed/aligned_50.pkl', 
        'seq_lens': 50, 'feature_dims': [768, 74, 35], 'train_samples': 16326, 'num_classes': 3, 
        'language': 'en', 'KeyEval': 'Loss', 'missing_rate': [0.2, 0.2, 0.2], 'missing_seed': [1111, 1111, 1111], 
        'need_data_aligned': True, 'need_model_aligned': True, 'need_normalized': True, 'early_stop': 8, 
        'use_bert': False, 'hidden_dims': [128, 128, 128], 'text_out': 128, 'post_fusion_dim': 128, 
        'dropouts': [0.2, 0.2, 0.2, 0.2], 'batch_size': 64, 'learning_rate': 0.002, 'weight_decay': 0.001, 
        'model_save_path': PosixPath('/home/huacong/MMSA/saved_models/lf_dnn-mosei.pth'), 
        'device': device(type='cuda', index=1), 'train_mode': 'classification',
          'custom_feature': None, 'feature_T': None, 'feature_A': None, 'feature_V': None, 'cur_seed': 1}

        '''
        '''
        'feature_dims': [768, 74, 35],
        'hidden_dims': [128, 128, 128],
        'dropouts': [0.2, 0.2, 0.2, 0.2]
        [768, 33, 709]
        '''
        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden = args.hidden_dims

        self.text_out= args.text_out
        self.post_fusion_dim = args.post_fusion_dim

        self.audio_prob, self.video_prob, self.text_prob, self.post_fusion_prob = args.dropouts

        output_dim = args.num_classes if args.train_mode == "classification" else 1

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)
        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.post_fusion_layer_1 = nn.Linear(self.text_out + self.video_hidden + self.audio_hidden, self.post_fusion_dim)
        # self.post_fusion_layer_2 = nn.Linear(self.post_fusion_dim, self.post_fusion_dim)
        self.post_fusion_layer_3 = nn.Linear(self.post_fusion_dim, output_dim)

    def forward(self, text_x, audio_x, video_x):
        audio_x = audio_x.squeeze(1)
        video_x = video_x.squeeze(1)

        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        text_h = self.text_subnet(text_x)

        fusion_h = torch.cat([audio_h, video_h, text_h], dim=-1)
        x = self.post_fusion_dropout(fusion_h)
        x = F.relu(self.post_fusion_layer_1(x), inplace=True)
        # x = F.relu(self.post_fusion_layer_2(x), inplace=True)
        output = self.post_fusion_layer_3(x)

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,
            'Feature_f': fusion_h,
            'M': output
        }
        return res
    

class MAudioNet(nn.Module):
    def __init__(self,audio_in = 74, audio_hidden = 128, n_classes = 3):
        super(MAudioNet, self).__init__()
        self.audio_prob = 0.2
        self.audio_in = audio_in
        self.audio_hidden = audio_hidden
       
        self.audio_out = n_classes
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        # #### new
        # self.audio_hidden2 = 512
        # self.projection = nn.Linear(self.audio_hidden, self.audio_hidden2)
        # self.fc = nn.Linear(self.audio_hidden2,self.audio_out)
        # # self.fc = nn.Linear(self.audio_hidden,self.audio_out)

        #### old version
        self.fc = nn.Linear(self.audio_hidden,self.audio_out)

        #### fusion 
        #self.fusion = nn.Linear(self.audio_hidden *2, self.audio_hidden)

    def forward(self,x, global_ft = False, middle_fts = None):
        audio = x
        audio_x = audio.squeeze(1)
        audio_h = self.audio_subnet(audio_x)
        
        # ##### new
        # audio_h = self.projection(audio_h)
        # if middle_fts is not None:
        #     audio_h = self.fusion(torch.cat([audio_h,middle_fts],dim=1))
        
        audio_output = self.fc(audio_h)
        if global_ft:
            return audio_output, audio_h
        else:
            return audio_output
        
class MVisualNet(nn.Module):
    def __init__(self,visual_in = 35, visual_hidden = 128, n_classes = 3):
        super(MVisualNet, self).__init__()
        self.visual_prob = 0.2
        self.visual_in = visual_in
        self.visual_hidden = visual_hidden
        # #### new
        # self.visual_hidden2 = 512
        # #### new
        # self.projection = nn.Linear(self.visual_hidden, self.visual_hidden2)
        # self.fc = nn.Linear(self.visual_hidden2,self.visual_out)

        self.visual_out = n_classes
        self.visual_subnet = SubNet(self.visual_in, self.visual_hidden, self.visual_prob)

        # ### old version
        self.fc = nn.Linear(self.visual_hidden,self.visual_out)

        #### fusion 
        #self.fusion = nn.Linear(self.visual_hidden *2, self.visual_hidden)

    def forward(self,x, global_ft = False, middle_fts = None):
        visual = x
        visual_x = visual.squeeze(1)
        visual_h = self.visual_subnet(visual_x)

        # #### new
        # visual_h = self.projection(visual_h)
        # if middle_fts is not None:
        #     # print("visual_h.shape:",visual_h.shape)
        #     # print("middle_fts.shape:",middle_fts.shape)
        #     # visual_h.shape: torch.Size([64, 128])
        #     # middle_fts.shape: torch.Size([64, 128])
        #     visual_h = self.fusion(torch.cat([visual_h,middle_fts],dim=1))

        visual_output = self.fc(visual_h)

        if global_ft:
            return visual_output, visual_h
        else:
            return visual_output

class MTextNet(nn.Module):
    def __init__(self, text_in = 768, text_hidden = 128, text_out = 128, n_classes = 3):
        super(MTextNet, self).__init__()
        self.text_prob = 0.2
        self.text_in = text_in
        self.text_hidden = text_hidden
        self.text_out = text_out
        self.text_subnet = TextSubNet(self.text_in, self.text_hidden, self.text_out, dropout=self.text_prob)
        #### new
        # self.text_hidden2 = 512

        # #### new
        # self.projection = nn.Linear(self.text_out, self.text_hidden2)
        # self.fc = nn.Linear(self.text_hidden2,n_classes)

        # old
        self.fc = nn.Linear(self.text_out,n_classes)

        #### fusion 
        #self.fusion = nn.Linear(self.text_out *2, self.text_out)

    def forward(self, x, global_ft = False, middle_fts = None):
        text = x
        text_h = self.text_subnet(text)

        # ### new
        # text_h = self.projection(text_h)
        # if middle_fts is not None:
        #     text_h = self.fusion(torch.cat([text_h,middle_fts],dim=1))
        
        text_output = self.fc(text_h)

        if global_ft:
            return text_output, text_h
        else:
            return text_output
        

