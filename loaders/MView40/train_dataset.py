import os
import sys
import glob
import h5py
import json
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import torch
import numpy as np
from .image import load_img
from .util_dataset import get_obj_lbl_list

class MView_train(Dataset):
    def __init__(self, dataset_dir, phase ='train'):
        
        self.dataset_dir = dataset_dir
        self.phase = phase
        self.object_list, self.lbl_list = get_obj_lbl_list(self.dataset_dir+'/pt',phase=phase)


    def __getitem__(self, item):
        ### label
        lbl = self.lbl_list[item]
        ## item name
        item_name = self.object_list[item].split('/')  

        ### img
        v1, v2 = load_img(self.dataset_dir + '/image', item_name, self.phase == 'train')
        return v1, v2,  lbl
    

    def __len__(self):
        return len(self.object_list)

