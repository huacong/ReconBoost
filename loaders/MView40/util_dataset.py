import glob
import json
import h5py
import os
import numpy as np


def get_obj_lbl_list(root, phase = 'train'):
    
    obj_list = []
    lbl_list = []

    for h5_name in glob.glob(os.path.join(root, 'ply_data_%s*.h5'%phase)):
        split = h5_name[-4]
        json_name = os.path.join(root, f'ply_data_{phase}_{split}_id2file.json')
        with open(json_name) as json_file:
            obj_tmp = json.load(json_file)
        obj_list = obj_list + obj_tmp
        f = h5py.File(h5_name)
        label = f['label'][:].astype('int64')
        f.close()
        lbl_list.append(label)
    lbl_list = np.concatenate(lbl_list, axis=0)
    return obj_list, lbl_list