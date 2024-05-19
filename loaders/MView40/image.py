import random
from PIL import Image
import torch
from torchvision import transforms, datasets

def load_img(root, names, argument=False):

    if argument: ### phase == 'train'
        phase = 'train'
        transform = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:  ### phase == 'test'
        phase = 'test'
        transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

 
    v1_id = random.randint(0, 90)
    v2_id = random.randint(90, 179)
    v1_names = root+'/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], v1_id)
    v2_names = root+'/%s/%s/%s.%d.png' % (names[0], names[1][:-4], names[1][:-4], v2_id)

    v1 = Image.open(v1_names).convert('RGB')
    v1 = transform(v1)

    
    v2 = Image.open(v2_names).convert('RGB')
    v2 = transform(v2)

    return v1, v2