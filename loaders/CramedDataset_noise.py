import copy
import csv
import os
import pickle
import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import random
##########CramedDataset config############
dataset = "CREMAD"

class CramedDataset(Dataset):

    def __init__(self, dataset_dir, mode='train',noise = 0, noise_per = 0.0, audio_noise = 10.0, audio_noise_per = 1.0):
        self.image = []
        self.audio = []
        self.label = []
        self.mode = mode
        # print("dataset_dir:",dataset_dir)
        # /data/huacong/CREMA/data
        self.data_root = dataset_dir # dataset_dir
        
        self.noise_level = noise
        self.noise_percentage = noise_per
        print("Noise_Level:",self.noise_level)
        print("Noise_Percentage:",self.noise_percentage)

        self.audio_noise_level = audio_noise
        self.audio_noise_percentage = audio_noise_per
        print("Audio_Noise_Level:",self.audio_noise_level)
        print("Audio_Noise_Per:",self.audio_noise_percentage)
        
        
        class_dict = {'NEU':0, 'HAP':1, 'SAD':2, 'FEA':3, 'DIS':4, 'ANG':5}

        self.visual_feature_path = '/data/huacong/CREMA/data/'
        self.audio_feature_path = '/data/huacong/CREMA/data/AudioWAV'
        self.fps = 1

        self.train_csv = os.path.join(self.data_root, 'train.csv')
        self.test_csv = os.path.join(self.data_root, 'test.csv')

        if mode == 'train':
            csv_file = self.train_csv
        else:
            csv_file = self.test_csv

        with open(csv_file, encoding='UTF-8-sig') as f2:
            csv_reader = csv.reader(f2)
            for item in csv_reader:
                audio_path = os.path.join(self.audio_feature_path, item[0] + '.wav')
                visual_path = os.path.join(self.visual_feature_path, 'Image-{:02d}-FPS'.format(self.fps), item[0])
                if os.path.exists(audio_path) and os.path.exists(visual_path):
                    self.image.append(visual_path)
                    self.audio.append(audio_path)
                    self.label.append(class_dict[item[1]])
                else:
                    continue
        if self.mode == 'train':
            num_images_with_noise = int(len(self.image)*self.noise_percentage)
            self.images_to_add_noise = random.sample(self.image, num_images_with_noise)
            
            num_audios_with_noise = int(len(self.audio)*self.audio_noise_percentage)
            self.audios_to_add_noise = random.sample(self.audio, num_audios_with_noise)


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):

        # audio
        samples, rate = librosa.load(self.audio[idx], sr=22050)
        #print("samples:",samples.shape)
        if self.mode == 'train' and (self.audio[idx] in self.audios_to_add_noise):
            noise = np.random.normal(0, self.audio_noise_level, samples.shape)
            noisy_samples = samples + noise
            resamples = np.tile(noisy_samples, 3)[:22050*3]
            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.
        else:
            resamples = np.tile(samples, 3)[:22050*3]
            resamples[resamples > 1.] = 1.
            resamples[resamples < -1.] = -1.

        spectrogram = librosa.stft(resamples, n_fft=512, hop_length=353)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        # Visual
        image_samples = os.listdir(self.image[idx])
        # print("img:",idx)
        select_index = np.random.choice(len(image_samples), size= self.fps, replace=False)
        select_index.sort()
        images = torch.zeros((self.fps, 3, 224, 224))
        for i in range(self.fps):
            img = Image.open(os.path.join(self.image[idx], image_samples[i])).convert('RGB')
            img = transform(img)
            if self.mode == 'train':
                if self.image[idx] in self.images_to_add_noise:
                    noise = torch.randn_like(img) * self.noise_level
                    img_with_noise = torch.clamp(img+noise,0,1)
                    # pil_image = to_pil_image(img_with_noise)
                    # plt.imshow(pil_image)
                    # plt.axis('off')
                    # plt.show()
                    images[i] = img_with_noise
                # noisy images
                else:
                    images[i] = img
            else:
                images[i] = img        

        images = torch.permute(images, (1,0,2,3))
        # label
        label = self.label[idx]
        #print("images.shape:",images.shape)
        return spectrogram, images, label