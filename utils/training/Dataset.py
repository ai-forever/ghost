from torch.utils.data import TensorDataset
import torchvision.transforms as transforms
from PIL import Image
import glob
import pickle
import random
import os
import cv2
import tqdm
import sys
sys.path.append('..')
# from utils.cap_aug import CAP_AUG
    

class FaceEmbed(TensorDataset):
    def __init__(self, data_path_list, same_prob=0.8):
        datasets = []
        # embeds = []
        self.N = []
        self.same_prob = same_prob
        for data_path in data_path_list:
            image_list = glob.glob(f'{data_path}/*.*g')
            datasets.append(image_list)
            self.N.append(len(image_list))
            # with open(f'{data_path}/embed.pkl', 'rb') as f:
            #     embed = pickle.load(f)
            #     embeds.append(embed)
        self.datasets = datasets
        # self.embeds = embeds
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
        idx = 0
        while item >= self.N[idx]:
            item -= self.N[idx]
            idx += 1
            
        image_path = self.datasets[idx][item]
        # name = os.path.split(image_path)[1]
        # embed = self.embeds[idx][name]
        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)

        if random.random() > self.same_prob:
            image_path = random.choice(self.datasets[random.randint(0, len(self.datasets)-1)])
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def __len__(self):
        return sum(self.N)


class FaceEmbedVGG2(TensorDataset):
    def __init__(self, data_path, same_prob=0.8, same_identity=False):

        self.same_prob = same_prob
        self.same_identity = same_identity
                
        self.images_list = glob.glob(f'{data_path}/*/*.*g')
        self.folders_list = glob.glob(f'{data_path}/*')
        
        self.folder2imgs = {}

        for folder in tqdm.tqdm(self.folders_list):
            folder_imgs = glob.glob(f'{folder}/*')
            self.folder2imgs[folder] = folder_imgs
             
        self.N = len(self.images_list)
        
        self.transforms_arcface = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        self.transforms_base = transforms.Compose([
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.01),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, item):
            
        image_path = self.images_list[item]

        Xs = cv2.imread(image_path)[:, :, ::-1]
        Xs = Image.fromarray(Xs)
        
        if self.same_identity:
            folder_name = '/'.join(image_path.split('/')[:-1])

        if random.random() > self.same_prob:
            image_path = random.choice(self.images_list)
            Xt = cv2.imread(image_path)[:, :, ::-1]
            Xt = Image.fromarray(Xt)
            same_person = 0
        else:
            if self.same_identity:
                image_path = random.choice(self.folder2imgs[folder_name])
                Xt = cv2.imread(image_path)[:, :, ::-1]
                Xt = Image.fromarray(Xt)
            else:
                Xt = Xs.copy()
            same_person = 1
            
        return self.transforms_arcface(Xs), self.transforms_base(Xs),  self.transforms_base(Xt), same_person

    def __len__(self):
        return self.N