import os
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import torch
import torch.nn as nn

from torch.optim import Adam
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models import resnet50
from torchvision import transforms
from torchvision.io import read_image
from src.Model_Setting import *  # ResNet_classifier, Densenet121

with open('/data/S/LinGroup/Users/sam/VRDL_HW1/testing_img_order.txt', 'r') as f:
    content = f.readlines()

with open('/data/S/LinGroup/Users/sam/VRDL_HW1/classes.txt', 'r') as f:
    name = f.readlines()
# print(content)
# print(name)

class BirdDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super(BirdDataset, self).__init__()
        self.df = content
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i, training=True):
        data = {'image': os.path.join('/data/S/LinGroup/Users/sam/VRDL_HW1/testing_images', self.df[i].strip()),
                'id': self.df[i].strip()}
        
        if self.transform:
            data = self.transform(data)
            
        return data
    

class read_imaged(): #
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = read_image(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')
                
        return data
    

class Scale01d():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = data[key].float()/255
            else:
                raise KeyError(f'{key} is not a key of {data}')
                
        return data      

    
class Resized(): # Resize to --> (128,128)
    def __init__(self, keys, size=(128,128)):
        self.keys = keys
        self.size = size
    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.Resize(self.size)(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')
                
        return data
    
    
class Normalized():
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        for key in self.keys:
            if key in data:
                data[key] = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(data[key])
            else:
                raise KeyError(f'{key} is not a key of {data}')
                
        return data   


def compute_accuracy(pred, label):
    return torch.sum(torch.argmax(pred, dim=1) == label)


test_transform = Compose([
    read_imaged(keys=['image']),
    Scale01d(keys=['image']),
    Resized(keys=['image'], size=(500, 500)),
    Normalized(keys=['image'])
])

test_set = BirdDataset(content, test_transform)
test_loader = DataLoader(test_set, batch_size=50, num_workers=4, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Net(path):
    if Model=='resnet50':
        model = ResNet_classifier()
    elif Model=='densenet121':
        model = Densenet121()
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    return model

# Net('/.hdf5').eval(),

# Model = 'resnet50' # densenet121, resnet50
# model_list = [
#         Net('record_1023_1911_MBS24/ep=029-acc=0.7383.hdf5').eval(),  # val  0.7733
#         Net('record_1023_1911_MBS24/ep=031-acc=0.7467.hdf5').eval(),  
        
#         Net('record_1024_1810/ep=094-acc=0.7467.hdf5').eval(),
#         Net('record_1024_1810/ep=057-acc=0.7483.hdf5').eval(),
#         Net('record_1024_1810/ep=075-acc=0.7467.hdf5').eval(),  
#         Net('record_1024_1810/ep=067-acc=0.7567.hdf5').eval(), 
    
#         Net('record_1024_1849/ep=067-acc=0.7583.hdf5').eval(),  
#         Net('record_1024_1849/ep=050-acc=0.7567.hdf5').eval(), 
#         Net('record_1024_1849/ep=051-acc=0.7600.hdf5').eval(),
#         ]


# Model = 'resnet50' # densenet121, resnet50
# # 前 7 val_acc = .7633
# model_list = [                                                              # val_acc = .7583
#               Net('record_1023_1911_MBS24/ep=031-acc=0.7467.hdf5').eval(),  # val_acc = .7400
#               Net('record_1023_1911_MBS24/ep=032-acc=0.7317.hdf5').eval(),  
#               Net('record_1023_1911_MBS24/ep=023-acc=0.7317.hdf5').eval(),  
#               Net('record_1023_1911_MBS24/ep=029-acc=0.7383.hdf5').eval(), 
              
#               Net('record_1026_0237/ep=072-acc=0.7417.hdf5').eval(), # val_acc = .7367
#               Net('record_1026_0237/ep=095-acc=0.7400.hdf5').eval(),  
#               Net('record_1026_0237/ep=089-acc=0.7400.hdf5').eval(),  
#             Net('record_1024_1810/ep=094-acc=0.7467.hdf5').eval(),          # 中 4 val_acc = .7517
#             Net('record_1024_1810/ep=057-acc=0.7483.hdf5').eval(),          # 前 3 val_acc = .7483
#             Net('record_1024_1810/ep=075-acc=0.7467.hdf5').eval(),  
#             Net('record_1024_1810/ep=067-acc=0.7567.hdf5').eval(), 
    
#             Net('record_1024_1849/ep=067-acc=0.7583.hdf5').eval(),  
#             Net('record_1024_1849/ep=050-acc=0.7567.hdf5').eval(), 
#             Net('record_1024_1849/ep=051-acc=0.7600.hdf5').eval()  
       
#         ]
# Net('record_1026_1951/.hdf5').eval()

Model = 'resnet50' 
model_list = [Net('record_1026_1951/ep=099-acc=0.7617.hdf5').eval(), # 全      .7683
              Net('record_1026_1951/ep=100-acc=0.7533.hdf5').eval(),
#               Net('record_1026_1951/.hdf5').eval(),
#               Net('record_1026_1951/.hdf5').eval(),
#               Net('record_1026_1951/.hdf5').eval()
        ]

with open('/data/S/LinGroup/Users/YC/DL_HW1/answer.txt', 'w') as fp:
#if __name__ == '__main__':
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            print(f'Here : {step}')
            image = data['image'].to(device)
            preds = torch.sum(torch.stack([m(image) for m in model_list]), dim=0).argmax(dim=1).detach().tolist()
            
            
            #preds = pred.argmax(dim=1).detach().tolist()
            for ID, pred in zip(data['id'], preds):
#                 print(pred+1, ID, name[pred].strip())
                fp.write(f"{ID} {name[pred].strip()}\n")
