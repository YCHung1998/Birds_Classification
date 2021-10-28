import pandas as pd
from torch.utils.data import Dataset, DataLoader

from src.Transform import Transforms
from src.Dataset import VRDL_BirdDataset


T = Transforms()
train_transforms, valid_transforms =  T.train_transforms(), T.valid_transforms()

def Prepare_data(MBS, fold_list):
    train_set = VRDL_BirdDataset(data_root, fold_list['train'], train_frame, train_transforms)
    valid_set = VRDL_BirdDataset(data_root, fold_list['valid'], valid_frame, valid_transforms)
    train_loader = DataLoader(train_set, batch_size=MBS, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=MBS, num_workers=4, shuffle=False)

    return train_set, valid_set, train_loader, valid_loader

# data_root = r'/data/S/LinGroup/Users/sam/VRDL_HW1/training_images'
# data_root = r'/data/S/LinGroup/Users/sam/VRDL_HW1'
# frame_root = r'/data/S/LinGroup/Users/YC/DL/HW1/Data.csv'
# data_frame = pd.read_csv(frame_root)
# train_frame = data_frame[data_frame.Type=='training'] 
# valid_frame = data_frame[data_frame.Type=='validation']

# fold_list = {'train' : ['training_images', 
#                          'training_images_aug1', 
#                          'training_images_aug2', 
#                          'training_images_aug3'
#                          ],
#              'valid' : ['training_images']}
