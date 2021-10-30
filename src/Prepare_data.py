import pandas as pd
from torch.utils.data import Dataset, DataLoader
from src.Transform import Transforms
from src.Dataset import VRDL_BirdDataset

T = Transforms()
train_transforms, valid_transforms = T.train_transforms(), T.valid_transforms()
data_root = r'/data/S/LinGroup/Users/sam/VRDL_HW1'
def Prepare_data(MBS, fold_list, train_frame, valid_frame):
    train_set = VRDL_BirdDataset(data_root, fold_list['train'], train_frame, train_transforms)
    valid_set = VRDL_BirdDataset(data_root, fold_list['valid'], valid_frame, valid_transforms)
    train_loader = DataLoader(train_set, batch_size=MBS, num_workers=4, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=MBS, num_workers=4, shuffle=False)
    return train_set, valid_set, train_loader, valid_loader
