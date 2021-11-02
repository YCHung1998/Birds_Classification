from torch.utils.data import Dataset
import pandas as pd
import os


class VRDL_BirdDataset(Dataset):
    def __init__(self, data_root, fold_list, data_frame, transform=None):
        super(VRDL_BirdDataset, self).__init__()
        self.data_root = data_root
        self.fold_list = fold_list
        self.data_frame = data_frame
        self.fold_id_list = [(fold, image_id, label)
                             for fold in self.fold_list
                             for image_id, label in zip(self.data_frame['image_id'], self.data_frame['label'])]
        self.transform = transform
        
    def __len__(self):
        return len(self.fold_id_list)        
        
    def __getitem__(self, i):
        fold, image_id, label = self.fold_id_list[i]
        data = {'image_id' : image_id,
                'image' : os.path.join(self.data_root, fold, image_id),
                'label' : int(label)-1 
               }
        if self.transform:
            data = self.transform(data)

        return data


class Kth_Folder():
    def __init__(self, k=1, txt_path=r'/data/S/LinGroup/Users/YC/VRDL_HW1/fold'):
        self.k = k
        self.txt_path = txt_path
        self.fold_list = sorted(os.listdir(self.txt_path))
    
    def valid_folds(self):
        return [ self.fold_list[self.k-1] ]
    
    def train_folds(self):
        train_folds = self.fold_list.copy()
        train_folds.remove(self.fold_list[self.k-1])
        return  train_folds
    
    def get_frame(self):
        train_frame = []
        for fold in self.train_folds():
            loc = os.path.join(self.txt_path, fold)
            train_frame = train_frame + [line.strip().split(' ') for line in open(loc,'r')]
        valid_frame = []
        for fold in self.valid_folds():
            loc = os.path.join(self.txt_path, fold)
            valid_frame = valid_frame + [line.strip().split(' ') for line in open(loc,'r')]
        return {'train' : train_frame, 'valid' : valid_frame}
    
    def Get_id_label(self, frame):
        '''[ ['1961.jpg', '115.Brewer_Sparrow'],
        ['0652.jpg', '115.Brewer_Sparrow'], ... ]
        '''
        IMG_ID = lambda x : x[0]
        IMG_LB = lambda x : int(x[1][:3])
        image_id = list(map(IMG_ID, frame))
        label = list(map(IMG_LB, frame))
        return {'image_id' : image_id, 'label' :label}
    
    def Get_train_valid_frame(self):
        train_frame = self.Get_id_label(self.get_frame()['train'])
        valid_frame = self.Get_id_label(self.get_frame()['valid'])
        return train_frame, valid_frame
