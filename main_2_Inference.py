import os
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import Compose, ToTensor, Resize

from src.Model_Setting import *  # ResNet_classifier, Densenet121
from src.Transform import read_imaged, Scale01d, Resized, Normalized, Transforms

Model = 'resnet50' # resnet50, densenet121
folder = 'ckpt'

class BirdDataset(Dataset):
    def __init__(self, data_list, transform=None):
        super(BirdDataset, self).__init__()
        self.df = content
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i, training=True):
        data = {'image': os.path.join('/data/S/LinGroup/Users/sam/VRDL_HW1/testing_images', self.df[i].strip()),
                'image_id': self.df[i].strip()}
        
        if self.transform:
            data = self.transform(data)
            
        return data
    

def Net(path):
    if Model=='resnet50':
        model = ResNet_classifier()
    elif Model=='densenet121':
        model = Densenet121()
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    return model


with open('/data/S/LinGroup/Users/sam/VRDL_HW1/testing_img_order.txt', 'r') as f:
    content = f.readlines()

with open('/data/S/LinGroup/Users/sam/VRDL_HW1/classes.txt', 'r') as f:
    name = f.readlines()

T =  Transforms()
test_transform = T.test_transforms()
test_set       = BirdDataset(content, test_transform)
test_loader    = DataLoader(test_set, batch_size=100, num_workers=4, shuffle=False)
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model_name_list = os.listdir(folder)
model_list = []
for model_name in model_name_list:
    if '.txt' in model_name:
        continue
    if '.hdf5' not in model_name or int(model_name[-9:-5])<7300:
        continue
    else:
        model_list.append(Net(os.path.join(folder, model_name)).eval())
        
print('Waiting for a minutes ...')
with open('/data/S/LinGroup/Users/YC/DL_HW1/answer.txt', 'w') as fp:
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            print(step)
            image = data['image'].to(device)
            preds = torch.sum(torch.stack([model(image) for model in model_list]), dim=0).argmax(dim=1).detach().tolist()
            for ID, pred in zip(data['image_id'], preds):
                fp.write(f"{ID} {name[pred].strip()}\n")
print('Finished create an answer.')
