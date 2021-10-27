import os
import pandas as pd


Data = pd.read_csv(r'/data/S/LinGroup/Users/YC/DL/HW1/Data.csv')

valid_id_path = r'/data/S/LinGroup/Users/YC/DL/HW1/valid_id.csv'

with open(valid_id_path) as f:
    valid_name = f.readlines()
STRIP = lambda x : x.strip('\n')
valid_name = list(map(STRIP,valid_name))
valid_name = valid_name[1:]
Type = 'validation'

if Type in ['training','validation']:
    image_root = r'/data/S/LinGroup/Users/sam/VRDL_HW1/training_images'
    ACC = 0
    tag = True
elif Type== 'testing':
    image_root = r'/data/S/LinGroup/Users/sam/VRDL_HW1/testing_images'
    tag = False


from torch.utils.data import Dataset, DataLoader
class EvaluateDataset(Dataset):
    def __init__(self, data_list, Type='testing', transform=None):
        super(EvaluateDataset, self).__init__()
        self.data_list = data_list # valid_name, test_name
        self.transform = transform
        self.Type = Type
        if self.Type in ['training','validation']:
            image_root = r'/data/S/LinGroup/Users/sam/VRDL_HW1/training_images'
        elif self.Type== 'testing':
            image_root = r'/data/S/LinGroup/Users/sam/VRDL_HW1/testing_images'
        self.image_root = image_root

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, i):
        if self.Type=='testing':
            data = {'image': os.path.join(self.image_root, self.data_list[i]),
                    'id': self.data_list[i]}
        elif self.Type=='validation':
            data = {'image': os.path.join(self.image_root, self.data_list[i]),
                    'id': self.data_list[i],
                    'label': int(Data.iloc[int(self.data_list[i].split('.')[0])-1].label-1) # 答案 1給 0;答案 10 給 9
               }
        if self.transform:
            data = self.transform(data)
        return data




import torch
def Accuracy(pred, label):
    return torch.sum(pred == label)

# 直接 call 外面的
from src.Transform import Transforms
T = Transforms() # call the class i had created
if Type=='training':
    transform = T.train_transforms()
elif Type=='validation':
    transform = T.valid_transforms()
elif Type=='testing':
    transform = T.test_transforms()


dataset = EvaluateDataset(data_list=valid_name,Type=Type ,transform=transform)
dataloader = DataLoader(dataset, batch_size=50, num_workers=4, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from src.Model_Setting import * 
# ResNet_classifier , Densenet121

def Net(path):
    if Model=='resnet50':
        model = ResNet_classifier()
    elif Model=='densenet121':
        model = Densenet121()
    ckpt = torch.load(path)
    model.load_state_dict(ckpt)
    model = model.to(device)

    return model


# model_list = [
#         Net('record_1023_1911_MBS24/ep=029-acc=0.7383.hdf5').eval(),  # 0.7567 後八
#         Net('record_1023_1911_MBS24/ep=031-acc=0.7467.hdf5').eval(),  
#         Net('record_1023_1911_MBS24/ep=023-acc=0.7317.hdf5').eval(),
#         Net('record_1023_1911_MBS24/ep=022-acc=0.7200.hdf5').eval(),  # 0.7400 後五
#         Net('record_1023_1911_MBS24/ep=016-acc=0.7267.hdf5').eval(), 
    
#         Net('record_1023_1525/ep=030-acc=0.7067.hdf5').eval(),  # 0.7167 後三
#         Net('record_1023_1916_MBS16/ep=033-acc=0.7083.hdf5').eval(), 
#         Net('record_1023_1916_MBS16/ep=035-acc=0.7100.hdf5').eval()  
#         ]


# Model = 'resnet50' # densenet121, resnet50
# model_list = [                                                            # val_acc = .7733
#         Net('record_1023_1911_MBS24/ep=029-acc=0.7383.hdf5').eval(),   
#         Net('record_1023_1911_MBS24/ep=031-acc=0.7467.hdf5').eval(),    # 前 2 val_acc = .7400
        
#         Net('record_1024_1810/ep=094-acc=0.7467.hdf5').eval(),          # 中 4 val_acc = .7517
#         Net('record_1024_1810/ep=057-acc=0.7483.hdf5').eval(),          # 前 3 val_acc = .7483
#         Net('record_1024_1810/ep=075-acc=0.7467.hdf5').eval(),  
#         Net('record_1024_1810/ep=067-acc=0.7567.hdf5').eval(), 
    
#         Net('record_1024_1849/ep=067-acc=0.7583.hdf5').eval(),        # val_acc = .7550
#         Net('record_1024_1849/ep=050-acc=0.7567.hdf5').eval(), 
#         Net('record_1024_1849/ep=051-acc=0.7600.hdf5').eval()  
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

#acc 85 89 63 88 81 92 82 99 56 65
#loss 65 80 81 82 47 85 63 84 97 56
# Model = 'resnet50' 
# model_list = [Net('record_1026_1227/ep=085-acc=0.7683.hdf5').eval(), # 全      .7683
#               Net('record_1026_1227/ep=089-acc=0.7667.hdf5').eval(), # 前 2     .7750
#               Net('record_1026_1227/ep=063-acc=0.7650.hdf5').eval(), # 前 3    .7633
#               Net('record_1026_1227/ep=088-acc=0.7650.hdf5').eval(), # 1,2,4 :  .7783
#               Net('record_1026_1227/ep=081-acc=0.7633.hdf5').eval(), # 1,2,4,5 : .7733

#         ]

# model_list = [
#             Net('record_1026_1227/ep=065-acc=0.7600.hdf5').eval(),
#             Net('record_1026_1227/ep=080-acc=0.7583.hdf5').eval(),
#             Net('record_1026_1227/ep=081-acc=0.7633.hdf5').eval(),
#             Net('record_1026_1227/ep=082-acc=0.7617.hdf5').eval(),
#             Net('record_1026_1227/ep=047-acc=0.7450.hdf5').eval(),

#         ]
# Net('record_1026_1227/.hdf5').eval(),
# Net('record_1026_1227/.hdf5').eval(),
# Net('record_1026_1227/.hdf5').eval(),
# Net('record_1026_1227/.hdf5').eval(),
# Net('record_1026_1227/.hdf5').eval(),

Model = 'resnet50' 
model_list = [Net('record_1026_1951/ep=099-acc=0.7617.hdf5').eval(), # 全      .7683
              Net('record_1026_1951/ep=100-acc=0.7533.hdf5').eval(),
#               Net('record_1026_1951/.hdf5').eval(),
#               Net('record_1026_1951/.hdf5').eval(),
#               Net('record_1026_1951/.hdf5').eval()
        ]

with open('/data/S/LinGroup/Users/YC/DL_HW1/answer_test.txt', 'w') as fp:
    with torch.no_grad():
        print('Wait for a second ...')
        for step, data in enumerate(dataloader):
            image = data['image'].to(device)
            preds = torch.sum(torch.stack([m(image) for m in model_list]), dim=0).argmax(dim=1).detach()
            if tag :
                label = data['label'].to(device)
#                 print(preds.shape, label.shape)
                acc = Accuracy(preds, label)
                ACC+=acc
            #preds = pred.argmax(dim=1).detach().tolist()
            else: 
                print(f'writing {step} ')
                for ID, pred in zip(data['id'], preds.tolist()):
                    fp.write(f"{ID} {name[pred].strip()}\n")
        print(f'Acc : {ACC/len(dataset):.4f}')
