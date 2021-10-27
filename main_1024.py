## 1022
from src.Model_Setting import * # ResNet_classifier
from src.Prepare_data import Prepare_data
# from src.Transform import *

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import os
import time

#data_root = r'/data/S/LinGroup/Users/sam/VRDL_HW1'
#frame_root = r'/data/S/LinGroup/Users/YC/DL/HW1/Data.csv'
#data_frame = pd.read_csv(frame_root)
#train_frame = data_frame[data_frame.Type=='training'] 
#valid_frame = data_frame[data_frame.Type=='validation']

Max_epoch = 100
MBS = 24
save_threshold = 0.65 # lager than .65 will be save
fold_list = {'train' : ['training_images', 
                         'training_images_aug1', 
                         'training_images_aug2', 
                         'training_images_aug3'
                         ],
             'valid' : ['training_images']}

# train_fold_list = fold_list
# valid_fold_list = list(fold_list[0])

train_set, valid_set, train_loader, valid_loader = Prepare_data(MBS, fold_list)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## ================== Model Seletct ================== 
model = ResNet_classifier() # ResNet_classifier, Densenet121
model = nn.DataParallel(model).to(device)

criterion = nn.CrossEntropyLoss()

def compute_accuracy(pred, label):
    return torch.sum(torch.argmax(pred, dim=1) == label)

def train_step(batch_data):
    model.train()
    optimizer.zero_grad()
    image = batch_data['image'].to(device)
    label = batch_data['label'].to(device)
    pred = model(image)
    loss = criterion(pred, label)
    acc  = compute_accuracy(pred, label)

    loss.backward()
    optimizer.step()
    
    return {'loss': loss, 'acc': acc}

def valid_step(batch_data): # from valid_loader
    model.eval()
    image = batch_data['image'].to(device)
    label = batch_data['label'].to(device)
    pred = model(image)
    loss = criterion(pred, label)
    acc  = compute_accuracy(pred, label)
    
    return {'loss': loss, 'acc': acc}


import time

LR = [8e-4, 6e-4, 5e-4, 3e-4, 1e-4]
lr_r_tuning = []
# tmp_root = '/data/S/LinGroup/Users/YC/DL_HW1/record_' + time.strftime("%m%d_%H%M",time.localtime())
# if os.path.exists(tmp_root)==False:
#     os.mkdir(tmp_root)

for lr_r in [LR[4]]: # LR , simple [LR[idx]]
    
    tmp_root = r'/data/S/LinGroup/Users/YC/DL_HW1/record_' + time.strftime("%m%d_%H%M",time.localtime())
    if os.path.exists(tmp_root)==False:
        os.mkdir(tmp_root)
    max_val_ep_acc = [0,-1]
    
    ## --------------- Ooptimizer --------------- 
    
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum =0.9) # 還沒修過課!!
    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_r, weight_decay=1e-4)
    
    ## --------------- Learning Schedule --------------- 
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.8) # step = 4 (default=3)

    
    for ep in range(1, Max_epoch+1):
        tic = time.time()
        model.train() # 轉模式，train step 裡面也有寫，重要所以先放外面
        train_loss_list, train_acc_list = [], []
        for step, batch_data in enumerate(train_loader):
            record = train_step(batch_data)
            train_loss_list.append(record['loss'].item())
            train_acc_list.append(record['acc'].item())
        train_loss, train_acc = np.mean(train_loss_list), np.sum(train_acc_list)/(2400*len(fold_list['train']))
        del train_loss_list, train_acc_list
        

        # Validation step
        model.eval() # 轉模式，valid step 裡面也有寫，重要所以先放外面
        valid_loss_list, valid_acc_list = [], []
        with torch.no_grad():
            for step, batch_data in enumerate(valid_loader):
                record = valid_step(batch_data)
                valid_loss_list.append(record['loss'].item())
                valid_acc_list.append(record['acc'].item())
        valid_loss, valid_acc = np.mean(valid_loss_list), np.sum(valid_acc_list)/600
        del valid_loss_list, valid_acc_list
        
        if max_val_ep_acc[1]<valid_acc:
            max_val_ep_acc = [ep, valid_acc]
        lr_scheduler.step()
        
        if valid_acc>=save_threshold:
            torch.save(model.module.state_dict(), tmp_root + f'/ep={ep:03}-acc={valid_acc:.4f}.hdf5')
#         torch.save(model.state_dict(), tmp_root + f'/ep={ep:03}-acc={valid_acc:.4f}.hdf5') # 用單張卡時使用，要把前面nn.paralle關掉
        print(f"epoch: {ep}/{Max_epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {valid_loss:.4f}, val_acc: {valid_acc:.4f}")
        print(f"total time: {int(time.time() - tic)}, lr: {lr_scheduler.get_last_lr()[0]:.6f}")
        
        if ep%5==0:print(f"Current best in lr : {lr_r}, Epoch : {max_val_ep_acc[0]:03}, Acc : {max_val_ep_acc[1]:.4f}")
        output_record = 'A_Record.txt'
        with open(os.path.join(tmp_root, output_record), 'a') as f:
            f.write(f"epoch: {ep}/{Max_epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {valid_loss:.4f}, val_acc: {valid_acc:.4f}\n")

    lr_r_tuning.append([lr_r, *max_val_ep_acc])  
print(*lr_r_tuning)
