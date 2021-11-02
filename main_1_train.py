from src.Model_Setting import ResNet_classifier, Densenet121
from src.Prepare_data import Prepare_data
from src.Dataset import Kth_Folder

import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import os
import time


kth_valid              = 3      # 1, 2, 3, 4, 5
Max_epoch              = 20     # 16, 20, 24
MBS                    = 20     # 20, 30, 50, 100
save_threshold         = 0.7 
lr                     = 1e-4
lr_weight_decay        = 1e-4
lr_scheduler_step      = 4
lr_scheduler_gamma     = 0.8
fold_list              = {'train' : ['training_images', 
                                     'training_images_HoriFlip'],
                          'valid' : ['training_images']}
data_root              = r'/data/S/LinGroup/Users/YC/VRDL_HW1'
save_root              = r'/data/S/LinGroup/Users/YC/DL_HW1/record_' + time.strftime("%m%d_%H%M",time.localtime())
output_record          = 'A_Record.txt'
max_val_ep_acc         = [0,-1]



def compute_accuracy(pred, label):
    return torch.sum(torch.argmax(pred, dim=1) == label)

def train_step(batch_data): # from train_loader
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

if os.path.exists(save_root)==False:
    os.mkdir(save_root)
kth_folder = Kth_Folder(k=kth_valid)
train_frame, valid_frame = kth_folder.Get_train_valid_frame()
train_set, valid_set, train_loader, valid_loader = Prepare_data(MBS, fold_list, train_frame, valid_frame)

## ================== Model Select ================== 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet_classifier() # ResNet_classifier, Densenet121
model = nn.DataParallel(model).to(device)
criterion = nn.CrossEntropyLoss()

## --------------- Ooptimizer --------------- 

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=lr_weight_decay)

## --------------- Learning Schedule --------------- 

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step, gamma=lr_scheduler_gamma)


for ep in range(1, Max_epoch+1):
    tic = time.time()
    train_loss_list, train_acc_list = [], []
    for step, batch_data in enumerate(train_loader):
        record = train_step(batch_data)
        train_loss_list.append(record['loss'].item())
        train_acc_list.append(record['acc'].item())

    train_loss, train_acc = np.mean(train_loss_list), np.sum(train_acc_list)/(2400*len(fold_list['train']))
    del train_loss_list, train_acc_list

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

    print(f"epoch: {ep}/{Max_epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {valid_loss:.4f}, val_acc: {valid_acc:.4f}")
    print(f"total time: {int(time.time() - tic)}, lr: {lr_scheduler.get_last_lr()[0]:.6f}")
    
    if valid_acc>=save_threshold:
        torch.save(model.module.state_dict(), save_root + f'/ep={ep:03}-acc={valid_acc:.4f}.hdf5')
    if ep%5==0:print(f"Current best, Epoch : {max_val_ep_acc[0]:03}, Acc : {max_val_ep_acc[1]:.4f}")
        
    with open(os.path.join(save_root, output_record), 'a') as f:
        f.write(f"epoch: {ep:3d}/{Max_epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {valid_loss:.4f}, val_acc: {valid_acc:.4f}\n")
