import os
import numpy as np
#import math

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from voc_4_2 import VOCDataset

from resnet_yolo2_3 import resnet_new
#from torchvision import models
from torchsummary import summary

from loss4_20_2 import Loss
import pandas as pd


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

#########################
# data
#########################

# Path to data dir.
train_dir1 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata1.npy'
train_dir2 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata2.npy'
train_dir3 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata3.npy'
train_dir4 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata4.npy'
train_dir5 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata5.npy'
train_dir6 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata6.npy'
train_dir7 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata7.npy'
train_dir8 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata8.npy'
train_dir9 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata9.npy'
train_dir10 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata10.npy'
train_dir11 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata11.npy'
train_dir12 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata12.npy'
train_dir13 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata13.npy'
train_dir14 = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trdata14.npy'

test_dir1  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata1.npy' 
test_dir2  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata2.npy' 
test_dir3  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata3.npy' 
test_dir4  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata4.npy' 
test_dir5  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata5.npy' 
test_dir6  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata6.npy' 
test_dir7  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata7.npy' 
test_dir8  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata8.npy' 
test_dir9  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata9.npy' 
test_dir10  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata10.npy' 
test_dir11  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata11.npy' 
test_dir12  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata12.npy' 
test_dir13  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata13.npy'
test_dir14  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\tedata14.npy'  


# Path to label files.
train_label = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\train_noise_m12\\trainlabel.txt'
val_label   = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_m12\\label1_1.txt'


# Training hyper parameters.
num_epochs = 50
batch_size = 32

use_gpu = True
if use_gpu:
    torch.cuda.empty_cache()


##################################
# model backbone loading - YOLO
##################################
yolo = resnet_new()

if use_gpu:
    yolo.cuda()
summary(yolo, input_size=(14, 640))
 
##################################
# dataloader
##################################

# Setup loss and optimizer.
criterion = Loss()
optimizer = torch.optim.Adam(yolo.parameters(), lr=0.1e-3, betas=(0.9,0.999), eps =1e-08, weight_decay=5e-5)

# Load Pascal-VOC dataset.
train_dataset = VOCDataset(train_label,train_dir1,train_dir2,train_dir3,train_dir4,train_dir5,train_dir6,train_dir7,train_dir8,train_dir9,train_dir10,train_dir11,train_dir12,train_dir13,train_dir14)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

val_dataset = VOCDataset(val_label,test_dir1,test_dir2,test_dir3,test_dir4,test_dir5,test_dir6,test_dir7,test_dir8,test_dir9,test_dir10,test_dir11,test_dir12,test_dir13,test_dir14)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print('Number of training images: ', len(train_dataset))

##################################
# where to save model
##################################

log_dir = 'weights'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

##################################
# start training
##################################
results_dict = {'Epoch':[], 'train_loss':[], 'val_loss':[], 'val_loss_xy':[], 'val_loss_wh':[], 'val_loss_obj':[], 'val_loss_noobj':[], 'val_loss_class':[]}

best_val_loss = np.inf

for epoch in range(num_epochs):
    print('\n')
    print('Starting epoch {} / {}'.format(epoch, num_epochs))

    # Training.
    yolo.train()
    #total_loss_xy = 0.0
    #total_loss_wh = 0.0
    #total_loss_obj = 0.0
    #total_loss_noobj = 0.0
    #total_loss_class = 0.0
    total_loss = 0.0
    total_batch = 0

    for i, (imgs, targets) in enumerate(train_loader):
        # Update learning rate.

        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        if use_gpu:
            imgs, targets = imgs.cuda(), targets.cuda()

        # Forward to compute loss.
        preds = yolo(imgs)
        loss_xy, loss_wh, loss_obj, loss_noobj, loss_class, loss = criterion(preds, targets)
        loss_xy_this_iter = loss_xy.item()
        loss_wh_this_iter = loss_wh.item()
        loss_obj_this_iter = loss_obj.item()
        loss_noobj_this_iter = loss_noobj.item()
        loss_class_this_iter = loss_class.item()    
        loss_this_iter = loss.item()
        #total_loss_xy += loss_xy_this_iter * batch_size_this_iter
        #total_loss_wh += loss_wh_this_iter * batch_size_this_iter
        #total_loss_obj += loss_obj_this_iter * batch_size_this_iter
        #total_loss_noobj += loss_noobj_this_iter * batch_size_this_iter
        #total_loss_class += loss_class_this_iter * batch_size_this_iter
        total_loss += loss_this_iter * batch_size_this_iter
        
        total_batch += batch_size_this_iter

        # Backward to update model weight.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print current loss.
        #if i%200 == 0:
            #print('Epoch [%d/%d], Iter [%d/%d],loss_xy:%.4f,loss_wh:%.4f,loss_obj:%.4f,loss_noobj:%.4f,loss_class:%.4f, Loss: %.4f, Average Loss: %.4f'
                  #% (epoch, num_epochs, i, len(train_loader), loss_xy_this_iter, loss_wh_this_iter,loss_obj_this_iter,loss_noobj_this_iter,loss_class_this_iter, loss_this_iter, total_loss / float(total_batch)))

        if i%200 == 0:
            print('Epoch [%d/%d], Iter [%d/%d],loss_xy:%.4f,loss_wh:%.4f,loss_obj:%.4f,loss_noobj:%.4f,loss_class:%.4f, Loss: %.4f'
                  % (epoch, num_epochs, i, len(train_loader), loss_xy_this_iter, loss_wh_this_iter,loss_obj_this_iter,loss_noobj_this_iter,loss_class_this_iter, loss_this_iter))

        # TensorBoard.
        n_iter = epoch * len(train_loader) + i

    # Validation.
    yolo.eval()
    val_loss_xy = 0.0
    val_loss_wh = 0.0
    val_loss_obj = 0.0
    val_loss_noobj = 0.0
    val_loss_class = 0.0    
    val_loss = 0.0
    total_batch1 = 0

    for i, (imgs, targets) in enumerate(val_loader):
        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        targets = Variable(targets)
        if use_gpu:
            imgs, targets = imgs.cuda(), targets.cuda()

        # Forward to compute validation loss.
        with torch.no_grad():
            preds = yolo(imgs)
        loss_xy, loss_wh, loss_obj, loss_noobj, loss_class, loss = criterion(preds, targets)
        loss_xy_this_iter = loss_xy.item()
        loss_wh_this_iter = loss_wh.item()
        loss_obj_this_iter = loss_obj.item()
        loss_noobj_this_iter = loss_noobj.item()
        loss_class_this_iter = loss_class.item()           
        loss_this_iter = loss.item()
        val_loss_xy += loss_xy_this_iter * batch_size_this_iter
        val_loss_wh += loss_wh_this_iter * batch_size_this_iter
        val_loss_obj += loss_obj_this_iter * batch_size_this_iter
        val_loss_noobj += loss_noobj_this_iter * batch_size_this_iter
        val_loss_class += loss_class_this_iter * batch_size_this_iter
        val_loss += loss_this_iter * batch_size_this_iter
        total_batch1 += batch_size_this_iter
    val_loss_xy /= float(total_batch1)
    val_loss_wh /= float(total_batch1)
    val_loss_obj /= float(total_batch1)
    val_loss_noobj /= float(total_batch1)
    val_loss_class /= float(total_batch1)
    val_loss /= float(total_batch1)
        
    # update dict
    results_dict['Epoch'].append(epoch)
    results_dict['train_loss'].append(total_loss / float(total_batch))
    results_dict['val_loss'].append(val_loss)
    results_dict['val_loss_xy'].append(val_loss_xy)
    results_dict['val_loss_wh'].append(val_loss_wh)
    results_dict['val_loss_obj'].append(val_loss_obj)
    results_dict['val_loss_noobj'].append(val_loss_noobj)
    results_dict['val_loss_class'].append(val_loss_class)  
    
    
    df = pd.DataFrame(results_dict)
    df.to_csv(os.path.join(log_dir,'history.csv'))
    
    # Save results.
    if best_val_loss > val_loss:
        best_val_loss = val_loss
        #torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_best.pth'))
    torch.save(yolo.state_dict(), os.path.join(log_dir, 'model_' + str(epoch) + '.pth'))

    # Print.
    print('Epoch [%d/%d], val_loss_xy: %.4f, val_loss_wh: %.4f, val_loss_obj: %.4f, val_loss_noobj: %.4f, val_loss_class: %.4f, Val Loss: %.4f, Best Val Loss: %.4f'
    % (epoch + 1, num_epochs, val_loss_xy, val_loss_wh, val_loss_obj, val_loss_noobj, val_loss_class, val_loss, best_val_loss))
