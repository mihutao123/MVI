import os
import numpy as np
#import math

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

from voc_4_2 import VOCDataset

from resnet_yolo2_3_test import resnet_new
#from torchvision import models
from torchsummary import summary

from loss4_20_2 import Loss
import pandas as pd

# VOC class names and BGR color.

class YOLODetector:
    def __init__(self,model_path,gpu_id=0):

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        use_gpu = torch.cuda.is_available()
        assert use_gpu, 'Current implementation does not support CPU mode. Enable CUDA.'

        # Load YOLO model.
        print("Loading YOLO model...")
        self.yolo = resnet_new()
        sd = torch.load(model_path)
        self.yolo.load_state_dict(sd)
        self.yolo.cuda()

        print("Done loading!")
        self.yolo.eval()


    def detect(self, image_bgr):

        with torch.no_grad():
            pred_tensor = self.yolo(image_bgr)
        return pred_tensor


    def compute_iou(self, bbox1, bbox2):
        """ Compute the IoU (Intersection over Union) of two set of bboxes, each bbox format: [x1, y1, x2, y2].
        Args:
            bbox1: (Tensor) bounding bboxes, sized [N, 4].
            bbox2: (Tensor) bounding bboxes, sized [M, 4].
        Returns:
            (Tensor) IoU, sized [N, M].
        """
        N = bbox1.size(0)
        M = bbox2.size(0)

        # Compute left-top coordinate of the intersections
        lt = torch.max(
            bbox1[:, :1].unsqueeze(1).expand(N, M, 1), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, :1].unsqueeze(0).expand(N, M, 1)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Conpute right-bottom coordinate of the intersections
        rb = torch.min(
            bbox1[:, 1:2].unsqueeze(1).expand(N, M, 1), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bbox2[:, 1:2].unsqueeze(0).expand(N, M, 1)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        # Compute area of the intersections from the coordinates
        wh = rb - lt   # width and height of the intersection, [N, M, 2]
        wh[wh < 0] = 0 # clip at 0
        inter = wh[:, :, 0] # [N, M]

        # Compute area of the bboxes
        area1 = (bbox1[:, 1] - bbox1[:, 0])  # [N, ]
        area2 = (bbox2[:, 1] - bbox2[:, 0])  # [M, ]
        area1 = area1.unsqueeze(1).expand_as(inter) # [N, ] -> [N, 1] -> [N, M]
        area2 = area2.unsqueeze(0).expand_as(inter) # [M, ] -> [1, M] -> [N, M]

        # Compute IoU from the areas
        union = area1 + area2 - inter # [N, M, 2]
        iou = inter / union           # [N, M, 2]

        return iou


if __name__ == '__main__':

    
    test_dir1  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata1.npy' 
    test_dir2  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata2.npy' 
    test_dir3  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata3.npy' 
    test_dir4  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata4.npy' 
    test_dir5  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata5.npy' 
    test_dir6  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata6.npy' 
    test_dir7  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata7.npy' 
    test_dir8  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata8.npy' 
    test_dir9  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata9.npy' 
    test_dir10  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata10.npy' 
    test_dir11  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata11.npy' 
    test_dir12  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata12.npy' 
    test_dir13  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata13.npy'
    test_dir14  = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\tedata14.npy'  
    val_label   = 'E:\\yolo_movingload_10_new_1_6_1_1_mu12\\test_noise_mu12\\label1_1.txt'

    

    val_dataset = VOCDataset(val_label,test_dir1,test_dir2,test_dir3,test_dir4,test_dir5,test_dir6,test_dir7,test_dir8,test_dir9,test_dir10,test_dir11,test_dir12,test_dir13,test_dir14)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    # Path to the yolo weight.
    model_path = 'weights/model_48.pth'
    # GPU device on which yolo is loaded.
    gpu_id = 0

    # Load model.
    yolo = YOLODetector(model_path, gpu_id=gpu_id)

    # Detect objects.
    data_test = np.zeros((32*10,24))
    
    for i, (imgs, targets) in enumerate(val_loader):
        if i==1250:
            break
        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        target_tensor = Variable(targets)
        imgs, target_tensor = imgs.cuda(), target_tensor.cuda()

        # Forward to compute validation loss.
        with torch.no_grad():
            pred_tensor = yolo.detect(imgs)
            pred_tensor1 = pred_tensor.cpu().numpy()
            pred_tensor1 = pred_tensor1.reshape(32*10,24)
        data_test = np.append(data_test,pred_tensor1,axis=0)
        print(i)
        #print(data_test)
        #print(data_test.shape)
        #print(target_tensor.size())
    np.save('data_test.npy', data_test)

