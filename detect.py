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
        """ Detect objects from given image.
        Args:
            image_bgr: (numpy array) input image in BGR ids_sorted, sized [h, w, 3].
            image_size: (int) image width and height to which input image is resized.
        Returns:
            boxes_detected: (list of tuple) box corner list like [((x1, y1), (x2, y2))_obj1, ...]. Re-scaled for original input image size.
            class_names_detected: (list of str) list of class name for each detected boxe.
            probs_detected: (list of float) list of probability(=confidence x class_score) for each detected box.
        """
        with torch.no_grad():
            pred_tensor = self.yolo(image_bgr)
        return pred_tensor

    def decoder(self, pred_tensor):
        grid_num = 10
        cell_size = 1/grid_num
        boxes = []
        cls_indexs = []
        scores = []
        cell_size = 1.0 / grid_num
        pred = pred_tensor.data
        pred = pred.squeeze(0)  # 7x7x30
        contain1 = pred[:, 2].unsqueeze(1)
        contain = contain1
        mask1 = contain > 0.4  # 大于阈值   
        coo_response_mask = torch.cuda.BoolTensor(mask1.size()).fill_(0)
        for i in range(0,coo_response_mask.size()[0]):
            score = contain[i:i+1]
            max_score, max_index = score.max(1)
            max_index = max_index.data.cuda()            
            coo_response_mask[i, max_index]=1
        final_mask =  mask1.mul(coo_response_mask)
        mass_mask = torch.cuda.FloatTensor([[1.0],[1.5],[2.0],[2.5],[3.0],[3.5],[4.0],[4.5],[5.0],[5.5],[6.0],[6.5],[7.0],[7.5],[8.0],[8.5],[9.0],[9.5],[10.0],[10.5],[11.0]])
        for i in range(grid_num):
            for b in range(1):
                if final_mask[i, b]:
                    box = pred[i, b*3:b*3+2]
                    xy = (torch.cuda.FloatTensor([i]) * cell_size) 
                    cxcy = (box[0]* cell_size+xy)*40
                    lxly = box[1]*40
                    box_xy = torch.cuda.FloatTensor(box.size())
                    box_xy[0] = cxcy - 0.5*lxly
                    box_xy[1] = cxcy + 0.5*lxly
                    cls_mass = pred[i, 3:]
                    score0 = contain[i, b]
                    mass = torch.mm(cls_mass.unsqueeze(dim=0),mass_mask)
                    boxes.append(box_xy)
                    cls_indexs.append(mass)
                    scores.append(score0)
       
        return boxes, cls_indexs, scores
        
    def decoder1(self, boxes0, cls_indexs0, scores0):
        iou0 = np.zeros((boxes0.shape[0],boxes0.shape[0]))
        mask_box = np.ones((boxes0.shape))
        mask_box = np.array(mask_box, dtype = bool)
        mask_cs = np.ones(cls_indexs0.shape)
        mask_cs = np.array(mask_cs, dtype = bool)
        for i in range(boxes0.shape[0]):
            for j in range(i+1, boxes0.shape[0]):
                wh = boxes0[i,1] - boxes0[j,0]
                if wh< 0:
                    wh = 0
                union = boxes0[j,1] - boxes0[i,0]              
                iou0[i,j] = wh/union
        mask0 = iou0 > 0.15
        mask1 = np.zeros((boxes0.shape[0],boxes0.shape[0]))
        for i in range(boxes0.shape[0]):
            for j in range(i+1, boxes0.shape[0]):
                if abs(boxes0[j, 0]-boxes0[i, 0])<3 and abs(boxes0[j, 1]-boxes0[i, 1])<3:
                    mask1[i,j] = 1
        mask2 =  np.multiply(mask0,mask1)
        for i in range(boxes0.shape[0]):
            for j in range(i+1, boxes0.shape[0]):
                if mask2[i, j]:
                    if scores0[i]>scores0[j]:
                       mask_box[j,:] = False
                       mask_cs[j] = False
                    else:
                       mask_box[i,:] = False 
                       mask_cs[i] = False
        boxes1 =  boxes0[mask_box]  
        boxes1 = boxes1.reshape((len(boxes1)//2,2))
        cls_indexs1 = cls_indexs0[mask_cs]  
        scores1 = scores0[mask_cs]  
        return boxes1, cls_indexs1, scores1



if __name__ == '__main__':

    
    test_dir1  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata1.npy' 
    test_dir2  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata2.npy' 
    test_dir3  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata3.npy' 
    test_dir4  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata4.npy' 
    test_dir5  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata5.npy' 
    test_dir6  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata6.npy' 
    test_dir7  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata7.npy' 
    test_dir8  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata8.npy' 
    test_dir9  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata9.npy' 
    test_dir10  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata10.npy' 
    test_dir11  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata11.npy' 
    test_dir12  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata12.npy' 
    test_dir13  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata13.npy'
    test_dir14  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata14.npy'  
    val_label   = 'E:\\yolo_movingload_10_new_1\\test_noise\\label1_1.txt'

    

    """
    test_dir1  = 'train/trdata1.npy' 
    test_dir2  = 'train/trdata2.npy' 
    test_dir3  = 'train/trdata3.npy' 
    test_dir4  = 'train/trdata4.npy' 
    test_dir5  = 'train/trdata5.npy' 
    test_dir6  = 'train/trdata6.npy' 
    test_dir7  = 'train/trdata7.npy' 
    test_dir8  = 'train/trdata8.npy' 
    test_dir9  = 'train/trdata9.npy' 
    test_dir10  = 'train/trdata10.npy' 
    test_dir11  = 'train/trdata11.npy' 
    test_dir12  = 'train/trdata12.npy' 
    test_dir13  = 'train/trdata13.npy'
    test_dir14  = 'train/trdata14.npy'  
    val_label   = 'train/trainlabel.txt'
    """

    val_dataset = VOCDataset(val_label,test_dir1,test_dir2,test_dir3,test_dir4,test_dir5,test_dir6,test_dir7,test_dir8,test_dir9,test_dir10,test_dir11,test_dir12,test_dir13,test_dir14)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    # Path to the yolo weight.
    #model_path = 'weights_lr01/model_91.pth'
    model_path = 'weights/model_46.pth'
    # GPU device on which yolo is loaded.
    gpu_id = 0

    # Load model.
    yolo = YOLODetector(model_path, gpu_id=gpu_id)

    # Detect objects.
    file = open('output.txt','a')
    
    for i, (imgs, targets) in enumerate(val_loader):
        #if i==300:
            #break
        # Load data as a batch.
        batch_size_this_iter = imgs.size(0)
        imgs = Variable(imgs)
        target_tensor = Variable(targets)
        imgs, target_tensor = imgs.cuda(), target_tensor.cuda()

        # Forward to compute validation loss.
        with torch.no_grad():
            pred_tensor = yolo.detect(imgs)
            boxes0, cls_indexs0, scores0 = yolo.decoder(pred_tensor)
        #print(i)
        #print(boxes)
        #print(cls_indexs)
        #print(scores)
        #print(pred_tensor)  
        #print(target_tensor)

        cls_indexs1 = torch.tensor(cls_indexs0).cuda().data.cpu().numpy()
        boxes1 = torch.tensor([item.cpu().detach().numpy() for item in boxes0])
        boxes1 = torch.tensor(boxes1).cuda().data.cpu().numpy()
        scores1 = torch.tensor(scores0).cuda().data.cpu().numpy()
        
        if len(cls_indexs0) == 0:
            boxes2 = boxes1
            cls_indexs2 = cls_indexs1
            scores2 = scores1
        if len(cls_indexs0) == 1:
            boxes2 = boxes1
            cls_indexs2 = cls_indexs1
            scores2 = scores1
        if len(cls_indexs0) > 1:
            boxes2, cls_indexs2, scores2 = yolo.decoder1(boxes1, cls_indexs1, scores1)
        print(i)
        #print(boxes2)
        #print(cls_indexs2)
        #print(scores2)
        if len(boxes2) ==0:
            file.write('\n') 
        else:
            for j in range(len(boxes2)):
                s = str(boxes2[j,0]).replace('[','').replace(']','') 
                s = s.replace("'",'').replace(',','')
                file.write(s)
                file.write('\t') 
                s = str(boxes2[j,1]).replace('[','').replace(']','') 
                s = s.replace("'",'').replace(',','')
                file.write(s)
                file.write('\t') 
                s = str(cls_indexs2[j]).replace('[','').replace(']','') 
                s = s.replace("'",'').replace(',','')
                file.write(s)
                file.write('\t') 
            file.write('\n')            
