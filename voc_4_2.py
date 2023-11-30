import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import math
torch.manual_seed(1) 

class VOCDataset(Dataset):
    
    def __init__(self, list_file, data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14):
        self.data1 = np.load(data1) #加载npy数据
        self.data2 = np.load(data2) #加载npy数据
        self.data3 = np.load(data3) #加载npy数据
        self.data4 = np.load(data4) #加载npy数据
        self.data5 = np.load(data5) #加载npy数据
        self.data6 = np.load(data6) #加载npy数据
        self.data7 = np.load(data7) #加载npy数据
        self.data8 = np.load(data8) #加载npy数据
        self.data9 = np.load(data9) #加载npy数据
        self.data10 = np.load(data10) #加载npy数据
        self.data11 = np.load(data11) #加载npy数据
        self.data12 = np.load(data12) #加载npy数据
        self.data13 = np.load(data13) #加载npy数据
        self.data14 = np.load(data14) #加载npy数据
        
        self.boxes = []
        self.labels = []
        
        
        if isinstance(list_file, list):
            # Cat multiple list files together.
            # This is especially useful for voc07/voc12 combination.
            tmp_file = '/tmp/listfile.txt'
            os.system('cat %s > %s' % (' '.join(list_file), tmp_file))
            list_file = tmp_file
        
        with open(list_file) as f:
            lines  = f.readlines()
        
        for line in lines:
            splited = line.strip().split()
            num_boxes = (len(splited)) // 3
            box=[]
            label=[]
            for i in range(num_boxes):
                x = float(splited[3*i])
                x2 = float(splited[1+3*i])
                c = float(splited[2+3*i])
                box.append([x,x2])
                label.append([c])
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.Tensor(label))
        self.num_samples = len(self.boxes)        
                
        
        #self.transforms = transform #转为tensor形式
    def __getitem__(self, idx):
        hdct1 = self.data1[idx, :]  # 读取每一个npy的数据
        hdct2 = self.data2[idx, :]  # 读取每一个npy的数据
        hdct3 = self.data3[idx, :]  # 读取每一个npy的数据
        hdct4 = self.data4[idx, :]  # 读取每一个npy的数据
        hdct5 = self.data5[idx, :]  # 读取每一个npy的数据
        hdct6 = self.data6[idx, :] # 读取每一个npy的数据
        hdct7 = self.data7[idx, :]  # 读取每一个npy的数据
        hdct8 = self.data8[idx, :]  # 读取每一个npy的数据
        hdct9 = self.data9[idx, :]  # 读取每一个npy的数据
        hdct10 = self.data10[idx, :]  # 读取每一个npy的数据
        hdct11 = self.data11[idx, :]  # 读取每一个npy的数据
        hdct12 = self.data12[idx, :]  # 读取每一个npy的数据
        hdct13 = self.data13[idx, :]  # 读取每一个npy的数据
        hdct14 = self.data14[idx, :] # 读取每一个npy的数据
        hdct = np.stack([hdct1,hdct2,hdct3,hdct4,hdct5,hdct6,hdct7,hdct8,hdct9,hdct10,hdct11,hdct12,hdct13,hdct14], axis=0)
        hdct = torch.tensor(hdct)
        #hdct= self.transforms(hdct)  #转为tensor形式
        
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        boxes /= torch.Tensor([40, 40]).expand_as(boxes)
        target = self.encoder(boxes,labels)
            
        return hdct,target #返回数据还有标签
    
    def __len__(self):
        return self.num_samples #返回数据的总个数

    def encoder(self,boxes,labels):
        '''
        boxes (tensor) [[x1,x2],[]]
        labels (tensor) [...]
        return 1x8x(2x2+2+6)
        '''
        grid_num = 10
        target = torch.zeros((grid_num,4))
        cell_size = 1./grid_num
        wh = boxes[:,1]-boxes[:,0]
        cxcy = (boxes[:,1]+boxes[:,0])/2
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample/cell_size).ceil()-1 #
            xy = ij*cell_size #匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample -xy)/cell_size
            target[int(ij),2] = 1
            target[int(ij),0] = delta_xy
            target[int(ij),1] = wh[i]          
            target[int(ij),3] = labels[i,:]
        return target
    
def test():

    data1  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata1.npy' 
    data2  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata2.npy' 
    data3  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata3.npy' 
    data4  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata4.npy' 
    data5  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata5.npy' 
    data6  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata6.npy' 
    data7  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata7.npy' 
    data8  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata8.npy' 
    data9  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata9.npy' 
    data10  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata10.npy' 
    data11  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata11.npy' 
    data12  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata12.npy' 
    data13  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata13.npy'
    data14  = 'E:\\yolo_movingload_10_new_1\\test_noise\\tedata14.npy'  


    
    label_txt   = 'E:\\yolo_movingload_10_new_1\\test_noise\\label1_1.txt'

    dataset = VOCDataset(label_txt, data1,data2,data3,data4,data5,data6,data7,data8,data9,data10,data11,data12,data13,data14)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    data_iter = iter(data_loader)
    for i in range(500):
        img, target = next(data_iter)
        print(i, img.size(), target.size())
        print(target)
        #print(img[0,0,:])
        #print(target)

if __name__ == '__main__':
    test()
