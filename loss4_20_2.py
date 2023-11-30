import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Loss(nn.Module):
    
    
    def __init__(self, feature_size=10, num_bboxes=1, num_classes=21, lambda_coord=10.0, lambda_noobj=1.0, lambda_class=0.5):

        super(Loss, self).__init__()

        self.S = feature_size
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

    def MSELoss(self, pred,target):
        return (pred-target)**2
    
    def clip_by_tensor(self, t,t_min,t_max):
        t=t.float()
 
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    
    def BCELoss(self, pred,target):
        epsilon = 1e-6
        pred = self.clip_by_tensor(pred, epsilon, 1.0 - epsilon)
        output = -target * torch.log(pred) - (1.0 - target) * torch.log(1.0 - pred)
        return output    


    def compute_iou(self, bbox1, bbox2):

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

    def forward(self, pred_tensor, target_tensor):

        S, B, C = self.S, self.B, self.C
        N = 3 * B + C    # 5=len([x, y, w, h, conf]，N=30

        #批的大小
        batch_size = pred_tensor.size(0)
        #有目标的张量[n_batch, S, S]
        coord_mask1 = target_tensor[..., 2] > 0 

        coord_mask0 = coord_mask1
        
        #没有目标的张量[n_batch, S, S]
        noobj_mask1 = target_tensor[..., 2] == 0 

        noobj_mask0 = noobj_mask1
        #扩展维度的布尔值相同，[n_batch, S, S] -> [n_batch, S, S, N]
        coord_mask = coord_mask0.unsqueeze(-1).expand_as(pred_tensor)  
        coord_mask1 = coord_mask0.unsqueeze(-1).expand_as(target_tensor) 
        noobj_mask = noobj_mask0.unsqueeze(-1).expand_as(pred_tensor)  
        noobj_mask1 = noobj_mask0.unsqueeze(-1).expand_as(target_tensor) 

        #int8-->bool
        noobj_mask = noobj_mask.bool()  
        noobj_mask1 = noobj_mask1.bool()  
        coord_mask = coord_mask.bool()  
        coord_mask1 = coord_mask1.bool() 

        ##################################################
        #预测值里含有目标的张量取出来，[n_coord, N]
        coord_pred = pred_tensor[coord_mask].view(-1, N)        
        
        #提取bbox和C,[n_coord x B, 5=len([x, y, w, h, conf])]
        #bbox_pred = coord_pred[:, :3*B].contiguous().view(-1, 3)   
        bbox_pred = coord_pred[:, :3*B].contiguous().view(-1, 3)   
        bbox_pred[:,0:1] = F.sigmoid(bbox_pred[:,0:1])
        bbox_pred[:,2] = F.sigmoid(bbox_pred[:,2])
        
        # 预测值的分类信息[n_coord, C]
        class_pred = coord_pred[:, 3*B:] 
        class_pred = F.softmax(class_pred, dim=1)                       

        #含有目标的标签张量，[n_coord, N]
        coord_target = target_tensor[coord_mask1].view(-1, 3 * B+1)        
        
        #提取标签bbox和C,[n_coord x B, 5=len([x, y, w, h, conf])]
        bbox_target = coord_target[:, :3*B].contiguous().view(-1, 3) 
        #标签的分类信息
        class_target = coord_target[:, 3*B:]                         
        ######################################################

        # ##################################################
        #没有目标的处理
        #找到预测值里没有目标的网格张量[n_noobj, N]，n_noobj=SxS-n_coord
        noobj_pred = F.sigmoid(pred_tensor[noobj_mask]).view(-1, N)         
        #标签的没有目标的网格张量 [n_noobj, N]                                                     
        noobj_target = target_tensor[noobj_mask1].view(-1, 3 * B+1)            
        
        noobj_conf_mask0 = torch.cuda.BoolTensor(noobj_pred.size()).fill_(0) # [n_noobj, N]
        noobj_conf_mask1 = torch.cuda.BoolTensor(noobj_target.size()).fill_(0) # [n_noobj, N]
        for b in range(B):
            noobj_conf_mask0[:, 2 + b*3] = 1 # 没有目标置信度置1，noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
            noobj_conf_mask1[:, 2 + b*3] = 1 # 没有目标置信度置1，noobj_conf_mask[:, 4] = 1; noobj_conf_mask[:, 9] = 1
        

        noobj_pred_conf = noobj_pred[noobj_conf_mask0]       # [n_noobj x 2=len([conf1, conf2])]
        noobj_target_conf = noobj_target[noobj_conf_mask1]   # [n_noobj x 2=len([conf1, conf2])]
        #计算没有目标的置信度损失
        #loss_noobj = F.mse_loss(noobj_pred_conf, noobj_target_conf, reduction='sum')
        loss_noobj = torch.sum(self.MSELoss(noobj_pred_conf, noobj_target_conf))
        #################################################################################


        loss_obj = torch.sum(self.MSELoss(bbox_pred[:, 2], bbox_target[:, 2]))
        mass_mask = torch.cuda.FloatTensor([[1.0],[1.5],[2.0],[2.5],[3.0],[3.5],[4.0],[4.5],[5.0],[5.5],[6.0],[6.5],[7.0],[7.5],[8.0],[8.5],[9.0],[9.5],[10.0],[10.5],[11.0]])
        class_pred_temp = torch.mm(class_pred,mass_mask)
        coord_mask4 = class_target > 0
        #cor_class_pred = torch.mul(class_pred_temp, coord_mask4)
        
        loss_class = F.smooth_l1_loss(10*torch.div(class_pred_temp[coord_mask4]+1,class_target[coord_mask4]+1), 10*torch.div(class_target[coord_mask4],class_target[coord_mask4]),reduction='sum')
        
        
        #loss_class = torch.sum(self.BCELoss(class_pred, class_target))
        
        #################################################################################
        # Compute loss for the cells with objects.
        coord_response_mask = torch.cuda.BoolTensor(bbox_target.size()).fill_(0)    # [n_coord x B, 5]
        coord_response_mask0 = torch.cuda.BoolTensor(bbox_target.size(0),1).fill_(0)
        bbox_target_scale = torch.ones(bbox_target.size(0)//1, 1).cuda()  
        bbox_target_scale[: , 0] = bbox_target_scale[: , 0] * (6.5131/40)
        bbox_target_scale = bbox_target_scale.contiguous().view(-1, 1) 


        # Choose the predicted bbox having the highest IoU for each target bbox.
        for i in range(0, bbox_target.size(0), B):

            target = bbox_target[i:i+B].view(-1, 3) # target bbox at i-th cell, [1, 5=len([x, y, w, h, conf])]

            max_iou, max_index = target[: , 2].max(0)
            max_index = max_index.data.cuda()

            coord_response_mask[i+max_index] = 1
            coord_response_mask0[i+max_index] = 1



        bbox_pred_response = bbox_pred[coord_response_mask].view(-1, 3)      # [n_response, 5]
        bbox_target_response = bbox_target[coord_response_mask].view(-1, 3)  # [n_response, 5], only the first 4=(x, y, w, h) are used
        target_scale = bbox_target_scale[coord_response_mask0].view(-1, 1)
        
        loss_xy = torch.sum(self.MSELoss(bbox_pred_response[:, :1], bbox_target_response[:, :1]))
        loss_wh = torch.sum(self.MSELoss(bbox_pred_response[:, 1:2], torch.log(torch.div(bbox_target_response[:, 1:2],target_scale))))        


        # Total loss
        loss = self.lambda_coord * (loss_xy + loss_wh) + loss_obj + self.lambda_noobj * loss_noobj + self.lambda_class*loss_class
        loss = loss / float(batch_size)
        loss_xy1 = loss_xy / float(batch_size)
        loss_wh1 = loss_wh / float(batch_size)
        loss_obj1 = loss_obj / float(batch_size)
        loss_noobj1 = loss_noobj / float(batch_size)
        loss_class1 = loss_class / float(batch_size)
        
        return loss_xy1, loss_wh1, loss_obj1, loss_noobj1, loss_class1, loss
