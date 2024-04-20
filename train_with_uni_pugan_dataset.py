import argparse
import os

parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
parser.add_argument("--model", type=str, default='punet')
parser.add_argument('--log_dir', default='logs/test', help='Log dir [default: logs/test_log]')
parser.add_argument('--npoint', type=int, default=256,help='Point Number [1024/2048] [default: 1024]')
parser.add_argument('--up_ratio',  type=int,  default=4, help='Upsampling Ratio [default: 4]')
parser.add_argument('--max_epoch', type=int, default=100, help='Epochs to run [default: 100]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training')
parser.add_argument("--use_bn", action='store_true', default=False)
parser.add_argument("--use_res", action='store_true', default=False)
parser.add_argument("--alpha", type=float, default=1.0) # for repulsion loss
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--use_decay', action='store_true', default=False)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.71)
parser.add_argument('--lr_clip', type=float, default=0.000001)
parser.add_argument('--decay_step_list', type=list, default=[30, 60])
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--workers', type=int, default=4)

args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pointnet2 import pointnet2_utils as pn2_utils
from utils.utils import knn_point
from chamfer_distance import chamfer_distance
from auction_match import auction_match

from dataset import PUGAN_Dataset
import numpy as np
import importlib

from knn_cuda import KNN

import math

class UpsampleLoss(nn.Module):
    def __init__(self, alpha=1.0, nn_size=5, radius=0.07, h=0.03, eps=1e-12):
        super().__init__()
        self.alpha = alpha
        self.nn_size = nn_size
        self.radius = radius
        self.h = h
        self.eps = eps
        
        self.knn_uniform=KNN(k=2,transpose_mode=True)
        self.knn_repulsion=KNN(k=20,transpose_mode=True)
        
        self.double()

    def get_emd_loss(self, pred, gt, pcd_radius):
        idx, _ = auction_match(pred, gt)
        matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
        matched_out = matched_out.transpose(1, 2).contiguous()
        dist2 = (pred - matched_out) ** 2
        dist2 = dist2.view(dist2.shape[0], -1) # <-- ???
        dist2 = torch.mean(dist2, dim=1, keepdims=True) # B,
        dist2 /= pcd_radius
        return torch.mean(dist2)

    def get_cd_loss(self, pred, gt, pcd_radius):
        cost_for, cost_bac = chamfer_distance(gt, pred)
        cost = 0.8 * cost_for + 0.2 * cost_bac
        cost /= pcd_radius
        cost = torch.mean(cost)
        return cost

    '''
    def get_repulsion_loss(self, pred):
        _, idx = knn_point(self.nn_size, pred, pred, transpose_mode=True)
        idx = idx[:, :, 1:].to(torch.int32) # remove first one
        idx = idx.contiguous() # B, N, nn

        pred = pred.transpose(1, 2).contiguous() # B, 3, N
        grouped_points = pn2_utils.grouping_operation(pred, idx) # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

        grouped_points = grouped_points - pred.unsqueeze(-1)
        dist2 = torch.sum(grouped_points ** 2, dim=1)
        dist2 = torch.max(dist2, torch.tensor(self.eps).cuda())
        dist = torch.sqrt(dist2)
        weight = torch.exp(- dist2 / self.h ** 2)

        uniform_loss = torch.mean((self.radius - dist) * weight)
        # uniform_loss = torch.mean(self.radius - dist * weight) # punet
        
        return uniform_loss
    '''
    
    def get_uniform_loss(self,pcd,percentage=[0.004,0.006,0.008,0.010,0.012],radius=1.0):
        B,N,C=pcd.shape[0],pcd.shape[1],pcd.shape[2]
        npoint=int(N*0.05)
        loss=0
        further_point_idx = pn2_utils.furthest_point_sample(pcd.contiguous(), npoint)
        new_xyz = pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N
        for p in percentage:
            nsample=int(N*p)
            r=math.sqrt(p*radius)
            disk_area=math.pi*(radius**2)/N

            idx=pn2_utils.ball_query(r,nsample,pcd.contiguous(),new_xyz.permute(0,2,1).contiguous()) #b N nsample

            expect_len=math.sqrt(disk_area)

            grouped_pcd=pn2_utils.grouping_operation(pcd.permute(0,2,1).contiguous(),idx)#B C N nsample
            grouped_pcd=grouped_pcd.permute(0,2,3,1) #B N nsample C

            grouped_pcd=torch.cat(torch.unbind(grouped_pcd,dim=1),dim=0)#B*N nsample C

            dist,_=self.knn_uniform(grouped_pcd,grouped_pcd)
            #print(dist.shape)
            uniform_dist=dist[:,:,1:] #B*N nsample 1
            uniform_dist=torch.abs(uniform_dist+1e-8)
            uniform_dist=torch.mean(uniform_dist,dim=1)
            uniform_dist=(uniform_dist-expect_len)**2/(expect_len+1e-8)
            mean_loss=torch.mean(uniform_dist)
            mean_loss=mean_loss*math.pow(p*100,2)
            loss+=mean_loss
        return loss/len(percentage)
    
    def get_repulsion_loss(self,pcd,h=0.0005):
        dist,idx=self.knn_repulsion(pcd,pcd)#B N k

        dist=dist[:,:,1:5]**2 #top 4 cloest neighbors

        loss=torch.clamp(-dist+h,min=0)
        loss=torch.mean(loss)
        #print(loss)
        return loss
    
    def forward(self, pred, gt, pcd_radius):
        return self.get_emd_loss(pred, gt, pcd_radius) * 100, \
            self.get_uniform_loss(pred) * 10, \
            self.get_repulsion_loss(pred) * 5, \

def get_optimizer():
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr, 
                                momentum=0.98, 
                                weight_decay=args.weight_decay, 
                                nesterov=True)
    else:
        raise NotImplementedError
    
    if args.use_decay:
        def lr_lbmd(cur_epoch):
            cur_decay = 1
            for decay_step in args.decay_step_list:
                if cur_epoch >= decay_step:
                    cur_decay = cur_decay * args.lr_decay
            return max(cur_decay, args.lr_clip / args.lr)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lbmd)
        return optimizer, lr_scheduler
    else:
        return optimizer, None


if __name__ == '__main__':
    train_dst = PUGAN_Dataset(npoint=args.npoint)
    train_loader = DataLoader(train_dst, batch_size=args.batch_size, 
                        shuffle=True, pin_memory=True, num_workers=args.workers)

    MODEL = importlib.import_module('models.' + args.model)
    model = MODEL.get_model(npoint=args.npoint, up_ratio=args.up_ratio, 
                use_normal=False, use_bn=args.use_bn, use_res=args.use_res)
    model.cuda()
    
    optimizer, lr_scheduler = get_optimizer()
    loss_func = UpsampleLoss(alpha=args.alpha)

    model.train()
    for epoch in range(args.max_epoch):
        loss_list = []
        emd_loss_list = []
        rep_loss_list = []
        for batch in train_loader:
            optimizer.zero_grad()
            input_data, gt_data, radius_data = batch

            input_data = input_data.float().cuda()
            gt_data = gt_data.float().cuda()
            gt_data = gt_data[..., :3].contiguous()
            radius_data = radius_data.float().cuda()

            preds = model(input_data)
            emd_loss, uni_loss, rep_loss = loss_func(preds, gt_data, radius_data)
            loss = emd_loss + rep_loss + uni_loss
        
            loss.backward()         
            optimizer.step()

            loss_list.append(loss.item())
            emd_loss_list.append(emd_loss.item())
            rep_loss_list.append(rep_loss.item())
        print(' -- epoch {}, loss {:.4f}, weighted emd loss {:.4f}, repulsion loss {:.4f}, lr {}.'.format(
            epoch, np.mean(loss_list), np.mean(emd_loss_list), np.mean(rep_loss_list), \
            optimizer.state_dict()['param_groups'][0]['lr']))
        
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)
        if (epoch + 1) % 20 == 0:
            state = {'epoch': epoch, 'model_state': model.state_dict()}
            save_path = os.path.join(args.log_dir, 'punet_epoch_{}.pth'.format(epoch))
            torch.save(state, save_path)