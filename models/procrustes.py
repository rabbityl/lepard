import torch
import torch.nn as nn


def topk(data, num_topk):
    sort, idx = data.sort(descending=True)
    return sort[:num_topk], idx[:num_topk]


class SoftProcrustesLayer(nn.Module):
    def __init__(self, config):
        super(SoftProcrustesLayer, self).__init__()

        self.sample_rate = config.sample_rate
        self.max_condition_num= config.max_condition_num

    @staticmethod
    def batch_weighted_procrustes( X, Y, w, eps=0.0001):
        '''
        @param X: source frame [B, N,3]
        @param Y: target frame [B, N,3]
        @param w: weights [B, N,1]
        @param eps:
        @return:
        '''
        # https://ieeexplore.ieee.org/document/88573

        bsize = X.shape[0]
        device = X.device
        W1 = torch.abs(w).sum(dim=1, keepdim=True)
        w_norm = w / (W1 + eps)
        mean_X = (w_norm * X).sum(dim=1, keepdim=True)
        mean_Y = (w_norm * Y).sum(dim=1, keepdim=True)
        Sxy = torch.matmul( (Y - mean_Y).transpose(1,2), w_norm * (X - mean_X) )
        Sxy = Sxy.cpu().double()
        U, D, V = Sxy.svd() # small SVD runs faster on cpu
        condition = D.max(dim=1)[0] / D.min(dim=1)[0]
        S = torch.eye(3)[None].repeat(bsize,1,1).double()
        UV_det = U.det() * V.det()
        S[:, 2:3, 2:3] = UV_det.view(-1, 1,1)
        svT = torch.matmul( S, V.transpose(1,2) )
        R = torch.matmul( U, svT).float().to(device)
        t = mean_Y.transpose(1,2) - torch.matmul( R, mean_X.transpose(1,2) )
        return R, t, condition



    def forward(self,  conf_matrix,  src_pcd, tgt_pcd,  src_mask, tgt_mask):
        '''
        @param conf_matrix:
        @param src_pcd:
        @param tgt_pcd:
        @param src_mask:
        @param tgt_mask:
        @return:
        '''

        bsize, N, M = conf_matrix.shape

        # subsample correspondence
        src_len = src_mask.sum(dim=1)
        tgt_len = tgt_mask.sum(dim=1)
        entry_max, _ = torch.stack([src_len,tgt_len], dim=0).max(dim=0)
        entry_max = (entry_max * self.sample_rate).int()
        sample_n_points = entry_max.float().mean().int() #entry_max.max()
        conf, idx = conf_matrix.view(bsize, -1).sort(descending=True,dim=1)
        w = conf [:, :sample_n_points]
        idx= idx[:, :sample_n_points]
        idx_src = idx//M #torch.div(idx, M, rounding_mode='trunc')
        idx_tgt = idx%M
        b_index = torch.arange(bsize).view(-1, 1).repeat((1, sample_n_points)).view(-1)
        src_pcd_sampled = src_pcd[b_index, idx_src.view(-1)].view(bsize, sample_n_points, -1)
        tgt_pcd_sampled = tgt_pcd[b_index, idx_tgt.view(-1)].view(bsize, sample_n_points, -1)
        w_mask = torch.arange(sample_n_points).view(1,-1).repeat(bsize,1).to(w)
        w_mask = w_mask < entry_max[:,None]
        w[~w_mask] = 0.

        # solve
        try :
            R, t, condition = self.batch_weighted_procrustes(src_pcd_sampled, tgt_pcd_sampled, w[...,None])
        except: # fail to get valid solution, this usually happens at the early stage of training
            R = torch.eye(3)[None].repeat(bsize,1,1).type_as(conf_matrix)
            t = torch.zeros(3, 1)[None].repeat(bsize,1,1).type_as(conf_matrix)
            condition = torch.zeros(bsize).type_as(conf_matrix)

        #filter unreliable solution with condition nnumber
        solution_mask = condition < self.max_condition_num
        R_forwd = R.clone()
        t_forwd = t.clone()
        R_forwd[~solution_mask] = torch.eye(3).type_as(R)
        t_forwd[~solution_mask] = torch.zeros(3, 1).type_as(R)

        return R, t, R_forwd, t_forwd, condition, solution_mask