import copy
import math
import torch
from torch import nn
from torch.nn import Module, Dropout
from models.position_encoding import VolumetricPositionEncoding as VolPE
from models.matching import Matching
from models.procrustes import SoftProcrustesLayer
import numpy as np
import random
from scipy.spatial.transform import Rotation

class GeometryAttentionLayer(nn.Module):

    def __init__(self, config):

        super(GeometryAttentionLayer, self).__init__()

        d_model = config['feature_dim']
        nhead =  config['n_head']

        self.dim = d_model // nhead
        self.nhead = nhead
        self.pe_type = config['pe_type']
        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # self.attention = Attention() #LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_pe, source_pe, x_mask=None, source_mask=None):

        bs = x.size(0)
        q, k, v = x, source, source
        qp, kvp  = x_pe, source_pe
        q_mask, kv_mask = x_mask, source_mask

        if self.pe_type == 'sinusoidal':
            #w(x+p), attention is all you need : https://arxiv.org/abs/1706.03762
            if qp is not None: # disentangeld
                q = q + qp
                k = k + kvp
            qw = self.q_proj(q).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
            kw = self.k_proj(k).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
            vw = self.v_proj(v).view(bs, -1, self.nhead, self.dim)

        elif self.pe_type == 'rotary':
            #Rwx roformer : https://arxiv.org/abs/2104.09864

            qw = self.q_proj(q)
            kw = self.k_proj(k)
            vw = self.v_proj(v)

            if qp is not None: # disentangeld
                q_cos, q_sin = qp[...,0] ,qp[...,1]
                k_cos, k_sin = kvp[...,0],kvp[...,1]
                qw = VolPE.embed_rotary(qw, q_cos, q_sin)
                kw = VolPE.embed_rotary(kw, k_cos, k_sin)

            qw = qw.view(bs, -1, self.nhead, self.dim)
            kw = kw.view(bs, -1, self.nhead, self.dim)
            vw = vw.view(bs, -1, self.nhead, self.dim)

        else:
            raise KeyError()

        # attention
        a = torch.einsum("nlhd,nshd->nlsh", qw, kw)
        if kv_mask is not None:
            a.masked_fill_( q_mask[:, :, None, None] * (~kv_mask[:, None, :, None]), float('-inf'))
        a =  a / qw.size(3) **0.5
        a = torch.softmax(a, dim=2)
        o = torch.einsum("nlsh,nshd->nlhd", a, vw).contiguous()  # [N, L, (H, D)]

        message = self.merge(o.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        e = x + message

        return e






class RepositioningTransformer(nn.Module):

    def __init__(self, config):
        super(RepositioningTransformer, self).__init__()

        self.d_model = config['feature_dim']
        self.nhead = config['n_head']
        self.layer_types = config['layer_types']
        self.positioning_type = config['positioning_type']
        self.pe_type =config['pe_type']

        self.entangled= config['entangled']

        self.positional_encoding = VolPE(config)


        encoder_layer = GeometryAttentionLayer (config)

        self.layers = nn.ModuleList()

        for l_type in self.layer_types:

            if l_type in ['self','cross']:

                self.layers.append( copy.deepcopy(encoder_layer))

            elif l_type == "positioning":

                if self.positioning_type == 'procrustes':
                    positioning_layer = nn.ModuleList()
                    positioning_layer.append( Matching(config['feature_matching']))
                    positioning_layer.append( SoftProcrustesLayer(config['procrustes']) )
                    self.layers.append(positioning_layer)

                elif self.positioning_type in ['oracle', 'randSO3']:
                    self.layers.append( None)

                else :
                    raise KeyError(self.positioning_type + " undefined positional encoding type")


            else:
                raise KeyError()

        self._reset_parameters()



    def forward(self, src_feat, tgt_feat, s_pcd, t_pcd, src_mask, tgt_mask, data, T = None, timers = None):

        self.timers = timers

        assert self.d_model == src_feat.size(2), "the feature number of src and transformer must be equal"

        if T is not None:
            R, t = T
            src_pcd_wrapped = (torch.matmul(R, s_pcd.transpose(1, 2)) + t).transpose(1, 2)
            tgt_pcd_wrapped = t_pcd
        else:
            src_pcd_wrapped = s_pcd
            tgt_pcd_wrapped = t_pcd

        src_pe = self.positional_encoding( src_pcd_wrapped)
        tgt_pe = self.positional_encoding( tgt_pcd_wrapped)


        if not self.entangled:

            position_layer = 0
            data.update({"position_layers":{}})

            for layer, name in zip(self.layers, self.layer_types) :

                if name == 'self':
                    if self.timers: self.timers.tic('self atten')
                    src_feat = layer(src_feat, src_feat, src_pe, src_pe, src_mask, src_mask,)
                    tgt_feat = layer(tgt_feat, tgt_feat, tgt_pe, tgt_pe, tgt_mask, tgt_mask)
                    if self.timers: self.timers.toc('self atten')

                elif name == 'cross':
                    if self.timers: self.timers.tic('cross atten')
                    src_feat = layer(src_feat, tgt_feat, src_pe, tgt_pe, src_mask, tgt_mask)
                    tgt_feat = layer(tgt_feat, src_feat, tgt_pe, src_pe, tgt_mask, src_mask)
                    if self.timers: self.timers.toc('cross atten')

                elif name =='positioning':

                    if self.positioning_type == 'procrustes':

                        conf_matrix, match_pred = layer[0](src_feat, tgt_feat, src_pe, tgt_pe, src_mask, tgt_mask, data, pe_type=self.pe_type)

                        position_layer += 1
                        data["position_layers"][position_layer] = {"conf_matrix": conf_matrix, "match_pred": match_pred}

                        if self.timers: self.timers.tic('procrustes_layer')
                        R, t, R_forwd, t_forwd, condition, solution_mask = layer[1] (conf_matrix, s_pcd, t_pcd,  src_mask, tgt_mask)
                        if self.timers: self.timers.toc('procrustes_layer')

                        data["position_layers"][position_layer].update({
                            "R_s2t_pred": R,"t_s2t_pred": t, "solution_mask": solution_mask, "condition": condition})

                        src_pcd_wrapped = (torch.matmul(R_forwd, s_pcd.transpose(1, 2)) + t_forwd).transpose(1, 2)
                        tgt_pcd_wrapped = t_pcd
                        src_pe = self.positional_encoding(src_pcd_wrapped)
                        tgt_pe = self.positional_encoding(tgt_pcd_wrapped)


                    elif self.positioning_type == 'randSO3':
                        src_pcd_wrapped = self.rand_rot_pcd( s_pcd, src_mask)
                        tgt_pcd_wrapped = t_pcd
                        src_pe = self.positional_encoding(src_pcd_wrapped)
                        tgt_pe = self.positional_encoding(tgt_pcd_wrapped)


                    elif self.positioning_type == 'oracle':
                        #Note R,t ground truth is only available for computing oracle position encoding
                        rot_gt = data['batched_rot']
                        trn_gt = data['batched_trn']
                        src_pcd_wrapped = (torch.matmul(rot_gt, s_pcd.transpose(1, 2)) + trn_gt).transpose(1, 2)
                        tgt_pcd_wrapped = t_pcd
                        src_pe = self.positional_encoding(src_pcd_wrapped)
                        tgt_pe = self.positional_encoding(tgt_pcd_wrapped)


                    else:
                        raise KeyError(self.positioning_type + " undefined positional encoding type")

                else :
                    raise KeyError

            return src_feat, tgt_feat, src_pe, tgt_pe

        else : # pos. fea. entangeled

            position_layer = 0
            data.update({"position_layers":{}})

            src_feat = VolPE.embed_pos(self.pe_type, src_feat, src_pe)
            tgt_feat = VolPE.embed_pos(self.pe_type, tgt_feat, tgt_pe)

            for layer, name in zip(self.layers, self.layer_types):
                if name == 'self':
                    if self.timers: self.timers.tic('self atten')
                    src_feat = layer(src_feat, src_feat, None, None, src_mask, src_mask, )
                    tgt_feat = layer(tgt_feat, tgt_feat, None, None, tgt_mask, tgt_mask)
                    if self.timers: self.timers.toc('self atten')
                elif name == 'cross':
                    if self.timers: self.timers.tic('cross atten')
                    src_feat = layer(src_feat, tgt_feat, None, None, src_mask, tgt_mask)
                    tgt_feat = layer(tgt_feat, src_feat, None, None, tgt_mask, src_mask)
                    if self.timers: self.timers.toc('cross atten')
                elif name == 'positioning':
                    pass

            return src_feat, tgt_feat, src_pe, tgt_pe




    def rand_rot_pcd (self, pcd, mask):
        '''
        @param pcd: B, N, 3
        @param mask: B, N
        @return:
        '''

        pcd[~mask]=0.
        N = mask.shape[1]
        n_points = mask.sum(dim=1, keepdim=True).view(-1,1,1)
        bs = pcd.shape[0]

        euler_ab = np.random.rand(bs, 3) * np.pi * 2   # anglez, angley, anglex
        rand_rot =  torch.from_numpy( Rotation.from_euler('zyx', euler_ab).as_matrix() ).to(pcd)
        pcd_u = pcd.mean(dim=1, keepdim=True) * N / n_points
        pcd_centered = pcd - pcd_u
        pcd_rand_rot =  torch.matmul( rand_rot, pcd_centered.transpose(1,2) ).transpose(1,2) + pcd_u
        return pcd_rand_rot

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)