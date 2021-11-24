import math
import torch
from torch import nn

class VolumetricPositionEncoding(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.feature_dim = config.feature_dim
        self.vol_bnds = config.vol_bnds
        self.voxel_size = config.voxel_size
        self.vol_origin  = self.vol_bnds[0]
        self.pe_type = config.pe_type

    def voxelize(self, xyz):
        '''
        @param xyz: B,N,3
        @return: B,N,3
        '''
        if type ( self.vol_origin ) == list :
            self.vol_origin = torch.FloatTensor(self.vol_origin ).view(1, 1, -1).to( xyz.device )
        return (xyz - self.vol_origin) / self.voxel_size

    @staticmethod
    def embed_rotary(x, cos, sin):
        '''
        @param x: [B,N,d]
        @param cos: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @param sin: [B,N,d]  [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        @return:
        '''
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape_as(x).contiguous()
        x = x * cos + x2 * sin
        return x

    @staticmethod
    def embed_pos(pe_type, x, pe):
        """ combine feature and position code
        """
        if  pe_type == 'rotary':
            return VolumetricPositionEncoding.embed_rotary(x, pe[..., 0], pe[..., 1])
        elif  pe_type == 'sinusoidal':
            return  x + pe
        else:
            raise KeyError()


    def forward(self,  XYZ):
        '''
        @param XYZ: [B,N,3]
        @return:
        '''
        bsize, npoint, _ = XYZ.shape

        vox = self.voxelize( XYZ)
        x_position, y_position, z_position = vox[..., 0:1], vox[...,1:2], vox[...,2:3]
        div_term = torch.exp( torch.arange(0, self.feature_dim // 3, 2, dtype=torch.float, device=XYZ.device) *  (-math.log(10000.0) / (self.feature_dim // 3)))
        div_term = div_term.view( 1,1, -1) # [1, 1, d//6]

        sinx = torch.sin(x_position * div_term) # [B, N, d//6]
        cosx = torch.cos(x_position * div_term)
        siny = torch.sin(y_position * div_term)
        cosy = torch.cos(y_position * div_term)
        sinz = torch.sin(z_position * div_term)
        cosz = torch.cos(z_position * div_term)

        if self.pe_type == 'sinusoidal' :
            position_code = torch.cat( [ sinx, cosx, siny, cosy, sinz, cosz] , dim=-1 )

        elif self.pe_type == "rotary" :
            # sin/cos [θ0,θ1,θ2......θd/6-1] -> sin/cos [θ0,θ0,θ1,θ1,θ2,θ2......θd/6-1,θd/6-1]
            sinx, cosx, siny, cosy, sinz, cosz = map( lambda  feat:torch.stack([feat, feat], dim=-1).view(bsize, npoint, -1),
                 [ sinx, cosx, siny, cosy, sinz, cosz] )
            sin_pos = torch.cat([sinx,siny,sinz], dim=-1)
            cos_pos = torch.cat([cosx,cosy,cosz], dim=-1)
            position_code = torch.stack( [cos_pos, sin_pos] , dim=-1)

        else:
            raise KeyError()


        if position_code.requires_grad:
            position_code = position_code.detach()


        return position_code