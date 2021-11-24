from models.blocks import *
import torch.nn.functional as F
import numpy as np

class KPFCN(nn.Module):

    def __init__(self, config):
        super(KPFCN, self).__init__()

        ############
        # Parameters
        ############
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_feats_dim
        out_dim = config.first_feats_dim

        #####################
        # List Encoder blocks
        #####################
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # bottleneck output & input layer

        self.coarse_out = nn.Conv1d(in_dim//2, config.coarse_feature_dim,  kernel_size=1, bias=True)
        coarse_in_dim = config.coarse_feature_dim
        self.coarse_in = nn.Conv1d(coarse_in_dim, in_dim//2,  kernel_size=1, bias=True)

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2


        #####################
        # fine output layer
        #####################
        fine_feature_dim =  config.fine_feature_dim
        self.fine_out = nn.Conv1d(out_dim, fine_feature_dim, kernel_size=1, bias=True)




    def forward(self, batch, phase = 'encode'):
        # Get input features

        if phase == 'coarse' :

            x = batch['features'].clone().detach()
            # 1. joint encoder part
            self.skip_x = []
            for block_i, block_op in enumerate(self.encoder_blocks):
                if block_i in self.encoder_skips:
                    self.skip_x.append(x)
                x = block_op(x, batch)  # [N,C]

            for block_i, block_op in enumerate(self.decoder_blocks):
                if block_i in self.decoder_concats:
                    x = torch.cat([x, self.skip_x.pop()], dim=1)
                x = block_op(x, batch)
                if block_i == 1 :
                    coarse_feats = x.transpose(0,1).unsqueeze(0)  #[B, C, N]
                    coarse_feats = self.coarse_out(coarse_feats)  #[B, C, N]
                    coarse_feats = coarse_feats.transpose(1,2).squeeze(0)

                    return coarse_feats #[N,C2]

        #
        # elif phase == "fine":
        #
        #     coarse_feats = batch['coarse_feats']
        #     coarse_feats = coarse_feats.transpose(0,1).unsqueeze(0)
        #     coarse_feats = self.coarse_in(coarse_feats)
        #     x = coarse_feats.transpose(1,2).squeeze(0)
        #
        #
        #     for block_i, block_op in enumerate(self.decoder_blocks):
        #         if block_i > 1  :
        #             if block_i in self.decoder_concats:
        #                 x = torch.cat([x, self.skip_x.pop()], dim=1)
        #             x = block_op(x, batch)
        #
        #     fine_feats = x.transpose(0, 1).unsqueeze(0)  # [1, C, N]
        #     fine_feats = self.fine_out(fine_feats)  # [1, C, N]
        #     fine_feats = fine_feats.transpose(1, 2).squeeze(0)
        #
        #     return fine_feats