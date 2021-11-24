import torch
import torch.nn as nn
import torch.nn.functional as F
from models.position_encoding import VolumetricPositionEncoding as VolPE

def log_optimal_transport(scores, alpha, iters, src_mask, tgt_mask ):

    b, m, n = scores.shape

    if src_mask is None:
        ms = m
        ns = n
    else :
        ms = src_mask.sum(dim=1, keepdim=True)
        ns = tgt_mask.sum(dim=1, keepdim=True)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    Z = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log() # [b, 1]

    log_mu = torch.cat([norm  .repeat(1, m), ns.log() + norm], dim=1)
    log_nu = torch.cat([norm.repeat(1, n), ms.log() + norm], dim=1)

    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp( Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)

    Z=  Z + u.unsqueeze(2) + v.unsqueeze(1)

    Z = Z - norm.view(-1,1,1)

    return Z


class Matching(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.match_type = config['match_type']

        self.confidence_threshold = config['confidence_threshold']

        d_model = config['feature_dim']

        self.src_proj = nn.Linear(d_model, d_model, bias=False)
        self.tgt_proj = nn.Linear(d_model, d_model, bias=False)

        self.entangled= config['entangled']


        if self.match_type == "dual_softmax":
            self.temperature = config['dsmax_temperature']
        elif self.match_type == 'sinkhorn':
            #sinkhorn algorithm
            self.skh_init_bin_score = config['skh_init_bin_score']
            self.skh_iters = config['skh_iters']
            self.skh_prefilter = config['skh_prefilter']
            self.bin_score = nn.Parameter(
                torch.tensor( self.skh_init_bin_score,  requires_grad=True))
        else:
            raise NotImplementedError()


    @staticmethod
    @torch.no_grad()
    def get_match( conf_matrix, thr, mutual=True):

        mask = conf_matrix > thr

        #mutual nearest
        if mutual:
            mask = mask \
                   * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                   * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        #find all valid coarse matches
        index = (mask==True).nonzero()
        b_ind, src_ind, tgt_ind = index[:,0], index[:,1], index[:,2]
        mconf = conf_matrix[b_ind, src_ind, tgt_ind]

        return index, mconf, mask

    @staticmethod
    @torch.no_grad()
    def get_topk_match( conf_matrix, thr, mutual=True):

        mask = conf_matrix > thr

        #mutual nearest
        if mutual:
            mask = mask \
                   * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) \
                   * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

        #find all valid coarse matches
        index = (mask==True).nonzero()
        b_ind, src_ind, tgt_ind = index[:,0], index[:,1], index[:,2]
        mconf = conf_matrix[b_ind, src_ind, tgt_ind]

        return index, mconf, mask










    def forward(self, src_feats, tgt_feats, src_pe, tgt_pe, src_mask, tgt_mask, data, pe_type="rotary"):
        '''
        @param src_feats: [B, S, C]
        @param tgt_feats: [B, T, C]
        @param src_mask: [B, S]
        @param tgt_mask: [B, T]
        @return:
        '''

        src_feats = self.src_proj(src_feats)
        tgt_feats = self.src_proj(tgt_feats)


        data["src_feats_nopos"] = src_feats
        data["tgt_feats_nopos"] = tgt_feats


        if not self.entangled :
            src_feats = VolPE.embed_pos(pe_type, src_feats, src_pe)
            tgt_feats = VolPE.embed_pos(pe_type, tgt_feats, tgt_pe)


        data["src_feats"] = src_feats
        data["tgt_feats"] = tgt_feats


        src_feats, tgt_feats = map(lambda feat: feat / feat.shape[-1] ** .5,
                                   [src_feats, tgt_feats])

        if self.match_type == "dual_softmax":
            # dual softmax matching
            sim_matrix_1 = torch.einsum("bsc,btc->bst", src_feats, tgt_feats) / self.temperature

            if src_mask is not None:
                sim_matrix_2 = sim_matrix_1.clone()
                sim_matrix_1.masked_fill_(~src_mask[:, :, None], float('-inf'))
                sim_matrix_2.masked_fill_(~tgt_mask[:, None, :], float('-inf'))
                conf_matrix = F.softmax(sim_matrix_1, 1) * F.softmax(sim_matrix_2, 2)
            else :
                conf_matrix = F.softmax(sim_matrix_1, 1) * F.softmax(sim_matrix_1, 2)

        elif self.match_type == "sinkhorn" :
            #optimal transport sinkhoron
            sim_matrix = torch.einsum("bsc,btc->bst", src_feats, tgt_feats)

            if src_mask is not None:
                sim_matrix.masked_fill_(
                    ~(src_mask[..., None] * tgt_mask[:, None]).bool(), float('-inf'))

            log_assign_matrix = log_optimal_transport( sim_matrix, self.bin_score, self.skh_iters, src_mask, tgt_mask)

            assign_matrix = log_assign_matrix.exp()
            conf_matrix = assign_matrix[:, :-1, :-1].contiguous()

        coarse_match, _, _ = self.get_match(conf_matrix, self.confidence_threshold)
        return conf_matrix, coarse_match

