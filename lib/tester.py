from lib.trainer import Trainer
import torch
from tqdm import tqdm
from models.loss import MatchMotionLoss as MML
import numpy as np
from models.matching import Matching as CM
import math

class _3DMatchTester(Trainer):
    """
    3DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self, args)


    def test(self):

        n = 3

        afmr = 0.
        arr = 0
        air = 0

        for i in range(n): # combat ransac nondeterministic

            thr =0.05
            rr, ir, fmr = self.test_thr(thr)
            afmr+=fmr
            arr+=rr
            air+=ir
            print( "conf_threshold", thr, "registration recall:", rr, " Inlier rate:", ir, "FMR:", fmr)

        print("average registration recall:", arr / n, afmr/n, air/n)
        # print ("registration recall:", self.test_thr())

    def test_thr(self, conf_threshold=None):

        # print('Start to evaluate on test datasets...')
        # os.makedirs(f'{self.snapshot_dir}/{self.config.dataset}',exist_ok=True)

        num_iter = math.ceil(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()


        self.model.eval()




        success1 = 0.
        IR=0.
        FMR=0.

        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch

                ##################################
                if self.timers: self.timers.tic('load batch')
                inputs = c_loader_iter.next()
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    elif type(v) in [dict, float, type(None), np.ndarray]:
                        pass
                    else:
                        inputs[k] = v.to(self.device)
                if self.timers: self.timers.toc('load batch')
                ##################################


                if self.timers: self.timers.tic('forward pass')
                data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
                if self.timers: self.timers.toc('forward pass')



                match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=conf_threshold, mutual=False)
                rot, trn = MML.ransac_regist_coarse(data['s_pcd'], data['t_pcd'], data['src_mask'], data['tgt_mask'], match_pred)

                ir = MML.compute_inlier_ratio(match_pred, data, inlier_thr=0.1).mean()

                rr1 = MML.compute_registration_recall(rot, trn, data, thr=0.2) # 0.2m



                vis = False
                if vis:
                    pcd = data['points'][0].cpu().numpy()
                    lenth = data['stack_lengths'][0][0]
                    spcd, tpcd = pcd[:lenth] , pcd[lenth:]

                    import mayavi.mlab as mlab
                    c_red = (224. / 255., 0 / 255., 125 / 255.)
                    c_pink = (224. / 255., 75. / 255., 232. / 255.)
                    c_blue = (0. / 255., 0. / 255., 255. / 255.)
                    scale_factor = 0.02
                    # mlab.points3d(s_pc[ :, 0]  , s_pc[ :, 1],  s_pc[:,  2],  scale_factor=scale_factor , color=c_blue)
                    mlab.points3d(spcd[:, 0], spcd[:, 1], spcd[:, 2], scale_factor=scale_factor,
                                  color=c_red)
                    mlab.points3d(tpcd[:, 0], tpcd[:, 1], tpcd[:, 2], scale_factor=scale_factor,
                                  color=c_blue)
                    mlab.show()

                    spcd = ( np.matmul(rot, spcd.T) + trn ).T
                    mlab.points3d(spcd[:, 0], spcd[:, 1], spcd[:, 2], scale_factor=scale_factor,
                                  color=c_red)
                    mlab.points3d(tpcd[:, 0], tpcd[:, 1], tpcd[:, 2], scale_factor=scale_factor,
                                  color=c_blue)
                    mlab.show()

                bs = len(rot)
                assert  bs==1
                success1 += bs * rr1
                IR += bs*ir
                FMR += (ir>0.05).float()


            recall1 = success1/len(self.loader['test'].dataset)
            IRate = IR/len(self.loader['test'].dataset)
            FMR = FMR/len(self.loader['test'].dataset)
            return recall1, IRate, FMR


def blend_anchor_motion (query_loc, reference_loc, reference_flow , knn=3, search_radius=0.1) :
    '''approximate flow on query points
    this function assume query points are sub- or un-sampled from reference locations
    @param query_loc:[m,3]
    @param reference_loc:[n,3]
    @param reference_flow:[n,3]
    @param knn:
    @return:
        blended_flow:[m,3]
    '''
    from datasets.utils import knn_point_np
    dists, idx = knn_point_np (knn, reference_loc, query_loc)
    dists[dists < 1e-10] = 1e-10
    mask = dists>search_radius
    dists[mask] = 1e+10
    weight = 1.0 / dists
    weight = weight / np.sum(weight, -1, keepdims=True)  # [B,N,3]
    blended_flow = np.sum (reference_flow [idx] * weight.reshape ([-1, knn, 1]), axis=1, keepdims=False)

    mask = mask.sum(axis=1)<3

    return blended_flow, mask

def compute_nrfmr( match_pred, data, recall_thr=0.04):


    s_pcd, t_pcd = data['s_pcd'], data['t_pcd']

    s_pcd_raw = data ['src_pcd_list']
    sflow_list = data['sflow_list']
    metric_index_list = data['metric_index_list']

    batched_rot = data['batched_rot']  # B,3,3
    batched_trn = data['batched_trn']


    nrfmr = 0.

    for i in range ( len(s_pcd_raw)):

        # get the metric points' transformed position
        metric_index = metric_index_list[i]
        sflow = sflow_list[i]
        s_pcd_raw_i = s_pcd_raw[i]
        metric_pcd = s_pcd_raw_i [ metric_index ]
        metric_sflow = sflow [ metric_index ]
        metric_pcd_deformed = metric_pcd + metric_sflow
        metric_pcd_wrapped_gt = ( torch.matmul( batched_rot[i], metric_pcd_deformed.T) + batched_trn[i] ).T


        # use the match prediction as the motion anchor
        match_pred_i = match_pred[ match_pred[:, 0] == i ]
        s_id , t_id = match_pred_i[:,1], match_pred_i[:,2]
        s_pcd_matched= s_pcd[i][s_id]
        t_pcd_matched= t_pcd[i][t_id]
        motion_pred = t_pcd_matched - s_pcd_matched
        metric_motion_pred, valid_mask = blend_anchor_motion(
            metric_pcd.cpu().numpy(), s_pcd_matched.cpu().numpy(), motion_pred.cpu().numpy(), knn=3, search_radius=0.1)
        metric_pcd_wrapped_pred = metric_pcd + torch.from_numpy(metric_motion_pred).to(metric_pcd)

        debug = False
        if debug:
            import mayavi.mlab as mlab
            c_red = (224. / 255., 0 / 255., 125 / 255.)
            c_pink = (224. / 255., 75. / 255., 232. / 255.)
            c_blue = (0. / 255., 0. / 255., 255. / 255.)
            scale_factor = 0.013
            metric_pcd_wrapped_gt = metric_pcd_wrapped_gt.cpu()
            metric_pcd_wrapped_pred = metric_pcd_wrapped_pred.cpu()
            err = metric_pcd_wrapped_pred - metric_pcd_wrapped_gt
            mlab.points3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], scale_factor=scale_factor, color=c_pink)
            mlab.points3d(metric_pcd_wrapped_pred[ :, 0] , metric_pcd_wrapped_pred[ :, 1], metric_pcd_wrapped_pred[:,  2], scale_factor=scale_factor , color=c_blue)
            mlab.quiver3d(metric_pcd_wrapped_gt[:, 0], metric_pcd_wrapped_gt[:, 1], metric_pcd_wrapped_gt[:, 2], err[:, 0], err[:, 1], err[:, 2],
                          scale_factor=1, mode='2ddash', line_width=1.)
            mlab.show()

        dist = torch.sqrt( torch.sum( (metric_pcd_wrapped_pred - metric_pcd_wrapped_gt)**2, dim=1 ) )

        r = (dist < recall_thr).float().sum() / len(dist)
        nrfmr = nrfmr + r

    nrfmr = nrfmr /len(s_pcd_raw)

    return  nrfmr

class _4DMatchTester(Trainer):
    """
    3DMatch tester
    """
    def __init__(self,args):
        Trainer.__init__(self, args)

    def test(self):

        for thr in [  0.05, 0.1, 0.2]:
        # for thr in [ 0.1 ]:
            import time
            start = time.time()
            ir, fmr, nspl = self.test_thr(thr)
            print( "conf_threshold", thr,  "NFMR:", fmr, " Inlier rate:", ir, "Number sample:", nspl)
            print( "time costs:", time.time() - start)

    def test_thr(self, conf_threshold=None):

        num_iter = math.ceil(len(self.loader['test'].dataset) // self.loader['test'].batch_size)
        c_loader_iter = self.loader['test'].__iter__()


        self.model.eval()


        assert self.loader['test'].batch_size == 1

        IR=0.
        NR_FMR=0.

        inlier_thr = recall_thr = 0.04

        n_sample = 0.

        with torch.no_grad():
            for idx in tqdm(range(num_iter)): # loop through this epoch



                ##################################
                if self.timers: self.timers.tic('load batch')
                inputs = c_loader_iter.next()
                for k, v in inputs.items():
                    if type(v) == list:
                        inputs[k] = [item.to(self.device) for item in v]
                    elif type(v) in [ dict, float, type(None), np.ndarray]:
                        pass
                    else:
                        inputs[k] = v.to(self.device)
                if self.timers: self.timers.toc('load batch')
                ##################################


                if self.timers: self.timers.tic('forward pass')
                data = self.model(inputs, timers=self.timers)  # [N1, C1], [N2, C2]
                if self.timers: self.timers.toc('forward pass')

                match_pred, _, _ = CM.get_match(data['conf_matrix_pred'], thr=conf_threshold, mutual=True)
                ir = MML.compute_inlier_ratio(match_pred, data, inlier_thr=inlier_thr, s2t_flow=data['coarse_flow'][0][None] )[0]

                nrfmr = compute_nrfmr(match_pred, data, recall_thr=recall_thr)

                IR += ir
                NR_FMR += nrfmr

                n_sample += match_pred.shape[0]


            IRate = IR/len(self.loader['test'].dataset)
            NR_FMR = NR_FMR/len(self.loader['test'].dataset)
            n_sample = n_sample/len(self.loader['test'].dataset)

            if self.timers: self.timers.print()

            return IRate, NR_FMR, n_sample





def get_trainer(config):
    if config.dataset == '3dmatch':
        return _3DMatchTester(config)
    elif config.dataset == '4dmatch':
        return _4DMatchTester(config)
    else:
        raise NotImplementedError
