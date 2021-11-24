import numpy as np
from functools import partial
import torch
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
from datasets._3dmatch import _3DMatch
from datasets._4dmatch import _4DMatch
from datasets.utils import blend_scene_flow, multual_nn_correspondence
from lib.visualization import *

from torch.utils.data import DataLoader

def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)

def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)



def collate_fn_3dmatch(list_data, config, neighborhood_limits ):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []

    correspondences_list = []
    src_pcd_list = []
    tgt_pcd_list = []

    batched_rot = []
    batched_trn = []

    gt_cov_list = []

    for ind, ( src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trn, gt_cov) in enumerate(list_data):

        correspondences_list.append(correspondences )
        src_pcd_list.append(torch.from_numpy(src_pcd) )
        tgt_pcd_list.append(torch.from_numpy(tgt_pcd) )

        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(src_feats)
        batched_features_list.append(tgt_feats)
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))



        batched_rot.append( torch.from_numpy(rot).float())
        batched_trn.append( torch.from_numpy(trn).float())

        gt_cov_list.append(gt_cov)


    gt_cov_list = None if gt_cov_list[0] is None \
        else np.stack(gt_cov_list, axis=0)

    # if timers: cnter['collate_load_batch'] = time.time() - st

    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    batched_rot = torch.stack(batched_rot,dim=0)
    batched_trn = torch.stack(batched_trn,dim=0)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []


    # construt kpfcn inds
    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []





    # coarse infomation
    coarse_level = config.coarse_level
    pts_num_coarse = input_batches_len[coarse_level].view(-1, 2)
    b_size = pts_num_coarse.shape[0]
    src_pts_max, tgt_pts_max = pts_num_coarse.amax(dim=0)
    coarse_pcd = input_points[coarse_level] # .numpy()
    coarse_matches= []
    src_ind_coarse_split= [] # src_feats shape :[b_size * src_pts_max]
    src_ind_coarse = []
    tgt_ind_coarse_split= []
    tgt_ind_coarse = []
    accumu = 0
    src_mask = torch.zeros([b_size, src_pts_max], dtype=torch.bool)
    tgt_mask = torch.zeros([b_size, tgt_pts_max], dtype=torch.bool)

    #grid subsample fine level points for differentiable matching
    fine_pts, fine_length = batch_grid_subsampling_kpconv(input_points[0], input_batches_len[0], sampleDl=dl*0.5*0.85)
    fine_ind = batch_neighbors_kpconv(fine_pts, input_points[0], fine_length, input_batches_len[0], dl*0.5*0.85, 1).squeeze().long()


    for entry_id, cnt in enumerate( pts_num_coarse ): #input_batches_len[-1].numpy().reshape(-1,2)) :

        n_s_pts, n_t_pts = cnt

        '''split mask for bottlenect feats'''
        src_mask[entry_id][:n_s_pts] = 1
        tgt_mask[entry_id][:n_t_pts] = 1


        '''split indices of bottleneck feats'''
        src_ind_coarse_split.append( torch.arange( n_s_pts ) + entry_id * src_pts_max )
        tgt_ind_coarse_split.append( torch.arange( n_t_pts ) + entry_id * tgt_pts_max )
        src_ind_coarse.append( torch.arange( n_s_pts ) + accumu )
        tgt_ind_coarse.append( torch.arange( n_t_pts ) + accumu + n_s_pts )


        '''get match at coarse level'''
        c_src_pcd = coarse_pcd[accumu : accumu + n_s_pts]
        c_tgt_pcd = coarse_pcd[accumu + n_s_pts: accumu + n_s_pts + n_t_pts]
        s_pc_wrapped = (torch.matmul( batched_rot[entry_id], c_src_pcd.T ) + batched_trn [entry_id]).T
        coarse_match_gt = torch.from_numpy( multual_nn_correspondence(s_pc_wrapped.numpy(), c_tgt_pcd.numpy(), search_radius=config['coarse_match_radius'])  )# 0.1m scaled
        coarse_matches.append(coarse_match_gt)

        accumu = accumu + n_s_pts + n_t_pts

        vis=False # for debug
        if vis :
            viz_coarse_nn_correspondence_mayavi(c_src_pcd, c_tgt_pcd, coarse_match_gt, scale_factor=0.04)




        vis=False # for debug
        if vis :
            pass
            import mayavi.mlab as mlab

            # src_nei_valid = src_nei_mask[coarse_match_gt[0]].view(-1)
            # tgt_nei_valid = tgt_nei_mask[coarse_match_gt[1]].view(-1)
            #
            # f_src_pcd = src_m_nei_pts.view(-1, 3)[src_nei_valid]
            # f_tgt_pcd = tgt_m_nei_pts.view(-1,3)[tgt_nei_valid]
            #
            # mlab.points3d(f_src_pcd[:, 0], f_src_pcd[:, 1], f_src_pcd[:, 2], scale_factor=0.02,color=c_gray1)
            # mlab.points3d(f_tgt_pcd[:, 0], f_tgt_pcd[:, 1], f_tgt_pcd[:, 2], scale_factor=0.02,color=c_gray2)
            #
            # src_m_nn_pts =src_m_nn_pts.view(-1, 3)
            # src_m_nn_pts_wrapped = src_m_nn_pts_wrapped.view(-1,3)
            # tgt_m_nn_pts =  tgt_m_nei_pts [ torch.arange(tgt_m_nei_pts.shape[0]), nni.view(-1), ... ]
            # mlab.points3d(src_m_nn_pts[:, 0], src_m_nn_pts[:, 1], src_m_nn_pts[:, 2], scale_factor=0.04,color=c_red)
            # mlab.points3d(src_m_nn_pts_wrapped[:, 0], src_m_nn_pts_wrapped[:, 1], src_m_nn_pts_wrapped[:, 2], scale_factor=0.04,color=c_red)
            # mlab.points3d(tgt_m_nn_pts[:, 0], tgt_m_nn_pts[:, 1], tgt_m_nn_pts[:, 2], scale_factor=0.04 ,color=c_blue)
            # mlab.show()
            # viz_coarse_nn_correspondence_mayavi(c_src_pcd, c_tgt_pcd, coarse_match_gt,
            #                                     f_src_pcd=src_m_nei_pts.view(-1,3)[src_nei_valid],
            #                                     f_tgt_pcd=tgt_m_nei_pts.view(-1,3)[tgt_nei_valid], scale_factor=0.08)



    src_ind_coarse_split = torch.cat(src_ind_coarse_split)
    tgt_ind_coarse_split = torch.cat(tgt_ind_coarse_split)
    src_ind_coarse = torch.cat(src_ind_coarse)
    tgt_ind_coarse = torch.cat(tgt_ind_coarse)


    dict_inputs = {
        'src_pcd_list': src_pcd_list,
        'tgt_pcd_list': tgt_pcd_list,
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'coarse_matches': coarse_matches,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_ind_coarse_split': src_ind_coarse_split,
        'tgt_ind_coarse_split': tgt_ind_coarse_split,
        'src_ind_coarse': src_ind_coarse,
        'tgt_ind_coarse': tgt_ind_coarse,
        'batched_rot': batched_rot,
        'batched_trn': batched_trn,
        'gt_cov': gt_cov_list,
        #for refine
        'correspondences_list': correspondences_list,
        'fine_ind': fine_ind,
        'fine_pts': fine_pts,
        'fine_length': fine_length
    }

    return dict_inputs



def collate_fn_4dmatch(list_data, config, neighborhood_limits ):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []

    correspondences_list = []
    src_pcd_list = []
    tgt_pcd_list = []

    batched_rot = []
    batched_trn = []

    sflow_list = []
    metric_index_list = [] #for feature matching recall computation

    for ind, ( src_pcd, tgt_pcd, src_feats, tgt_feats, correspondences, rot, trn, s2t_flow, metric_index) in enumerate(list_data):

        correspondences_list.append(correspondences )
        src_pcd_list.append(torch.from_numpy(src_pcd) )
        tgt_pcd_list.append(torch.from_numpy(tgt_pcd) )

        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(src_feats)
        batched_features_list.append(tgt_feats)
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))



        batched_rot.append( torch.from_numpy(rot).float())
        batched_trn.append( torch.from_numpy(trn).float())

        # gt_cov_list.append(gt_cov)
        sflow_list.append( torch.from_numpy(s2t_flow).float() )

        if metric_index is None:
            metric_index_list = None
        else :
            metric_index_list.append ( torch.from_numpy(metric_index))




    # if timers: cnter['collate_load_batch'] = time.time() - st

    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    batched_rot = torch.stack(batched_rot,dim=0)
    batched_trn = torch.stack(batched_trn,dim=0)

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []


    # construt kpfcn inds
    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []


    # coarse infomation
    coarse_level = config.coarse_level
    pts_num_coarse = input_batches_len[coarse_level].view(-1, 2)
    b_size = pts_num_coarse.shape[0]
    src_pts_max, tgt_pts_max = pts_num_coarse.amax(dim=0)
    coarse_pcd = input_points[coarse_level] # .numpy()
    coarse_matches= []
    coarse_flow = []
    src_ind_coarse_split= [] # src_feats shape :[b_size * src_pts_max]
    src_ind_coarse = []
    tgt_ind_coarse_split= []
    tgt_ind_coarse = []
    accumu = 0
    src_mask = torch.zeros([b_size, src_pts_max], dtype=torch.bool)
    tgt_mask = torch.zeros([b_size, tgt_pts_max], dtype=torch.bool)


    for entry_id, cnt in enumerate( pts_num_coarse ): #input_batches_len[-1].numpy().reshape(-1,2)) :

        n_s_pts, n_t_pts = cnt

        '''split mask for bottlenect feats'''
        src_mask[entry_id][:n_s_pts] = 1
        tgt_mask[entry_id][:n_t_pts] = 1


        '''split indices of bottleneck feats'''
        src_ind_coarse_split.append( torch.arange( n_s_pts ) + entry_id * src_pts_max )
        tgt_ind_coarse_split.append( torch.arange( n_t_pts ) + entry_id * tgt_pts_max )
        src_ind_coarse.append( torch.arange( n_s_pts ) + accumu )
        tgt_ind_coarse.append( torch.arange( n_t_pts ) + accumu + n_s_pts )


        '''get match at coarse level'''
        c_src_pcd_np = coarse_pcd[accumu : accumu + n_s_pts].numpy()
        c_tgt_pcd_np = coarse_pcd[accumu + n_s_pts: accumu + n_s_pts + n_t_pts].numpy()
        #interpolate flow
        f_src_pcd = batched_points_list[entry_id * 2]
        c_flow = blend_scene_flow( c_src_pcd_np, f_src_pcd, sflow_list[entry_id].numpy(), knn=3)
        c_src_pcd_deformed = c_src_pcd_np + c_flow
        s_pc_wrapped = (np.matmul( batched_rot[entry_id].numpy(), c_src_pcd_deformed.T ) + batched_trn [entry_id].numpy()).T
        coarse_match_gt = torch.from_numpy( multual_nn_correspondence(s_pc_wrapped , c_tgt_pcd_np , search_radius=config['coarse_match_radius'])  )# 0.1m scaled
        coarse_matches.append(coarse_match_gt)
        coarse_flow.append(torch.from_numpy(c_flow) )

        accumu = accumu + n_s_pts + n_t_pts

        vis=False # for debug
        if vis :
            viz_coarse_nn_correspondence_mayavi(c_src_pcd_np, c_tgt_pcd_np, coarse_match_gt, scale_factor=0.02)


    src_ind_coarse_split = torch.cat(src_ind_coarse_split)
    tgt_ind_coarse_split = torch.cat(tgt_ind_coarse_split)
    src_ind_coarse = torch.cat(src_ind_coarse)
    tgt_ind_coarse = torch.cat(tgt_ind_coarse)


    dict_inputs = {
        'src_pcd_list': src_pcd_list,
        'tgt_pcd_list': tgt_pcd_list,
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'coarse_matches': coarse_matches,
        'coarse_flow' : coarse_flow,
        'src_mask': src_mask,
        'tgt_mask': tgt_mask,
        'src_ind_coarse_split': src_ind_coarse_split,
        'tgt_ind_coarse_split': tgt_ind_coarse_split,
        'src_ind_coarse': src_ind_coarse,
        'tgt_ind_coarse': tgt_ind_coarse,
        'batched_rot': batched_rot,
        'batched_trn': batched_trn,
        'sflow_list': sflow_list,
        "metric_index_list": metric_index_list
    }

    return dict_inputs



def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)

        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)

        # if timer.total_time - last_display > 0.1:
        #     last_display = timer.total_time
        #     print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits




def get_datasets(config):
    if (config.dataset == '3dmatch'):
        train_set = _3DMatch(config, 'train', data_augmentation=True)
        val_set = _3DMatch(config, 'val', data_augmentation=False)
        test_set = _3DMatch(config, 'test', data_augmentation=False)
    elif(config.dataset == '4dmatch'):
        train_set = _4DMatch(config, 'train', data_augmentation=True)
        val_set = _4DMatch(config, 'val', data_augmentation=False)
        test_set = _4DMatch(config, 'test', data_augmentation=False)
    else:
        raise NotImplementedError

    return train_set, val_set, test_set



def get_dataloader(dataset, config, shuffle=True, neighborhood_limits=None):

    if config.dataset=='4dmatch':
        collate_fn = collate_fn_4dmatch
    elif config.dataset == '3dmatch':
        collate_fn = collate_fn_3dmatch
    else:
        raise NotImplementedError()

    if neighborhood_limits is None:
        neighborhood_limits = calibrate_neighbors(dataset, config['kpfcn_config'], collate_fn=collate_fn)
    print("neighborhood:", neighborhood_limits)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=shuffle,
        num_workers=config['num_workers'],
        collate_fn=partial(collate_fn, config=config['kpfcn_config'], neighborhood_limits=neighborhood_limits ),
        drop_last=False
    )

    return dataloader, neighborhood_limits




if __name__ == '__main__':


    pass
