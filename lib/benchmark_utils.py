import os,re,sys,json,yaml,random, glob, argparse, torch, pickle
from tqdm import tqdm
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d

_EPS = 1e-7  # To prevent division by zero


def viz_coarse_nn_correspondence_mayavi(s_pc, t_pc, good_c, bad_c, f_src_pcd=None, f_tgt_pcd=None, scale_factor=0.02):
    '''
    @param s_pc:  [S,3]
    @param t_pc:  [T,3]
    @param correspondence: [2,K]
    @param f_src_pcd: [S1,3]
    @param f_tgt_pcd: [T1,3]
    @param scale_factor:
    @return:
    '''

    import mayavi.mlab as mlab
    c_red = (224. / 255., 0 / 255., 0 / 255.)
    c_pink = (224. / 255., 75. / 255., 232. / 255.)
    c_blue = (0. / 255., 0. / 255., 255. / 255.)
    c_green = (0. / 255., 255. / 255., 0. / 255.)
    c_gray1 = (255 / 255., 255 / 255., 125 / 255.)
    c_gray2 = (125. / 255., 125. / 255., 255. / 255.)

    if f_src_pcd is not None:
        mlab.points3d(f_src_pcd[:, 0], f_src_pcd[:, 1], f_src_pcd[:, 2], scale_factor=scale_factor * 0.25,
                      color=c_gray1)
    else:
        mlab.points3d(s_pc[:, 0], s_pc[:, 1], s_pc[:, 2], scale_factor=scale_factor * 0.75, color=c_gray1)

    if f_tgt_pcd is not None:
        mlab.points3d(f_tgt_pcd[:, 0], f_tgt_pcd[:, 1], f_tgt_pcd[:, 2], scale_factor=scale_factor * 0.25,
                      color=c_gray2)
    else:
        mlab.points3d(t_pc[:, 0], t_pc[:, 1], t_pc[:, 2], scale_factor=scale_factor * 0.75, color=c_gray2)

    s_cpts_god = s_pc[good_c[0]]
    t_cpts_god = t_pc[good_c[1]]
    flow_good = t_cpts_god - s_cpts_god

    s_cpts_bd = s_pc[bad_c[0]]
    t_cpts_bd = t_pc[bad_c[1]]
    flow_bad = t_cpts_bd - s_cpts_bd

    def match_draw(s_cpts, t_cpts, flow, color):

        mlab.points3d(s_cpts[:, 0], s_cpts[:, 1], s_cpts[:, 2], scale_factor=scale_factor * 0.35, color=c_blue)
        mlab.points3d(t_cpts[:, 0], t_cpts[:, 1], t_cpts[:, 2], scale_factor=scale_factor * 0.35, color=c_pink)
        mlab.quiver3d(s_cpts[:, 0], s_cpts[:, 1], s_cpts[:, 2], flow[:, 0], flow[:, 1], flow[:, 2],
                      scale_factor=1, mode='2ddash', line_width=1., color=color)

    match_draw(s_cpts_god, t_cpts_god, flow_good, c_green)
    match_draw(s_cpts_bd, t_cpts_bd, flow_bad, c_red)

    mlab.show()


def correspondence_viz(src_raw, tgt_raw, src_pcd, tgt_pcd, corrs, inlier_mask, max=200):
    perm = np.random.permutation(corrs.shape[1])
    ind = perm[:max]

    corrs = corrs[:, ind]
    inlier_mask = inlier_mask[ind]

    good_c = corrs[:, inlier_mask]
    bad_c = corrs[:, ~inlier_mask]

    offset = np.array([[1.45, 0, 0]])
    # src_pcd = src_pcd + offset
    # src_raw = src_raw + offset
    tgt_pcd = tgt_pcd + offset
    tgt_raw = tgt_raw + offset

    viz_coarse_nn_correspondence_mayavi(src_pcd, tgt_pcd, good_c, bad_c, src_raw, tgt_raw, scale_factor=0.07)


def fmr_wrt_distance(data,split,inlier_ratio_threshold=0.05):
    """
    calculate feature match recall wrt distance threshold
    """
    fmr_wrt_distance =[]
    for distance_threshold in range(1,21):
        inlier_ratios =[]
        distance_threshold /=100.0
        for idx in range(data.shape[0]):
            inlier_ratio = (data[idx] < distance_threshold).mean()
            inlier_ratios.append(inlier_ratio)
        fmr = 0
        for ele in split:
            fmr += (np.array(inlier_ratios[ele[0]:ele[1]]) > inlier_ratio_threshold).mean()
        fmr /= 8
        fmr_wrt_distance.append(fmr*100)
    return fmr_wrt_distance

def fmr_wrt_inlier_ratio(data, split, distance_threshold=0.1):
    """
    calculate feature match recall wrt inlier ratio threshold
    """
    fmr_wrt_inlier =[]
    for inlier_ratio_threshold in range(1,21):
        inlier_ratios =[]
        inlier_ratio_threshold /=100.0
        for idx in range(data.shape[0]):
            inlier_ratio = (data[idx] < distance_threshold).mean()
            inlier_ratios.append(inlier_ratio)
        
        fmr = 0
        for ele in split:
            fmr += (np.array(inlier_ratios[ele[0]:ele[1]]) > inlier_ratio_threshold).mean()
        fmr /= 8
        fmr_wrt_inlier.append(fmr*100)

    return fmr_wrt_inlier



def to_tensor(array):
    """
    Convert array to tensor
    """
    if(not isinstance(array,torch.Tensor)):
        return torch.from_numpy(array).float()
    else:
        return array

def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor

def to_tsfm(rot,trans):
    tsfm = np.eye(4)
    tsfm[:3,:3]=rot
    tsfm[:3,3]=trans.flatten()
    return tsfm
    
def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats

def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):

    src_pcd.transform(trans)
    correspondences =  KDTree_corr ( src_pcd, tgt_pcd, search_voxel_size, K=None)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def KDTree_corr ( src_pcd_transformed, tgt_pcd, search_voxel_size, K=None):

    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)
    correspondences = []
    for i, point in enumerate(src_pcd_transformed.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])
    correspondences = np.array(correspondences)
    return correspondences



def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]

def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]

def random_sample(pcd, feats, N):
    """
    Do random sampling to get exact N points and associated features
    pcd:    [N,3]
    feats:  [N,C]
    """
    if(isinstance(pcd,torch.Tensor)):
        n1 = pcd.size(0)
    elif(isinstance(pcd, np.ndarray)):
        n1 = pcd.shape[0]

    if n1 == N:
        return pcd, feats

    if n1 > N:
        choice = np.random.permutation(n1)[:N]
    else:
        choice = np.random.choice(n1, N)

    return pcd[choice], feats[choice]
    
def get_angle_deviation(R_pred,R_gt):
    """
    Calculate the angle deviation between two rotaion matrice
    The rotation error is between [0,180]
    Input:
        R_pred: [B,3,3]
        R_gt  : [B,3,3]
    Return: 
        degs:   [B]
    """
    R=np.matmul(R_pred,R_gt.transpose(0,2,1))
    tr=np.trace(R,0,1,2) 
    rads=np.arccos(np.clip((tr-1)/2,-1,1))  # clip to valid range
    degs=rads/np.pi*180

    return degs

def ransac_pose_estimation(src_pcd, tgt_pcd, src_feat, tgt_feat, mutual = False, distance_threshold = 0.05, ransac_n = 3):
    """
    RANSAC pose estimation with two checkers
    We follow D3Feat to set ransac_n = 3 for 3DMatch and ransac_n = 4 for KITTI. 
    For 3DMatch dataset, we observe significant improvement after changing ransac_n from 4 to 3.
    """
    if(mutual):
        if(torch.cuda.device_count()>=1):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        src_feat, tgt_feat = to_tensor(src_feat), to_tensor(tgt_feat)
        scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0,1).to(device)).cpu()
        selection = mutual_selection(scores[None,:,:])[0]
        row_sel, col_sel = np.where(selection)
        corrs = o3d.utility.Vector2iVector(np.array([row_sel,col_sel]).T)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
            source=src_pcd, target=tgt_pcd,corres=corrs, 
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000))
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_feats = to_o3d_feats(src_feat)
        tgt_feats = to_o3d_feats(tgt_feat)

        result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
            src_pcd, tgt_pcd, src_feats, tgt_feats,distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False), ransac_n,
            [o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.registration.RANSACConvergenceCriteria(50000, 1000))
            
    return result_ransac.transformation

def get_inlier_ratio(src_pcd, tgt_pcd, src_feat, tgt_feat, rot, trans, inlier_distance_threshold = 0.1):
    """
    Compute inlier ratios with and without mutual check, return both
    """
    src_pcd = to_tensor(src_pcd)
    tgt_pcd = to_tensor(tgt_pcd)
    src_feat = to_tensor(src_feat)
    tgt_feat = to_tensor(tgt_feat)
    rot, trans = to_tensor(rot), to_tensor(trans)

    results =dict()
    results['w']=dict()
    results['wo']=dict()

    if(torch.cuda.device_count()>=1):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    src_pcd = (torch.matmul(rot, src_pcd.transpose(0,1)) + trans).transpose(0,1)
    scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0,1).to(device)).cpu()

    ########################################
    # 1. calculate inlier ratios wo mutual check
    _, idx = scores.max(-1)
    dist = torch.norm(src_pcd- tgt_pcd[idx],dim=1)
    results['wo']['distance'] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results['wo']['inlier_ratio'] = c_inlier_ratio

    ########################################
    # 2. calculate inlier ratios w mutual check
    selection = mutual_selection(scores[None,:,:])[0]
    row_sel, col_sel = np.where(selection)
    dist = torch.norm(src_pcd[row_sel]- tgt_pcd[col_sel],dim=1)
    results['w']['distance'] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results['w']['inlier_ratio'] = c_inlier_ratio

    return results


def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column
    
    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N] 
    """
    score_mat=to_array(score_mat)
    if(score_mat.ndim==2):
        score_mat=score_mat[None,:,:]
    
    mutuals=np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]): # loop through the batch
        c_mat=score_mat[i]
        flag_row=np.zeros_like(c_mat)
        flag_column=np.zeros_like(c_mat)

        max_along_row=np.argmax(c_mat,1)[:,None]
        max_along_column=np.argmax(c_mat,0)[None,:]
        np.put_along_axis(flag_row,max_along_row,1,1)
        np.put_along_axis(flag_column,max_along_column,1,0)
        mutuals[i]=(flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)  

