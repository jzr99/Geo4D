# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Initialization functions for global alignment
# --------------------------------------------------------
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import torch
import cv2
import roma
from tqdm import tqdm

from dust3r.utils.geometry import geotrf, inv, get_med_dist_between_poses
from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.viz import to_numpy

from dust3r.cloud_opt.commons import edge_str, i_j_ij, compute_edge_scores
import matplotlib.pyplot as plt
import seaborn as sns
from utils.geometry import point_map_to_depth, intrinsics_from_fov_xy


def fast_pnp_all_image(imshapes, pred_i, device, has_im_poses=True, niter_PnP=10, verbose=True, save_score_path=None, conf=None):
    n_imgs = len(imshapes)
    # eadge_and_scores = compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j)
    # sparse_graph = -dict_to_sparse_graph(eadge_and_scores)
    # msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # temp variable to store 3d points
    pts3d = [None] * len(imshapes)

    # todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs

    # i_j = edge_str(0, 0)
    # im_poses[0] = torch.eye(4, device=device)
    # im_focals[0] = estimate_focal(pred_i[i_j])
    for i in range(n_imgs):
        i_j = edge_str(0, i)
        pts3d[i] = pred_i[i_j].clone()
        H, W, _ = pred_i[i_j].shape
        if conf is None:
            msk = torch.ones((H, W), dtype=torch.bool, device=device)
        else:
            msk = conf[i] > 0.5
            # import pdb;pdb.set_trace()
        res = fast_pnp(pts3d[i], None, msk=msk, device=device, niter_PnP=niter_PnP)
        if res:
            im_focals[i], im_poses[i] = res
        if im_poses[i] is None:
            print(f"Failed to initialize camera pose for image {i}")
            im_poses[i] = torch.eye(4, device=device)
    return pts3d, im_focals, im_poses


@torch.no_grad()
def init_from_group(self, save_score_path=None, save_score_only=False,fast_focal=True, **kw):
    """ Init all camera poses (image-wise and pairwise poses) given
        an initial set of pairwise estimations.
    """
    device = self.device
    # if save_score_only:
    #     eadge_and_scores = compute_edge_scores(map(i_j_ij, self.edges), self.conf_i, self.conf_j)
    #     draw_edge_scores_map(eadge_and_scores, save_score_path)
    #     return
    if self.opt_raydir:
        print('init intrinsics from raymap')
        pts3d, im_focals, im_poses, conf_list = align_group(self.imshapes, self.groups, self.pred_dict, self.conf_dict, device=device, has_im_poses=self.has_im_poses, verbose=self.verbose, save_score_path=save_score_path, raymap_dict=self.raydir_dict, **kw)
    elif fast_focal:
        pts3d, im_focals, im_poses, conf_list = align_group_prefix(self.imshapes, self.groups, self.pred_dict, self.conf_dict, device=device, has_im_poses=self.has_im_poses, verbose=self.verbose, save_score_path=save_score_path, **kw)
    else:
        pts3d, im_focals, im_poses, conf_list = align_group(self.imshapes, self.groups, self.pred_dict, self.conf_dict, device=device, has_im_poses=self.has_im_poses, verbose=self.verbose, save_score_path=save_score_path, **kw)
    
    # (imshapes, pred_i, device, has_im_poses=True, niter_PnP=10, verbose=True, save_score_path=None)

    return init_from_pts3d_group(self, pts3d, im_focals, im_poses, conf_list)

def align_group(imshapes, groups, pred_dict, conf_dict,
                          device, has_im_poses=True, niter_PnP=100, verbose=True, save_score_path=None, raymap_dict=None, fast_focal=False):
    stride = groups[1][0] - groups[0][0]
    n_imgs = len(imshapes)
    # eadge_and_scores = compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j)
    # sparse_graph = -dict_to_sparse_graph(eadge_and_scores)
    # msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # temp variable to store 3d points
    pts3d = [None] * len(imshapes)
    conf_list = [None] * len(imshapes)

    # todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs
    # done = {}
    done = set()

    if fast_focal:
        ref_pointmap = []
        ref_conf = []
        ref_idx = []
        for i, group in enumerate(groups):
            ref_pointmap.append(pred_dict[i][0])
            ref_conf.append(conf_dict[i][0])
            ref_idx.append(group[0])
        import pdb;pdb.set_trace()
        ref_pointmap = torch.stack(ref_pointmap, dim=0)
        ref_conf = torch.stack(ref_conf, dim=0)
        B, H, W, _ = ref_pointmap.shape
        mask = ref_conf > 0.5
        depth, fov_x, fov_y, optim_shift = point_map_to_depth(ref_pointmap, mask, downsample_size=(64,64))
        intrinsics = intrinsics_from_fov_xy(fov_x, fov_y)
        focal_group = ((intrinsics[:,0,0] * W) + (intrinsics[:, 1,1] * H)) / 2
        # for i, idx in enumerate(ref_idx):
        #     im_focals[idx] = focal_group[i]
        # for i in range(n_imgs):
        #     if im_focals[i] is None:
        #         im_focals[i] = im_focals[i-1]
        

    # set the first frame to be the world coordinate
    first_group = groups[0]
    for group_idx, img_idx in enumerate(first_group):
        pts3d[img_idx] = pred_dict[0][group_idx].clone()
        conf_list[img_idx] = conf_dict[0][group_idx].clone()
        if raymap_dict is not None:
            im_focals[img_idx] = estimate_focal(raymap_dict[0][group_idx])
            print(f"Estimated focal length for image {img_idx} is {im_focals[img_idx]}")
            # im_focals[img_idx] = estimate_focal(pred_dict[0][group_idx])
        # pts3d[i] = pred_i[i_j].clone()
        H, W, _ = pred_dict[0][group_idx].shape
        
        if conf_dict is None:
            msk = torch.ones((H, W), dtype=torch.bool, device=device)
        else:
            msk = conf_dict[0][group_idx] > 0.5
            # import pdb;pdb.set_trace()
        res = fast_pnp(pts3d[img_idx], im_focals[img_idx], msk=msk, device=device, niter_PnP=niter_PnP)
        # res = fast_pnp(pts3d[img_idx], im_focals[first_group[0]], msk=msk, device=device, niter_PnP=niter_PnP) # here we init the focal length using the first frame of the first group
        if res:
            im_focals[img_idx], im_poses[img_idx] = res
            print("img_idx: ", img_idx, "Focal length by Ransac PnP: ", im_focals[img_idx])
        if im_poses[img_idx] is None:
            print(f"Failed to initialize camera pose for image {img_idx}")
            im_poses[img_idx] = torch.eye(4, device=device)
        done.add(img_idx)
    other_groups = groups[1:]
    for i, group in enumerate(groups):
        if i==0:
            continue
        assert group[0] in done, "The first image of the following group should be in the previous group"
        # TODO we need to use multiple overlapping groups to initialize the camera poses
        seen_imgs_group_idx = []
        seen_imgs_img_idx = []
        pred_temp_list = []
        pts3d_temp_list = []
        conf_temp_list = []
        for group_idx, img_idx in enumerate(group):
            if img_idx in done:
                seen_imgs_group_idx.append(group_idx)
                seen_imgs_img_idx.append(img_idx)
                pred_temp_list.append(pred_dict[i][group_idx])
                pts3d_temp_list.append(pts3d[img_idx])
                conf_temp_list.append(conf_dict[i][group_idx] * conf_list[img_idx])
        pred_temp_list = torch.stack(pred_temp_list)
        pts3d_temp_list = torch.stack(pts3d_temp_list)
        conf_temp_list = torch.stack(conf_temp_list)

        s, R, T = rigid_points_registration(pred_temp_list, pts3d_temp_list, conf=conf_temp_list) 
        # s, R, T = rigid_points_registration(pred_dict[i][seen_imgs_group_idx], pts3d[seen_imgs_img_idx], conf=conf_dict[i][seen_imgs_group_idx])
        # TODO, double check the conf_dict here
        trf = sRT_to_4x4(s, R, T, device)
        for group_idx, img_idx in enumerate(group):
            if pts3d[img_idx] is None:
                pts3d[img_idx] = geotrf(trf, pred_dict[i][group_idx])
                conf_list[img_idx] = conf_dict[i][group_idx]
                done.add(img_idx)
            if group_idx==0 and has_im_poses and im_poses[img_idx] is None:
                im_poses[img_idx] = sRT_to_4x4(1, R, T, device)
                # TODO, if we know the pose, we can estimate the focal length and shift z
            # if group_idx==0 and has_im_poses and im_focals[img_idx] is None:
            #     im_focals[img_idx] = estimate_focal(pred_dict[i][group_idx])
            if raymap_dict is not None and im_focals[img_idx] is None:
                im_focals[img_idx] = estimate_focal(raymap_dict[i][group_idx])
                print(f"Estimated focal length for image {img_idx} is {im_focals[img_idx]}")
        
            if has_im_poses:

                if conf_dict is None:
                    msk = torch.ones((H, W), dtype=torch.bool, device=device)
                else:
                    msk = conf_dict[i][group_idx] > 0.5
                    # import pdb;pdb.set_trace()
                res = fast_pnp(pts3d[img_idx], im_focals[img_idx], msk=msk, device=device, niter_PnP=niter_PnP)
                print("Run img_idx: ", img_idx, "Focal length by Ransac PnP: ", im_focals[img_idx])
                # res = fast_pnp(pts3d[img_idx], im_focals[group[0]], msk=msk, device=device, niter_PnP=niter_PnP) # here we init the focal length using the first frame of each group
                # res = fast_pnp(pts3d[img_idx], im_focals[first_group[0]], msk=msk, device=device, niter_PnP=niter_PnP) # here we init the focal length using the first frame
                if res:
                    focals, pose = res
                    if im_poses[img_idx] is None:
                        im_poses[img_idx] = pose
                    if im_focals[img_idx] is None:
                        im_focals[img_idx] = focals
                        print("Set img_idx: ", img_idx, "Focal length by Ransac PnP: ", im_focals[img_idx])
                    # im_focals[img_idx], im_poses[img_idx] = res
                if im_poses[img_idx] is None:
                    print(f"Failed to initialize camera pose for image {img_idx}")
                    im_poses[img_idx] = torch.eye(4, device=device)


            # if im_poses[img_idx] is None:
        # pts3d[j] = geotrf(trf, pred_j[i_j])
        # msp_edges.append((i, j))

        # if has_im_poses and im_poses[i] is None:
        #     im_poses[i] = sRT_to_4x4(1, R, T, device)
        # TODO don't know how to set the camera poses

    im_poses = torch.stack(im_poses)    
    return pts3d, im_focals, im_poses, conf_list



def align_group_prefix(imshapes, groups, pred_dict, conf_dict,
                          device, has_im_poses=True, niter_PnP=100, verbose=True, save_score_path=None, raymap_dict=None, fast_focal=True):
    # stride = groups[1][0] - groups[0][0]
    n_imgs = len(imshapes)
    # eadge_and_scores = compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j)
    # sparse_graph = -dict_to_sparse_graph(eadge_and_scores)
    # msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # temp variable to store 3d points
    pts3d = [None] * len(imshapes)
    conf_list = [None] * len(imshapes)

    # todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs
    # done = {}
    done = set()

    if fast_focal:
        ref_pointmap = []
        ref_conf = []
        ref_idx = []
        for i, group in enumerate(groups):
            ref_pointmap.append(pred_dict[i][0])
            ref_conf.append(conf_dict[i][0])
            ref_idx.append(group[0])
        # import pdb;pdb.set_trace()
        ref_pointmap = torch.stack(ref_pointmap, dim=0)
        ref_conf = torch.stack(ref_conf, dim=0)
        B, H, W, _ = ref_pointmap.shape
        mask = ref_conf > 0.5
        try:
            # import pdb;pdb.set_trace()
            ref_pointmap = ref_pointmap.clone()
            ref_pointmap[...,2] = ref_pointmap[...,2] - ref_pointmap[...,2].min() + 1
            depth, fov_x, fov_y, optim_shift = point_map_to_depth(ref_pointmap, mask, downsample_size=(H,W))
            intrinsics = intrinsics_from_fov_xy(fov_x, fov_y)
            focal_group = ((intrinsics[:,0,0] * W) + (intrinsics[:, 1,1] * H)) / 2
            mean_focal_group = focal_group[focal_group>30].mean()
            # print(f"Focal length for all the refence frames are {focal_group}")
            # print(f"reference frame indices are {ref_idx}")
            # detect the outlier group
            relative_error = torch.abs(focal_group - mean_focal_group) / mean_focal_group
            focal_group[relative_error > 0.6] = mean_focal_group
            # print(f"after filter outlier group Focal length for all the refence frames are {focal_group}")
            focal_group = focal_group.cpu().numpy().tolist()
        except:
            print("Error in computing focal length")
            msk = conf_dict[0][0] > 0.5
            res = fast_pnp(pred_dict[0][0].clone(), None, msk=msk, device=device, niter_PnP=niter_PnP)
            focals, _ = res
            focal_group = [focals] * len(groups)

        # TODO we can also filter out the outlier group based on the (focal_i - focal_mean)
        # for i, idx in enumerate(ref_idx):
        #     im_focals[idx] = focal_group[i]
        # for i in range(n_imgs):
        #     if im_focals[i] is None:
        #         im_focals[i] = im_focals[i-1]
        

    # set the first frame to be the world coordinate
    first_group = groups[0]
    for group_idx, img_idx in enumerate(first_group):
        focal_group_i = focal_group[0]
        if group_idx == 0:
            im_focals[img_idx] = focal_group_i
        pts3d[img_idx] = pred_dict[0][group_idx].clone()
        conf_list[img_idx] = conf_dict[0][group_idx].clone()
        if raymap_dict is not None:
            im_focals[img_idx] = estimate_focal(raymap_dict[0][group_idx])
            print(f"Estimated focal length for image {img_idx} is {im_focals[img_idx]}")
            # im_focals[img_idx] = estimate_focal(pred_dict[0][group_idx])
        # pts3d[i] = pred_i[i_j].clone()
        H, W, _ = pred_dict[0][group_idx].shape
        
        if conf_dict is None:
            msk = torch.ones((H, W), dtype=torch.bool, device=device)
        else:
            msk = conf_dict[0][group_idx] > 0.5
            # import pdb;pdb.set_trace()
        if img_idx != 0:
            temp_focal = im_focals[img_idx-1]
        else:
            temp_focal = im_focals[img_idx]
        res = fast_pnp(pts3d[img_idx], temp_focal, msk=msk, device=device, niter_PnP=niter_PnP)
        # res = fast_pnp(pts3d[img_idx], im_focals[first_group[0]], msk=msk, device=device, niter_PnP=niter_PnP) # here we init the focal length using the first frame of the first group
        if res:
            im_focals[img_idx], im_poses[img_idx] = res
            # print("img_idx: ", img_idx, "Focal length by Ransac PnP: ", im_focals[img_idx])
        if im_poses[img_idx] is None:
            print(f"Failed to initialize camera pose for image {img_idx}")
            im_poses[img_idx] = torch.eye(4, device=device)
        done.add(img_idx)
    other_groups = groups[1:]
    for i, group in enumerate(groups):
        focal_group_i = focal_group[i]
        if i==0:
            continue
        assert group[0] in done, "The first image of the following group should be in the previous group"
        # TODO we need to use multiple overlapping groups to initialize the camera poses
        seen_imgs_group_idx = []
        seen_imgs_img_idx = []
        pred_temp_list = []
        pts3d_temp_list = []
        conf_temp_list = []
        for group_idx, img_idx in enumerate(group):
            if img_idx in done:
                seen_imgs_group_idx.append(group_idx)
                seen_imgs_img_idx.append(img_idx)
                pred_temp_list.append(pred_dict[i][group_idx])
                pts3d_temp_list.append(pts3d[img_idx])
                conf_temp_list.append(conf_dict[i][group_idx] * conf_list[img_idx])
        pred_temp_list = torch.stack(pred_temp_list)
        pts3d_temp_list = torch.stack(pts3d_temp_list)
        conf_temp_list = torch.stack(conf_temp_list)

        s, R, T = rigid_points_registration(pred_temp_list, pts3d_temp_list, conf=conf_temp_list) 
        # s, R, T = rigid_points_registration(pred_dict[i][seen_imgs_group_idx], pts3d[seen_imgs_img_idx], conf=conf_dict[i][seen_imgs_group_idx])
        # TODO, double check the conf_dict here
        trf = sRT_to_4x4(s, R, T, device)
        for group_idx, img_idx in enumerate(group):
            if pts3d[img_idx] is None:
                pts3d[img_idx] = geotrf(trf, pred_dict[i][group_idx])
                conf_list[img_idx] = conf_dict[i][group_idx]
                done.add(img_idx)
            else: # even if we already init the 3d point, we overwrite it with the new one, we assume the closer to the 0 frame, the accurate pointmap we have
                pts3d[img_idx] = geotrf(trf, pred_dict[i][group_idx])
                conf_list[img_idx] = conf_dict[i][group_idx]
                done.add(img_idx)
            if group_idx==0 and has_im_poses and im_poses[img_idx] is None:
                im_poses[img_idx] = sRT_to_4x4(1, R, T, device)
                # TODO, if we know the pose, we can estimate the focal length and shift z
            # if group_idx==0 and has_im_poses and im_focals[img_idx] is None:
            #     im_focals[img_idx] = estimate_focal(pred_dict[i][group_idx])
            if raymap_dict is not None and im_focals[img_idx] is None:
                im_focals[img_idx] = estimate_focal(raymap_dict[i][group_idx])
                print(f"Estimated focal length for image {img_idx} is {im_focals[img_idx]}")
        
            if has_im_poses:

                if conf_dict is None:
                    msk = torch.ones((H, W), dtype=torch.bool, device=device)
                else:
                    msk = conf_dict[i][group_idx] > 0.5
                    # import pdb;pdb.set_trace()
                if group_idx == 0:
                    temp_focal = focal_group_i
                else:
                    temp_focal = im_focals[img_idx - 1]
                res = fast_pnp(pts3d[img_idx], temp_focal, msk=msk, device=device, niter_PnP=niter_PnP)
                # res = fast_pnp(pts3d[img_idx], im_focals[img_idx], msk=msk, device=device, niter_PnP=niter_PnP)
                # print("Run img_idx: ", img_idx, "Focal length by Ransac PnP: ", im_focals[img_idx])
                # res = fast_pnp(pts3d[img_idx], im_focals[group[0]], msk=msk, device=device, niter_PnP=niter_PnP) # here we init the focal length using the first frame of each group
                # res = fast_pnp(pts3d[img_idx], im_focals[first_group[0]], msk=msk, device=device, niter_PnP=niter_PnP) # here we init the focal length using the first frame
                if res:
                    focals, pose = res
                    # if im_poses[img_idx] is None:
                    #     im_poses[img_idx] = pose
                    # if im_focals[img_idx] is None:
                    #     im_focals[img_idx] = focals
                    im_poses[img_idx] = pose
                    im_focals[img_idx] = focals
                    # print("Set img_idx: ", img_idx, "Focal length by Ransac PnP: ", im_focals[img_idx])
                    # im_focals[img_idx], im_poses[img_idx] = res
                if im_poses[img_idx] is None:
                    print(f"Failed to initialize camera pose for image {img_idx}")
                    im_poses[img_idx] = torch.eye(4, device=device)


            # if im_poses[img_idx] is None:
        # pts3d[j] = geotrf(trf, pred_j[i_j])
        # msp_edges.append((i, j))

        # if has_im_poses and im_poses[i] is None:
        #     im_poses[i] = sRT_to_4x4(1, R, T, device)
        # TODO don't know how to set the camera poses

    im_poses = torch.stack(im_poses)    
    return pts3d, im_focals, im_poses, conf_list




def draw_edge_scores_map(edge_scores, save_path, n_imgs=None):
    # Determine the size of the heatmap
    if n_imgs is None:
        n_imgs = max(max(edge) for edge in edge_scores) + 1

    # Create a matrix to hold the scores
    heatmap_matrix = np.full((n_imgs, n_imgs), np.nan)

    # Populate the matrix with the edge scores
    for (i, j), score in edge_scores.items():
        heatmap_matrix[i, j] = score

    # Plotting the heatmap
    plt.figure(figsize=(int(5.5*np.log(n_imgs)-2), int((5.5*np.log(n_imgs)-2) * 3 / 4)))
    sns.heatmap(heatmap_matrix, annot=True, fmt=".1f", cmap="viridis", cbar=True, annot_kws={"fontsize": int(-4.2*np.log(n_imgs)+22.4)})
    plt.title("Heatmap of Edge Scores")
    plt.xlabel("Node")
    plt.ylabel("Node")
    plt.savefig(save_path)

@torch.no_grad()
def init_from_known_poses(self, niter_PnP=10, min_conf_thr=3):
    device = self.device

    # indices of known poses
    nkp, known_poses_msk, known_poses = get_known_poses(self)
    # assert nkp == self.n_imgs, 'not all poses are known'

    # get all focals
    nkf, _, im_focals = get_known_focals(self)
    # assert nkf == self.n_imgs
    im_pp = self.get_principal_points()

    best_depthmaps = {}
    # init all pairwise poses
    for e, (i, j) in enumerate(tqdm(self.edges, disable=not self.verbose)):
        i_j = edge_str(i, j)

        # find relative pose for this pair
        P1 = torch.eye(4, device=device)
        msk = self.conf_i[i_j] > min(min_conf_thr, self.conf_i[i_j].min() - 0.1)
        _, P2 = fast_pnp(self.pred_j[i_j], float(im_focals[i].mean()),
                         pp=im_pp[i], msk=msk, device=device, niter_PnP=niter_PnP)

        # align the two predicted camera with the two gt cameras
        s, R, T = align_multiple_poses(torch.stack((P1, P2)), known_poses[[i, j]])
        # normally we have known_poses[i] ~= sRT_to_4x4(s,R,T,device) @ P1
        # and geotrf(sRT_to_4x4(1,R,T,device), s*P2[:3,3])
        self._set_pose(self.pw_poses, e, R, T, scale=s)

        # remember if this is a good depthmap
        score = float(self.conf_i[i_j].mean())
        if score > best_depthmaps.get(i, (0,))[0]:
            best_depthmaps[i] = score, i_j, s

    # init all image poses
    for n in range(self.n_imgs):
        # assert known_poses_msk[n]
        if n in best_depthmaps:
            _, i_j, scale = best_depthmaps[n]
            depth = self.pred_i[i_j][:, :, 2]
            self._set_depthmap(n, depth * scale)


@torch.no_grad()
def init_minimum_spanning_tree(self, save_score_path=None, save_score_only=False, **kw):
    """ Init all camera poses (image-wise and pairwise poses) given
        an initial set of pairwise estimations.
    """
    device = self.device
    if save_score_only:
        eadge_and_scores = compute_edge_scores(map(i_j_ij, self.edges), self.conf_i, self.conf_j)
        draw_edge_scores_map(eadge_and_scores, save_score_path)
        return
    pts3d, _, im_focals, im_poses = minimum_spanning_tree(self.imshapes, self.edges,
                                                          self.pred_i, self.pred_j, self.conf_i, self.conf_j, self.im_conf, self.min_conf_thr,
                                                          device, has_im_poses=self.has_im_poses, verbose=self.verbose, save_score_path=save_score_path,
                                                           **kw)

    return init_from_pts3d(self, pts3d, im_focals, im_poses)

@torch.no_grad()
def init_from_fast_pnp(self, save_score_path=None, save_score_only=False, **kw):
    """ Init all camera poses (image-wise and pairwise poses) given
        an initial set of pairwise estimations.
    """
    device = self.device
    # if save_score_only:
    #     eadge_and_scores = compute_edge_scores(map(i_j_ij, self.edges), self.conf_i, self.conf_j)
    #     draw_edge_scores_map(eadge_and_scores, save_score_path)
    #     return
    pts3d, im_focals, im_poses = fast_pnp_all_image(self.imshapes, self.pred_i, device, has_im_poses=self.has_im_poses, verbose=self.verbose, save_score_path=save_score_path, conf = self.init_conf_maps, **kw)
    
    # (imshapes, pred_i, device, has_im_poses=True, niter_PnP=10, verbose=True, save_score_path=None)

    return init_from_pts3d_wo_pairwise(self, pts3d, im_focals, im_poses, self.init_conf_maps)


def init_from_pts3d_wo_pairwise(self, pts3d, im_focals, im_poses, conf):
    # init poses
    nkp, known_poses_msk, known_poses = get_known_poses(self)
    if nkp == 1:
        raise NotImplementedError("Would be simpler to just align everything afterwards on the single known pose")
    elif nkp > 1:
        # global rigid SE3 alignment
        s, R, T = align_multiple_poses(im_poses[known_poses_msk], known_poses[known_poses_msk])
        trf = sRT_to_4x4(s, R, T, device=known_poses.device)

        # rotate everything
        im_poses = trf @ im_poses
        im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
        for img_pts3d in pts3d:
            img_pts3d[:] = geotrf(trf, img_pts3d)
    else: pass # no known poses

    # # set all pairwise poses
    # for e, (i, j) in enumerate(self.edges):
    #     i_j = edge_str(i, j)
    #     # compute transform that goes from cam to world
    #     s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])
    #     self._set_pose(self.pw_poses, e, R, T, scale=s)

    # TODO, scale should be considered when you run multiple times, to use rigid_points_registration to calculate the scale! 
    # # take into account the scale normalization
    # s_factor = self.get_pw_norm_scale_factor()
    # im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    # for img_pts3d in pts3d:
    #     img_pts3d *= s_factor

    # init all image poses
    sky_distance = 0
    if self.has_im_poses:
        for i in range(self.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            # import pdb;pdb.set_trace()
            sky_mask = conf[i] < 1e-4
            if i==0:
                depth[sky_mask] = depth.max()
                sky_distance = depth.max()
            else:
                depth[sky_mask] = sky_distance
            self._set_depthmap(i, depth)
            self._set_pose(self.im_poses, i, cam2world)
            if im_focals[i] is not None:
                if not self.shared_focal:
                    self._set_focal(i, im_focals[i])
        if self.shared_focal:
            self._set_focal(0, sum(im_focals) / self.n_imgs)
        if self.n_imgs > 2:
            self._set_init_depthmap()
        

    if self.verbose:
        with torch.no_grad():
            print(' init loss =', float(self()))



def init_from_pts3d_group(self, pts3d, im_focals, im_poses, conf_list):
    # init poses
    nkp, known_poses_msk, known_poses = get_known_poses(self)
    if nkp == 1:
        raise NotImplementedError("Would be simpler to just align everything afterwards on the single known pose")
    elif nkp > 1:
        # global rigid SE3 alignment
        s, R, T = align_multiple_poses(im_poses[known_poses_msk], known_poses[known_poses_msk])
        trf = sRT_to_4x4(s, R, T, device=known_poses.device)

        # rotate everything
        im_poses = trf @ im_poses
        im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
        for img_pts3d in pts3d:
            img_pts3d[:] = geotrf(trf, img_pts3d)
    else: pass # no known poses

    # set all pairwise poses
    for e, group in enumerate(self.groups):
        # compute transform that goes from cam to world
        pred_temp_list = []
        pts3d_temp_list = []
        conf_temp_list = []
        for i, g in enumerate(group):
            pred_temp_list.append(self.pred_dict[e][i])
            pts3d_temp_list.append(pts3d[g])
            conf_temp_list.append(self.conf_dict[e][i] * conf_list[g])
        pred_temp_list = torch.stack(pred_temp_list)
        pts3d_temp_list = torch.stack(pts3d_temp_list)
        conf_temp_list = torch.stack(conf_temp_list)
        s, R, T = rigid_points_registration(pred_temp_list, pts3d_temp_list, conf=conf_temp_list)
        # s, R, T = rigid_points_registration(self.pred_dict[e], pts3d[group], conf=self.conf_dict[e])
        self._set_pose(self.pw_poses, e, R, T, scale=s)

    # take into account the scale normalization
    s_factor = self.get_pw_norm_scale_factor()
    im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    for img_pts3d in pts3d:
        img_pts3d *= s_factor

    sky_distance = 0
    # init all image poses
    if self.has_im_poses:
        for i in range(self.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            sky_mask = conf_list[i] < 1e-4
            if i==0:
                depth[sky_mask] = depth.max()
                sky_distance = depth.max()
            else:
                depth[sky_mask] = sky_distance
            self._set_depthmap(i, depth)
            self._set_pose(self.im_poses, i, cam2world)
            if im_focals[i] is not None:
                if not self.shared_focal:
                    self._set_focal(i, im_focals[i])
        if self.shared_focal:
            self._set_focal(0, sum(im_focals) / self.n_imgs)
        if self.n_imgs > 2:
            self._set_init_depthmap()

    if self.verbose:
        with torch.no_grad():
            print(' init loss =', float(self()))




def init_from_pts3d(self, pts3d, im_focals, im_poses):
    # init poses
    nkp, known_poses_msk, known_poses = get_known_poses(self)
    if nkp == 1:
        raise NotImplementedError("Would be simpler to just align everything afterwards on the single known pose")
    elif nkp > 1:
        # global rigid SE3 alignment
        s, R, T = align_multiple_poses(im_poses[known_poses_msk], known_poses[known_poses_msk])
        trf = sRT_to_4x4(s, R, T, device=known_poses.device)

        # rotate everything
        im_poses = trf @ im_poses
        im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
        for img_pts3d in pts3d:
            img_pts3d[:] = geotrf(trf, img_pts3d)
    else: pass # no known poses

    # set all pairwise poses
    for e, (i, j) in enumerate(self.edges):
        i_j = edge_str(i, j)
        # compute transform that goes from cam to world
        s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])
        self._set_pose(self.pw_poses, e, R, T, scale=s)

    # take into account the scale normalization
    s_factor = self.get_pw_norm_scale_factor()
    im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    for img_pts3d in pts3d:
        img_pts3d *= s_factor

    # init all image poses
    if self.has_im_poses:
        for i in range(self.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            self._set_depthmap(i, depth)
            self._set_pose(self.im_poses, i, cam2world)
            if im_focals[i] is not None:
                if not self.shared_focal:
                    self._set_focal(i, im_focals[i])
        if self.shared_focal:
            self._set_focal(0, sum(im_focals) / self.n_imgs)
        if self.n_imgs > 2:
            self._set_init_depthmap()

    if self.verbose:
        with torch.no_grad():
            print(' init loss =', float(self()))


def minimum_spanning_tree(imshapes, edges, pred_i, pred_j, conf_i, conf_j, im_conf, min_conf_thr,
                          device, has_im_poses=True, niter_PnP=10, verbose=True, save_score_path=None):
    n_imgs = len(imshapes)
    eadge_and_scores = compute_edge_scores(map(i_j_ij, edges), conf_i, conf_j)
    sparse_graph = -dict_to_sparse_graph(eadge_and_scores)
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # temp variable to store 3d points
    pts3d = [None] * len(imshapes)

    todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs

    # init with strongest edge
    score, i, j = todo.pop()
    if verbose:
        print(f' init edge ({i}*,{j}*) {score=}')
    if save_score_path is not None:
        draw_edge_scores_map(eadge_and_scores, save_score_path, n_imgs=n_imgs)
        save_tree_path = save_score_path.replace(".png", "_tree.txt")
        with open(save_tree_path, "w") as f:
            f.write(f'init edge ({i}*,{j}*) {score=}\n')
    i_j = edge_str(i, j)
    pts3d[i] = pred_i[i_j].clone() # the first one is set to be world coordinate
    pts3d[j] = pred_j[i_j].clone()
    done = {i, j}
    if has_im_poses:
        im_poses[i] = torch.eye(4, device=device)
        im_focals[i] = estimate_focal(pred_i[i_j])

    # set initial pointcloud based on pairwise graph
    msp_edges = [(i, j)]
    while todo:
        # each time, predict the next one
        score, i, j = todo.pop()

        if im_focals[i] is None:
            im_focals[i] = estimate_focal(pred_i[i_j])

        if i in done:   # the first frame is already set, align the second frame with the first frame
            if verbose:
                print(f' init edge ({i},{j}*) {score=}')
            if save_score_path is not None:
                with open(save_tree_path, "a") as f:
                    f.write(f'init edge ({i},{j}*) {score=}\n')
            assert j not in done
            # align pred[i] with pts3d[i], and then set j accordingly
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(pred_i[i_j], pts3d[i], conf=conf_i[i_j])
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[j] = geotrf(trf, pred_j[i_j])
            done.add(j)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)

        elif j in done:  # the second frame is already set, align the first frame with the second frame
            if verbose:
                print(f' init edge ({i}*,{j}) {score=}')
            if save_score_path is not None:
                with open(save_tree_path, "a") as f:
                    f.write(f'init edge ({i}*,{j}) {score=}\n')
            assert i not in done
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(pred_j[i_j], pts3d[j], conf=conf_j[i_j])
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[i] = geotrf(trf, pred_i[i_j])
            done.add(i)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)
        else:
            # let's try again later
            todo.insert(0, (score, i, j))

    if has_im_poses:
        # complete all missing informations
        pair_scores = list(sparse_graph.values())  # already negative scores: less is best
        edges_from_best_to_worse = np.array(list(sparse_graph.keys()))[np.argsort(pair_scores)]
        for i, j in edges_from_best_to_worse.tolist():
            if im_focals[i] is None:
                im_focals[i] = estimate_focal(pred_i[edge_str(i, j)])

        for i in range(n_imgs):
            if im_poses[i] is None:
                msk = im_conf[i] > min_conf_thr
                res = fast_pnp(pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP)
                if res:
                    im_focals[i], im_poses[i] = res
            if im_poses[i] is None:
                im_poses[i] = torch.eye(4, device=device)
        im_poses = torch.stack(im_poses)
    else:
        im_poses = im_focals = None

    return pts3d, msp_edges, im_focals, im_poses


def dict_to_sparse_graph(dic):
    n_imgs = max(max(e) for e in dic) + 1
    res = sp.dok_array((n_imgs, n_imgs))
    for edge, value in dic.items():
        res[edge] = value
    return res


def rigid_points_registration(pts1, pts2, conf):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf.ravel(), compute_scaling=True)
    return s, R, T  # return un-scaled (R, T)


def sRT_to_4x4(scale, R, T, device):
    trf = torch.eye(4, device=device)
    trf[:3, :3] = R * scale
    trf[:3, 3] = T.ravel()  # doesn't need scaling
    return trf


def estimate_focal(pts3d_i, pp=None):
    if pp is None:
        H, W, THREE = pts3d_i.shape
        assert THREE == 3
        pp = torch.tensor((W/2, H/2), device=pts3d_i.device)
    focal = estimate_focal_knowing_depth(pts3d_i.unsqueeze(0), pp.unsqueeze(0), focal_mode='weiszfeld').ravel()
    return float(focal)


@lru_cache(maxsize=None)
def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)


def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10):
    # extract camera poses and focals with RANSAC-PnP
    if msk.sum() < 4:
        return None  # we need at least 4 points for PnP
    pts3d, msk = map(to_numpy, (pts3d, msk))

    H, W, THREE = pts3d.shape
    assert THREE == 3
    pixels = pixel_grid(H, W)

    if focal is None:
        S = max(W, H)
        tentative_focals = np.geomspace(S/2, S*3, 63) # from 63 -> 256
    else:
        S = max(W, H)
        tentative_focals = [focal] + list(np.geomspace(-0.03 * S + focal, 0.03 * S + focal, 2))

    if pp is None:
        pp = (W/2, H/2)
    else:
        pp = to_numpy(pp)

    best = 0,
    for focal in tentative_focals:
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

        success, R, T, inliers = cv2.solvePnPRansac(pts3d[msk], pixels[msk], K, None,
                                                    iterationsCount=niter_PnP, reprojectionError=5, flags=cv2.SOLVEPNP_SQPNP)
        if not success:
            continue

        score = len(inliers)
        if success and score > best[0]:
            best = score, R, T, focal

    if not best[0]:
        return None

    _, R, T, best_focal = best
    R = cv2.Rodrigues(R)[0]  # world to cam
    R, T = map(torch.from_numpy, (R, T))
    return best_focal, inv(sRT_to_4x4(1, R, T, device))  # cam to world


def get_known_poses(self):
    if self.has_im_poses:
        known_poses_msk = torch.tensor([not (p.requires_grad) for p in self.im_poses])
        known_poses = self.get_im_poses()
        return known_poses_msk.sum(), known_poses_msk, known_poses
    else:
        return 0, None, None


def get_known_focals(self):
    if self.has_im_poses:
        known_focal_msk = self.get_known_focal_mask()
        known_focals = self.get_focals()
        return known_focal_msk.sum(), known_focal_msk, known_focals
    else:
        return 0, None, None


def align_multiple_poses(src_poses, target_poses):
    N = len(src_poses)
    assert src_poses.shape == target_poses.shape == (N, 4, 4)

    def center_and_z(poses):
        eps = get_med_dist_between_poses(poses) / 100
        return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps*poses[:, :3, 2]))
    R, T, s = roma.rigid_points_registration(center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True)
    return s, R, T
