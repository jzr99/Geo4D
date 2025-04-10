import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import contextlib

from dust3r.cloud_opt.base_opt_group import LightBaseGroupPCOptimizer
from dust3r.utils.geometry import xy_grid, geotrf, depthmap_to_pts3d
from dust3r.utils.device import to_cpu, to_numpy
from dust3r.utils.goem_opt import DepthBasedWarping, OccMask, WarpImage, depth_regularization_si_weighted, tum_to_pose_matrix
from dust3r.depth_eval import depth_evaluation
import cv2
import os

from dust3r.cloud_opt.base_opt_group import get_tum_poses
from dust3r.utils.vo_eval import align_trajectory_with_eval

def smooth_L1_loss_fn(estimate, gt, mask, beta=1.0, per_pixel_thre=50.):
    loss_raw_shape = F.smooth_l1_loss(estimate*mask, gt*mask, beta=beta, reduction='none')
    if per_pixel_thre > 0:
        per_pixel_mask = (loss_raw_shape < per_pixel_thre) * mask
    else:
        per_pixel_mask = mask
    return torch.sum(loss_raw_shape * per_pixel_mask) / torch.sum(per_pixel_mask)

def mse_loss_fn(estimate, gt, mask):
    v = torch.sum((estimate*mask-gt*mask)**2) / torch.sum(mask)
    return v  # , v.item()

class LightPointCloudGroupOptimizer(LightBaseGroupPCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, optimize_pp=False, focal_break=20, shared_focal=False, flow_loss_fn='smooth_l1', flow_loss_weight=0.0, 
                 depth_regularize_weight=0.0, num_total_iter=300, temporal_smoothing_weight=0, translation_weight=0.1, flow_loss_start_epoch=0.15, flow_loss_thre=50,
                 sintel_ckpt=False, use_self_mask=False, pxl_thre=50, sam2_mask_refine=True, motion_mask_thre=0.35, conf_optimize=False, depth_traj_start_iter=150, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break
        self.num_total_iter = num_total_iter
        self.temporal_smoothing_weight = temporal_smoothing_weight
        self.translation_weight = translation_weight
        self.flow_loss_flag = False
        self.flow_loss_start_epoch = flow_loss_start_epoch
        self.flow_loss_thre = flow_loss_thre
        self.optimize_pp = optimize_pp
        self.pxl_thre = pxl_thre
        self.motion_mask_thre = motion_mask_thre
        self.conf_optimize = conf_optimize
        self.depth_traj_start_iter = depth_traj_start_iter

        # self.imshapes is [(H, W) * N]
        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        self.shared_focal = shared_focal
        if self.shared_focal:
            self.im_focals = nn.ParameterList(torch.FloatTensor(
                [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes[:1])  # camera intrinsics
        else:
            self.im_focals = nn.ParameterList(torch.FloatTensor(
                [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes)  # camera intrinsics
        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area) #(num_imgs, H*W)

        self.im_poses = ParameterStack(self.im_poses, is_param=True)
        self.im_focals = ParameterStack(self.im_focals, is_param=True)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
        self.register_buffer('_grid', ParameterStack(
            [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))


        self.valid_traj_group_list = []
        self.invalid_depth_group = []
        self.valid_group_idx = []
        self.register_buffer('_weight_all', ParameterStack(
            [self.conf_trf(self.conf_dict[i][idx]) for i, group in enumerate(self.groups) for idx, group_idx in enumerate(group)], fill=self.max_area))

        self.register_buffer('_stacked_pred_all', ParameterStack([self.pred_dict[i][idx] for i, group in enumerate(self.groups) for idx, group_idx in enumerate(group)], fill=self.max_area))
        
        if self.opt_raydir:
            self.register_buffer('_stacked_raydir_all', ParameterStack([self.raydir_dict[i][idx] for i, group in enumerate(self.groups) for idx, group_idx in enumerate(group)], fill=self.max_area))
        
        if self.inverse_depthmap_dict is not None:
            self.register_buffer('_stacked_depthmap_all', ParameterStack([self.inverse_depthmap_dict[i][idx] for i, group in enumerate(self.groups) for idx, group_idx in enumerate(group)], fill=self.max_area))
        
        if self.crossmap_dict is not None:
            self.register_buffer('_stacked_crossmap_all', ParameterStack([self.crossmap_dict[i][idx] for i, group in enumerate(self.groups) for idx, group_idx in enumerate(group)], fill=self.max_area))
        
        if self.traj_dict is not None:
            self.register_buffer('_stacked_traj_all', ParameterStack([self.traj_dict[i][idx] for i, group in enumerate(self.groups) for idx, group_idx in enumerate(group)], fill=0))

        self.register_buffer('_e_all', torch.tensor([j for group in self.groups for j in group]))

        self.total_area_all = sum([im_areas[j] for group in self.groups for j in group])


    def _check_all_imgs_are_selected(self, msk):
        self.msk = torch.from_numpy(np.array(msk, dtype=bool)).to(self.device)
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'
        pass

    def preset_pose(self, known_poses, pose_msk=None, requires_grad=False):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        if known_poses.shape[-1] == 7: # xyz wxyz
            known_poses = [tum_to_pose_matrix(pose) for pose in known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)
        if len(known_poses) == self.n_imgs:
            if requires_grad:
                self.im_poses.requires_grad_(True)
            else:
                self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_intrinsics(self, known_intrinsics, msk=None):
        if isinstance(known_intrinsics, torch.Tensor) and known_intrinsics.ndim == 2:
            known_intrinsics = [known_intrinsics]
        for K in known_intrinsics:
            assert K.shape == (3, 3)
        self.preset_focal([K.diagonal()[:2].mean() for K in known_intrinsics], msk)
        if self.optimize_pp:
            self.preset_principal_point([K[:2, 2] for K in known_intrinsics], msk)

    def preset_focal(self, known_focals, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))
        if len(known_focals) == self.n_imgs:
            if requires_grad:
                self.im_focals.requires_grad_(True)
            else:
                self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = self.focal_break * np.log(focal)
        return param

    def get_focals(self):
        if self.shared_focal:
            log_focals = torch.stack([self.im_focals[0]] * self.n_imgs, dim=0)
        else:
            log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K
    
    def get_intrinsics_dev(self, dev=8):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = (focals/dev)
        K[:, :2, 2] = (self.get_principal_points() / dev)
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param
    
    
    def _set_traj(self):
        print("start aligning trajectory")
        im_pose = self.get_im_poses()
        pw_scale = self.get_pw_scale()
        new_traj_list = []
        valid_group = []
        valid_group_idx = []
        aligned_traj_list = []
        for i in range(self.n_groups):
            pw_scale_i = pw_scale[i]
            group = self.groups[i]
            im_pose_group = im_pose[group]
            traj = self.traj_dict[i].detach().clone()
            traj[:,:3,3] = traj[:,:3,3] * pw_scale_i
            traj_tum = get_tum_poses(traj)
            im_pose_tum = get_tum_poses(im_pose_group)
            ate, rpe_trans, rpe_rot, P, aligned_traj = align_trajectory_with_eval(traj_tum, im_pose_tum, return_aligned_traj=True, correct_scale=False, align_origin=True)
            aligned_traj_list.append(aligned_traj)
            self._set_pose(self.traj_align_poses, i, R=torch.from_numpy(P).to(im_pose.device), scale=torch.from_numpy(np.array(pw_scale_i.detach().cpu().numpy())).to(im_pose.device), scale_T=False)
            # print(f"Traj align ATE: {ate:.5f}, RPE trans: {rpe_trans:.5f}, RPE rot: {rpe_rot:.5f}")
            if rpe_rot < 4:
                valid_group.append(i)
                valid_group_idx = valid_group_idx + group
            
        
        return valid_group, valid_group_idx, aligned_traj_list
    

    def get_align_diffuse_traj_tum_poses(self, set_pose=False):
        global_traj = torch.zeros((self.n_imgs, 4, 4))
        first_group = self.groups[0]
        global_traj[first_group] = self.traj_dict[0].detach().cpu()
        for i in range(self.n_groups):
            if i==0:
                continue
            group = self.groups[i]
            traj = self.traj_dict[i].detach().clone().cpu()
            traj_tum = get_tum_poses(traj)
            im_pose_tum = get_tum_poses(global_traj[group])
            ate, rpe_trans, rpe_rot, P, aligned_traj = align_trajectory_with_eval(traj_tum, im_pose_tum, return_aligned_traj=True, correct_scale=False, align_origin=True)
            for group_idx, global_idx in enumerate(group):
                if global_traj[global_idx][3,3]==1:
                    pass
                else:
                    global_traj[global_idx] = torch.from_numpy(P).float() @ traj[group_idx]
        
        if set_pose:
            for i in range(self.n_imgs):
                self._set_pose(self.im_poses, i, R=global_traj[i], force=True)
            self._set_focal(0, 400.0, force=True)

        return get_tum_poses(global_traj)


    def get_align_inverse_depthmap(self, set_depth=False):
        # import pdb; pdb.set_trace()
        global_inverse_depthmap = torch.zeros((self.n_imgs, *self.inverse_depthmap_dict[0][0].shape))
        T, H, W, _ = global_inverse_depthmap.shape
        aligned_idx = []
        first_group = self.groups[0]
        global_inverse_depthmap[first_group] = self.inverse_depthmap_dict[0].detach().clone().cpu()
        aligned_idx = aligned_idx + first_group
        for i in range(self.n_groups):
            if i==0:
                continue
            group = self.groups[i]
            inverse_depthmap = self.inverse_depthmap_dict[i].detach().clone().cpu()
            gt_inverse_depthmap = global_inverse_depthmap[group]
            custom_mask = torch.zeros_like(inverse_depthmap, dtype=torch.bool)
            for group_idx, global_idx in enumerate(group):
                if global_idx in aligned_idx:
                    custom_mask[group_idx] = True
            # import pdb;pdb.set_trace()
            depth_results, error_map, depth_predict, depth_gt  = depth_evaluation(inverse_depthmap.to(self.device).reshape(-1), gt_inverse_depthmap.to(self.device).reshape(-1), max_depth=None,  align_with_lad2=True, use_gpu=True, lr=1e-2, max_iters=5000, custom_mask=custom_mask.to(self.device).reshape(-1), return_st=True)
            aligned_result = inverse_depthmap * depth_results['s'] + depth_results['t']
            for group_idx, global_idx in enumerate(group):
                if global_idx in aligned_idx:
                    pass
                else:
                    global_inverse_depthmap[global_idx] = aligned_result[group_idx]
                    aligned_idx.append(global_idx)
        if set_depth:
            for i in range(self.n_imgs):
                real_depth = 1.0 / (global_inverse_depthmap[i,...,0] + 1e-6)
                # clip the depth between 0 to 70
                real_depth = torch.clamp(real_depth, 0, 70)
                self._set_depthmap(i, real_depth, force=True)
        return global_inverse_depthmap

    
    
    def _set_st_depth(self):
        print("start aligning depth")
        depth = self.get_depthmaps(raw=True)
        invdepth = 1.0 / (depth + 1e-6)
        invdepth_group = invdepth[self._e_all].reshape(self.n_groups, self.group_size, -1, 1).reshape(self.n_groups, -1).clone().detach()
        reshaped_stacked_depth = self._stacked_depthmap_all.reshape(self.n_groups, -1).clone().detach()
        weight = self._weight_all.reshape(self.n_groups, -1).clone().detach()
        custom_mask = (weight > 0.5)
        non_neg_mask = (reshaped_stacked_depth > 0.05)
        custom_mask = torch.logical_and(custom_mask, non_neg_mask)
        invalid_depth_group = []
        # TODO need to check whether reshaped_stacked_depth (diffused depth) has the same mask to apply
        H = 256
        W = 512
        for i in tqdm(range(self.n_groups)):
            best_results = None
            depth_results, error_map, depth_predict, depth_gt  = depth_evaluation(reshaped_stacked_depth[i], invdepth_group[i], max_depth=None,  align_with_lad2=True, use_gpu=True, lr=1e-2, max_iters=5000, custom_mask=custom_mask[i], return_st=True)
            best_results = depth_results
            self.s_depth.data[i] = depth_results['s']
            self.t_depth.data[i] = depth_results['t']
            if best_results['δ < 1.25'] < 0.8:
                depth_results, error_map, depth_predict, depth_gt  = depth_evaluation(reshaped_stacked_depth[i], invdepth_group[i], max_depth=None, align_with_lad2=True, use_gpu=True, lr=1e-4, max_iters=3000, custom_mask=custom_mask[i], return_st=True)
                if depth_results['δ < 1.25'] > best_results['δ < 1.25']:
                    print("picked 1e-4")
                    best_results = depth_results
                    self.s_depth.data[i] = depth_results['s']
                    self.t_depth.data[i] = depth_results['t']
                
                depth_results, error_map, depth_predict, depth_gt  = depth_evaluation(reshaped_stacked_depth[i], invdepth_group[i], max_depth=None, align_with_lad2=True, use_gpu=True, lr=1e-3, max_iters=3000, custom_mask=custom_mask[i], return_st=True)
                if depth_results['δ < 1.25'] > best_results['δ < 1.25']:
                    print("picked 1e-3")
                    best_results = depth_results
                    self.s_depth.data[i] = depth_results['s']
                    self.t_depth.data[i] = depth_results['t']
            
            if best_results['δ < 1.25'] < 0.3:
                invalid_depth_group.append(i)
            # print(f'align disparity for group {i}', best_results)

        return invalid_depth_group

    def preset_depthmap(self, known_depthmaps, msk=None, requires_grad=False):
        self._check_all_imgs_are_selected(msk)

        for idx, depth in zip(self._get_msk_indices(msk), known_depthmaps):
            if self.verbose:
                print(f' (setting depthmap #{idx})')
            self._no_grad(self._set_depthmap(idx, depth))

        if len(known_depthmaps) == self.n_imgs:
            if requires_grad:
                self.im_depthmaps.requires_grad_(True)
            else:
                self.im_depthmaps.requires_grad_(False)
    
    def _set_init_depthmap(self):
        depth_maps = self.get_depthmaps(raw=True)
        self.init_depthmap = [dm.detach().clone() for dm in depth_maps]

    def get_init_depthmaps(self, raw=False):
        res = self.init_depthmap
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

        

        

    def depth_to_pts3d_partial(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps()

        # convert focal to (1,2,H,W) constant field
        def focal_ex(i): return focals[i][..., None, None].expand(1, *focals[i].shape, *self.imshapes[i])
        # get pointmaps in camera frame
        rel_ptmaps = [depthmap_to_pts3d(depth[i][None], focal_ex(i), pp=pp[i:i+1])[0] for i in range(im_poses.shape[0])]
        # project to world frame
        return [geotrf(pose, ptmap) for pose, ptmap in zip(im_poses, rel_ptmaps)]
    
    def get_pts3d(self, raw=False, **kwargs):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def forward(self, epoch=9999):
        pw_poses = self.get_pw_poses()  # cam-to-world 

        pw_adapt = self.get_adaptors().unsqueeze(1) # [G, 1, 3]
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        new_pw_poses = pw_poses.unsqueeze(1).repeat(1, self.group_size, 1, 1).reshape(-1, 4, 4)
        new_pw_adapt = pw_adapt.unsqueeze(1).repeat(1, self.group_size, 1, 1).reshape(-1, 1, 3)


        aligned_pred = geotrf(new_pw_poses, new_pw_adapt * self._stacked_pred_all)

        # compute the loss

        if self.conf_optimize:
            self._weight_all[self._weight_all>10] = 10
            li = self.dist(proj_pts3d[self._e_all], aligned_pred, weight=self._weight_all).sum() / self.total_area_all
        else:
            li = self.dist(proj_pts3d[self._e_all], aligned_pred, weight=torch.ones_like(proj_pts3d[self._e_all])[..., 0]).sum() / self.total_area_all

        ray_loss = 0
        depth_loss = 0
        loss_traj = 0
        
        if self.inverse_depthmap_dict is not None and epoch >= self.depth_traj_start_iter:
            # invalid_depth_group = []
            if epoch == self.depth_traj_start_iter:
                self.invalid_depth_group = self._set_st_depth()
                print("invalid_depth_group", self.invalid_depth_group)

            depth = self.get_depthmaps(raw=True) # [50, 262144]
            inverse_predict_depth = 1 / (depth + 1e-6)
            inverse_predict_depth = inverse_predict_depth.unsqueeze(-1)
            # import pdb;pdb.set_trace() # TODO here the weight should based on the confidence score of the depth map
            # s_depth shape (self.n_groups, 1)
            s = self.s_depth.unsqueeze(1).repeat(1, self.group_size,1).reshape(-1, 1, 1)
            t = self.t_depth.unsqueeze(1).repeat(1, self.group_size,1).reshape(-1, 1, 1)
            weight = torch.ones_like(inverse_predict_depth[self._e_all])
            mask = self._stacked_depthmap_all  > 0.05
            weight[~mask] = 0 
            if len(self.invalid_depth_group) > 0:
                weight = weight.reshape(self.n_groups, self.group_size, -1, 1)
                weight[self.invalid_depth_group] = 0
                weight = weight.reshape(self.n_groups * self.group_size, -1, 1)
            
            scaled_invdepth = self._stacked_depthmap_all * s + t
            depth_loss = self.dist(inverse_predict_depth[self._e_all], scaled_invdepth, weight=weight[..., 0]).sum() / self.total_area_all
            depth_loss = depth_loss * 2
            # if epoch % 100 == 0:
            #     print(f'depth loss: {depth_loss}')

        if self.traj_dict is not None and epoch >= self.depth_traj_start_iter:
            # valid_traj_group_list = []
            if epoch == self.depth_traj_start_iter:
                # import pdb;pdb.set_trace() # we need to do some check here
                self.valid_traj_group_list, self.valid_group_idx, aligned_traj = self._set_traj()  # we need to check if the aligned trajectory is the same with tray_rot_xyz_homo
                print('valid_traj_group_list', self.valid_traj_group_list)
            if len(self.valid_traj_group_list) > 0:
                scale, RT = self.get_traj_align_poses()
                # valid_traj_align_poses = traj_align_poses[valid_group_list]
                scale, RT = scale[self.valid_traj_group_list], RT[self.valid_traj_group_list]
                stacked_traj = self._stacked_traj_all.reshape(self.n_groups, self.group_size, 4, 4)
                stacked_traj = stacked_traj[self.valid_traj_group_list]
                tray_xyz = stacked_traj[:, :, :3, [3]] * scale.reshape(-1, 1, 1, 1).repeat(1, self.group_size, 1, 1)
                tray_rot = stacked_traj[:, :, :3, :3]
                tray_rot_xyz = torch.cat([tray_rot, tray_xyz], dim=-1)
                cat_tensor = torch.tensor([0,0,0,1]).to(self.device).reshape(1, 1, 1, 4).repeat(tray_rot_xyz.shape[0], tray_rot_xyz.shape[1], 1, 1)
                tray_rot_xyz_homo = torch.cat([tray_rot_xyz, cat_tensor], dim=-2)
                tray_rot_xyz_homo = torch.bmm(RT.reshape(-1, 1, 4, 4).repeat(1, self.group_size, 1, 1).reshape(-1, 4, 4), tray_rot_xyz_homo.reshape(-1,4,4))
                loss_traj = self.relative_pose_loss(tray_rot_xyz_homo.reshape(-1, 4, 4), self.get_im_poses()[self.valid_group_idx]).sum()
                # loss_traj = self.relative_pose_loss_no_rotation(tray_rot_xyz_homo.reshape(-1, 4, 4), self.get_im_poses()[self.valid_group_idx]).sum()

            # if epoch % 100 == 0:
            #     print(f'traj loss: {loss_traj}')


        # camera temporal loss
        if self.temporal_smoothing_weight > 0:
            temporal_smoothing_loss = self.relative_pose_loss(self.get_im_poses()[:-1], self.get_im_poses()[1:]).sum()
        else:
            temporal_smoothing_loss = 0

        loss = (li + ray_loss + depth_loss) * 1 + loss_traj * 0.005 + self.temporal_smoothing_weight * temporal_smoothing_loss

        return loss
    
    

    def relative_pose_loss(self, RT1, RT2):
        relative_RT = torch.matmul(torch.inverse(RT1), RT2)
        rotation_diff = relative_RT[:, :3, :3]
        translation_diff = relative_RT[:, :3, 3]

        # Frobenius norm for rotation difference
        rotation_loss = torch.norm(rotation_diff - (torch.eye(3, device=RT1.device)), dim=(1, 2))

        # L2 norm for translation difference
        translation_loss = torch.norm(translation_diff, dim=1)

        # Combined loss (one can weigh these differently if needed)
        pose_loss = rotation_loss + translation_loss * self.translation_weight
        return pose_loss

    def relative_pose_loss_no_rotation(self, RT1, RT2):
        relative_RT = torch.matmul(torch.inverse(RT1), RT2)
        rotation_diff = relative_RT[:, :3, :3]
        translation_diff = relative_RT[:, :3, 3]

        # Frobenius norm for rotation difference
        rotation_loss = torch.norm(rotation_diff - (torch.eye(3, device=RT1.device)), dim=(1, 2))

        # L2 norm for translation difference
        translation_loss = torch.norm(translation_diff, dim=1)

        # Combined loss (one can weigh these differently if needed)
        pose_loss = rotation_loss * 0.1 + translation_loss * self.translation_weight
        return pose_loss
    
def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img

def ordered_ratio(disp_a, disp_b, mask=None):
    ratio_a = torch.maximum(disp_a, disp_b) / \
        (torch.minimum(disp_a, disp_b)+1e-5)
    if mask is not None:
        ratio_a = ratio_a[mask]
    return ratio_a - 1