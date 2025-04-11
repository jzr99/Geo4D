# --------------------------------------------------------
# Base class for the global alignement procedure
# --------------------------------------------------------
from copy import deepcopy
import cv2

import numpy as np
import torch
import torch.nn as nn
import roma
from copy import deepcopy
import tqdm

from dust3r.utils.geometry import inv, geotrf
from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb
from dust3r.viz import SceneViz, segment_sky, auto_cam_size
from dust3r.optim_factory import adjust_learning_rate_by_lr

from dust3r.cloud_opt.commons import (edge_str, ALL_DISTS, NoGradParamDict, NoGradParamList, get_imshapes, signed_expm1, signed_log1p,
                                      cosine_schedule, linear_schedule, cycled_linear_schedule, get_conf_trf, get_imshapes_group)
import dust3r.cloud_opt.init_im_poses as init_fun
from scipy.spatial.transform import Rotation
from dust3r.utils.vo_eval import save_trajectory_tum_format
import os
import matplotlib.pyplot as plt
from PIL import Image

def c2w_to_tumpose(c2w):
    """
    Convert a camera-to-world matrix to a tuple of translation and rotation
    
    input: c2w: 4x4 matrix
    output: tuple of translation and rotation (x y z qw qx qy qz)
    """
    # convert input to numpy
    c2w = to_numpy(c2w)
    xyz = c2w[:3, -1]
    rot = Rotation.from_matrix(c2w[:3, :3])
    qx, qy, qz, qw = rot.as_quat()
    tum_pose = np.concatenate([xyz, [qw, qx, qy, qz]])
    return tum_pose

def get_tum_poses(c2w_poses):
        poses = c2w_poses
        tt = np.arange(len(poses)).astype(float)
        tum_poses = [c2w_to_tumpose(p) for p in poses]
        tum_poses = np.stack(tum_poses, 0)
        return [tum_poses, tt]


import matplotlib.cm as cm

class ColorMapper:
    # a color mapper to map depth values to a certain colormap
    def __init__(self, colormap: str = "inferno"):

        if colormap == 'Greys':
            self.colormap = None
        else:
            self.colormap = torch.tensor(cm.get_cmap(colormap).colors)

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        # assert len(image.shape) == 2
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        if self.colormap is None:
            return image
        else:
            image = (image * 255).long()
            image = image.clamp(0, 255)
            image = self.colormap[image]
            return image


def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None, colormap=None):
    if colormap is not None:
        visualizer = ColorMapper(colormap)
    else:
        visualizer = ColorMapper()
    if v_min is None:
        # v_min = depths.min()
        v_min = np.percentile(depths, 2)
    if v_max is None:
        # v_max = depths.max()
        v_max = np.percentile(depths, 98)
    res = visualizer.apply(torch.tensor(depths), v_min=v_min, v_max=v_max).numpy()
    return res



class LightBaseGroupPCOptimizer (nn.Module):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0:
            other = deepcopy(args[0])
            attrs = '''edges is_symmetrized dist n_imgs pred_i pred_j imshapes 
                        min_conf_thr conf_thr conf_i conf_j im_conf
                        base_scale norm_pw_scale POSE_DIM pw_poses 
                        pw_adaptors pw_adaptors has_im_poses rand_pose imgs verbose'''.split()
            self.__dict__.update({k: other[k] for k in attrs})
        else:
            self._init_from_views(*args, **kwargs)

    def _init_from_views(self, view_list, pred_list,
                         dist='l1',
                         conf='log',
                         min_conf_thr=3,
                         thr_for_init_conf=False,
                         base_scale=0.5,
                         allow_pw_adaptors=False,
                         pw_break=20,
                         rand_pose=torch.randn,
                         empty_cache=False,
                         verbose=True,
                         opt_raydir=False):
        super().__init__()
        self.opt_raydir = opt_raydir
        # print('opt_raydir', self.opt_raydir)
        self.groups = []
        for view in view_list:
            self.groups.append([int(v['idx'][-1]) for v in view])
        # print('groups', self.groups)
        self.dist = ALL_DISTS[dist]
        self.verbose = verbose
        self.empty_cache = empty_cache
        self.n_imgs = max(max(e) for e in self.groups) + 1
        self.group_size = len(self.groups[0])
        self.pred_dict = NoGradParamList([pred['pts3d'] for n, pred in enumerate(pred_list)])
        self.imshapes = get_imshapes_group(self.groups, self.pred_dict)
        self.conf_dict = NoGradParamList([pred['conf'].squeeze(-1) for n, pred in enumerate(pred_list)])
        
        if self.opt_raydir:
            self.raydir_dict = NoGradParamList([pred['raydir'] for n, pred in enumerate(pred_list)])
        else:
            self.raydir_dict = None

        if pred_list[0].get('inverse_depthmap', None) is not None:
            self.inverse_depthmap_dict = NoGradParamList([pred['inverse_depthmap'] for n, pred in enumerate(pred_list)])
            # TODO init s & t
            self.s_depth = nn.Parameter(torch.ones((self.n_groups, 1), device=self.device))
            self.t_depth = nn.Parameter(torch.zeros((self.n_groups, 1), device=self.device))
        else:
            self.inverse_depthmap_dict = None
        
        if pred_list[0].get('crossmap', None) is not None:
            self.crossmap_dict = NoGradParamList([pred['crossmap'] for n, pred in enumerate(pred_list)])
        else:
            self.crossmap_dict = None

        if pred_list[0].get('traj',  None) is not None:
            self.traj_dict = NoGradParamList([pred['traj'] for n, pred in enumerate(pred_list)])
        else:
            self.traj_dict = None

        self.min_conf_thr = min_conf_thr
        self.thr_for_init_conf = thr_for_init_conf
        self.conf_trf = get_conf_trf(conf)
        
        self.im_conf = self._compute_single_img_conf_group(self.conf_dict)
        for i in range(len(self.im_conf)):
            self.im_conf[i].requires_grad = False

        self.init_conf_maps = [c.clone() for c in self.im_conf]

        # pairwise pose parameters
        self.base_scale = base_scale
        self.norm_pw_scale = True
        self.pw_break = pw_break
        self.POSE_DIM = 7
        self.pw_poses = nn.Parameter(rand_pose((self.n_groups, 1+self.POSE_DIM)))  # pairwise poses
        self.pw_adaptors = nn.Parameter(torch.zeros((self.n_groups, 2)))  # slight xy/z adaptation
        self.pw_adaptors.requires_grad_(allow_pw_adaptors)
        self.has_im_poses = False
        self.rand_pose = rand_pose

        self.traj_align_poses = nn.Parameter(rand_pose((self.n_groups, 1+self.POSE_DIM)))

        # possibly store images, camera_pose, instance for show_pointcloud
        self.imgs = None
        if 'img' in view_list[0][0]:
            imgs = [torch.zeros((3,)+hw) for hw in self.imshapes]
            for v in range(len(self.groups)):
                for view in view_list[v]:
                    idx = view['idx'][-1]
                    if len(view['img'].shape)==4:
                        imgs[idx] = view['img'][0]
                    else:
                        imgs[idx] = view['img']

                # idx = view2['idx'][v]
                # imgs[idx] = view2['img'][v]
            self.imgs = rgb(imgs)


    @property
    def n_groups(self):
        return len(self.groups)
    

    @property
    def imsizes(self):
        return [(w, h) for h, w in self.imshapes]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def state_dict(self, trainable=True):
        all_params = super().state_dict()
        # return {k: v for k, v in all_params.items() if k.startswith(('_', 'pred_i.', 'pred_j.', 'conf_i.', 'conf_j.')) != trainable}
        return {k: v for k, v in all_params.items() if k.startswith(('_', 'pred_dict.', 'conf_dict.')) != trainable}

    def load_state_dict(self, data):
        return super().load_state_dict(self.state_dict(trainable=False) | data)

    def _check_edges(self):
        indices = sorted({i for edge in self.edges for i in edge})
        assert indices == list(range(len(indices))), 'bad pair indices: missing values '
        return len(indices)

    @torch.no_grad()
    def _compute_single_img_conf(self, pred1_conf):
        im_conf = nn.ParameterList([torch.zeros(hw, device=self.device) for hw in self.imshapes])
        for e, (i, j) in enumerate(self.edges):
            im_conf[j] = torch.maximum(im_conf[j], pred1_conf[e])
        return im_conf

    @torch.no_grad()
    def _compute_single_img_conf_group(self, pred_dict):
        im_conf = nn.ParameterList([torch.zeros(hw, device=self.device) for hw in self.imshapes])
        for e, group in enumerate(self.groups):
            for idx,image_idx in enumerate(group):
                im_conf[image_idx] = torch.maximum(im_conf[image_idx], pred_dict[e][idx])
        return im_conf
    
    
    @torch.no_grad()
    def _compute_img_conf(self, pred1_conf, pred2_conf):
        im_conf = nn.ParameterList([torch.zeros(hw, device=self.device) for hw in self.imshapes])
        for e, (i, j) in enumerate(self.edges):
            im_conf[i] = torch.maximum(im_conf[i], pred1_conf[e])
            im_conf[j] = torch.maximum(im_conf[j], pred2_conf[e])
        return im_conf

    def get_adaptors(self):
        adapt = self.pw_adaptors
        adapt = torch.cat((adapt[:, 0:1], adapt), dim=-1)  # (scale_xy, scale_xy, scale_z)
        if self.norm_pw_scale:  # normalize so that the product == 1
            adapt = adapt - adapt.mean(dim=1, keepdim=True)
        return (adapt / self.pw_break).exp()

    def _get_poses(self, poses):
        # normalize rotation
        Q = poses[:, :4]
        T = signed_expm1(poses[:, 4:7])
        RT = roma.RigidUnitQuat(Q, T).normalize().to_homogeneous()
        return RT

    def _set_pose(self, poses, idx, R, T=None, scale=None, force=False, scale_T=True):
        # all poses == cam-to-world
        pose = poses[idx]
        if not (pose.requires_grad or force):
            return pose

        if R.shape == (4, 4):
            assert T is None
            T = R[:3, 3]
            R = R[:3, :3]

        if R is not None:
            pose.data[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None and scale_T:
            pose.data[4:7] = signed_log1p(T / (scale or 1))  # translation is function of scale
        if T is not None and not scale_T:
            pose.data[4:7] = signed_log1p(T)

        if scale is not None:
            assert poses.shape[-1] in (8, 13)
            pose.data[-1] = np.log(float(scale))
        return pose

    
    def se3(self, r: np.ndarray = np.eye(3), t: np.ndarray = np.array([0, 0, 0])) -> np.ndarray:
        """
        :param r: SO(3) rotation matrix
        :param t: 3x1 translation vector
        :return: SE(3) transformation matrix
        """
        se3 = np.eye(4)
        se3[:3, :3] = r
        se3[:3, 3] = t
        return se3
    
    
    def get_pw_norm_scale_factor(self):
        if self.norm_pw_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale
            return (np.log(self.base_scale) - self.pw_poses[:, -1].mean()).exp()
        else:
            return 1  # don't norm scale for known poses

    def get_pw_scale(self):
        scale = self.pw_poses[:, -1].exp()  # (n_edges,)
        scale = scale * self.get_pw_norm_scale_factor()
        return scale

    def get_pw_poses(self):  # cam to world
        RT = self._get_poses(self.pw_poses)
        scaled_RT = RT.clone()
        scaled_RT[:, :3] *= self.get_pw_scale().view(-1, 1, 1)  # scale the rotation AND translation
        return scaled_RT
    
    def get_traj_align_poses(self):
        RT = self._get_poses(self.traj_align_poses)
        scale = self.traj_align_poses[:, -1].exp()
        return scale, RT

    def get_masks(self):
        if self.thr_for_init_conf:
            return [(conf > self.min_conf_thr) for conf in self.init_conf_maps]
        else:
            return [(conf > self.min_conf_thr) for conf in self.im_conf]
    
    def depth_to_pts3d(self):
        raise NotImplementedError()

    def get_pts3d(self, raw=False, **kwargs):
        res = self.depth_to_pts3d(**kwargs)
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def _set_focal(self, idx, focal, force=False):
        raise NotImplementedError()

    def get_focals(self):
        raise NotImplementedError()

    def get_known_focal_mask(self):
        raise NotImplementedError()

    def get_principal_points(self):
        raise NotImplementedError()

    def get_conf(self, mode=None):
        trf = self.conf_trf if mode is None else get_conf_trf(mode)
        return [trf(c) for c in self.im_conf]
    
    def get_init_conf(self, mode=None):
        trf = self.conf_trf if mode is None else get_conf_trf(mode)
        return [trf(c) for c in self.init_conf_maps]

    def get_im_poses(self):
        raise NotImplementedError()

    def _set_depthmap(self, idx, depth, force=False):
        raise NotImplementedError()

    def get_depthmaps(self, raw=False):
        raise NotImplementedError()

    def clean_pointcloud(self, **kw):
        cams = inv(self.get_im_poses())
        K = self.get_intrinsics()
        depthmaps = self.get_depthmaps()
        all_pts3d = self.get_pts3d()

        new_im_confs = clean_pointcloud(self.im_conf, K, cams, depthmaps, all_pts3d, **kw)

        for i, new_conf in enumerate(new_im_confs):
            self.im_conf[i].data[:] = new_conf
        return self

    def get_tum_poses(self):
        poses = self.get_im_poses()
        tt = np.arange(len(poses)).astype(float)
        tum_poses = [c2w_to_tumpose(p) for p in poses]
        tum_poses = np.stack(tum_poses, 0)
        return [tum_poses, tt]

    def save_tum_poses(self, path):
        traj = self.get_tum_poses()
        save_trajectory_tum_format(traj, path)
        return traj[0] # return the poses
    
    def save_focals(self, path):
        # convert focal to txt
        focals = self.get_focals()
        np.savetxt(path, focals.detach().cpu().numpy(), fmt='%.6f')
        return focals

    def save_intrinsics(self, path):
        K_raw = self.get_intrinsics()
        K = K_raw.reshape(-1, 9)
        np.savetxt(path, K.detach().cpu().numpy(), fmt='%.6f')
        return K_raw

    def save_conf_maps(self, path):
        conf = self.get_conf()
        for i, c in enumerate(conf):
            np.save(f'{path}/conf_{i}.npy', c.detach().cpu().numpy())
        return conf
    
    def save_init_conf_maps(self, path):
        conf = self.get_init_conf()
        for i, c in enumerate(conf):
            np.save(f'{path}/init_conf_{i}.npy', c.detach().cpu().numpy())
        return conf

    def save_rgb_imgs(self, path):
        imgs = self.imgs
        for i, img in enumerate(imgs):
            # convert from rgb to bgr
            img = img[..., ::-1]
            cv2.imwrite(f'{path}/frame_{i:04d}.png', img*255)
        return imgs

    def save_dynamic_masks(self, path):
        dynamic_masks = self.dynamic_masks if getattr(self, 'sam2_dynamic_masks', None) is None else self.sam2_dynamic_masks
        for i, dynamic_mask in enumerate(dynamic_masks):
            cv2.imwrite(f'{path}/dynamic_mask_{i}.png', (dynamic_mask * 255).detach().cpu().numpy().astype(np.uint8))
        return dynamic_masks

    def save_depth_maps(self, path):
        
        depth_maps = self.get_depthmaps()
        for i, depth_map in enumerate(depth_maps):
            # Apply color map to depth map
            np.save(f'{path}/frame_{(i):04d}.npy', depth_map.detach().cpu().numpy())
        
        
        # import pdb;pdb.set_trace()
        depth_maps = self.get_depthmaps(raw=True)
        # self.im_shapes = self.imshapes
        H,W = self.imshapes[0]
        depth_maps = depth_maps.reshape(-1, H, W)
        depth_maps = 1 / (depth_maps + 1e-6) 
        images = []
        # images_grey = []

        colored_depth_map = vis_sequence_depth(depth_maps.detach().cpu().numpy())
        # greys_depth_map = vis_sequence_depth(depth_maps.detach().cpu().numpy(), colormap='Greys')
        for i, depth_map in enumerate(colored_depth_map):
            # img_path_gray = f'{path}/frame_graydepth_{(i):04d}.png'
            img_path = f'{path}/frame_colordepth_{(i):04d}.png'
            # cv2.imwrite(img_path_gray, (greys_depth_map[i] * 255).astype(np.uint8))
            cv2.imwrite(img_path, cv2.cvtColor((depth_map * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
            images.append(Image.open(img_path))
            # images_grey.append(Image.open(img_path_gray))
        
        images[0].save(f'{path}/colored_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
        # images_grey[0].save(f'{path}/grey_depth_maps.gif', save_all=True, append_images=images_grey[1:], duration=100, loop=0)

        
        return depth_maps

    def forward(self, ret_details=False):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()
        # pre-compute pixel weights
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0
        if ret_details:
            details = -torch.ones((self.n_imgs, self.n_imgs))

        for e, (i, j) in enumerate(self.edges):
            i_j = edge_str(i, j)
            # distance in image i and j
            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss = loss + li + lj

            if ret_details:
                details[i, j] = li + lj
        loss /= self.n_edges  # average over all pairs

        if ret_details:
            return loss, details
        return loss

    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment(self, init=None, save_score_path=None, save_score_only=False, niter_PnP=10, **kw):
        if init is None:
            pass
        elif init == 'msp' or init == 'mst':
            init_fun.init_minimum_spanning_tree(self, save_score_path=save_score_path, save_score_only=save_score_only, niter_PnP=niter_PnP)
            if save_score_only: # if only want the score map
                return None
        elif init == 'fast_pnp':
            init_fun.init_from_fast_pnp(self, niter_PnP=niter_PnP)
        elif init == 'group':
            init_fun.init_from_group(self, niter_PnP=niter_PnP)
        elif init == 'known_poses':
            self.preset_pose(known_poses=self.camera_poses, requires_grad=True)
            init_fun.init_from_known_poses(self, min_conf_thr=self.min_conf_thr,
                                           niter_PnP=niter_PnP)
        else:
            raise ValueError(f'bad value for {init=}')

        return global_alignment_loop(self, **kw)

    @torch.no_grad()
    def mask_sky(self):
        res = deepcopy(self)
        for i in range(self.n_imgs):
            sky = segment_sky(self.imgs[i])
            res.im_conf[i][sky] = 0
        return res

    def show(self, show_pw_cams=False, show_pw_pts3d=False, cam_size=None, **kw):
        viz = SceneViz()
        if self.imgs is None:
            colors = np.random.randint(0, 256, size=(self.n_imgs, 3))
            colors = list(map(tuple, colors.tolist()))
            for n in range(self.n_imgs):
                viz.add_pointcloud(self.get_pts3d()[n], colors[n], self.get_masks()[n])
        else:
            viz.add_pointcloud(self.get_pts3d(), self.imgs, self.get_masks())
            colors = np.random.randint(256, size=(self.n_imgs, 3))

        # camera poses
        im_poses = to_numpy(self.get_im_poses())
        if cam_size is None:
            cam_size = auto_cam_size(im_poses)
        viz.add_cameras(im_poses, self.get_focals(), colors=colors,
                        images=self.imgs, imsizes=self.imsizes, cam_size=cam_size)
        if show_pw_cams:
            pw_poses = self.get_pw_poses()
            viz.add_cameras(pw_poses, color=(192, 0, 192), cam_size=cam_size)

            if show_pw_pts3d:
                pts = [geotrf(pw_poses[e], self.pred_i[edge_str(i, j)]) for e, (i, j) in enumerate(self.edges)]
                viz.add_pointcloud(pts, (128, 0, 128))

        viz.show(**kw)
        return viz


def global_alignment_loop(net, lr=0.01, niter=300, schedule='cosine', lr_min=1e-3, temporal_smoothing_weight=0, depth_map_save_dir=None):
    print(">>>>>>>before alignment loop<<<<<<<<<")
    # if net.inverse_depthmap_dict is not None:
    #     net._set_st_depth()
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return net

    verbose = net.verbose
    if verbose:
        print('Global alignement - optimizing for:')
        print([name for name, value in net.named_parameters() if value.requires_grad])

    lr_base = lr
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

    loss = float('inf')
    if verbose:
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                if bar.n % 500 == 0 and depth_map_save_dir is not None:
                    if not os.path.exists(depth_map_save_dir):
                        os.makedirs(depth_map_save_dir)
                    # visualize the depthmaps
                    depth_maps = net.get_depthmaps()
                    for i, depth_map in enumerate(depth_maps):
                        depth_map_save_path = os.path.join(depth_map_save_dir, f'depthmaps_{i}_iter_{bar.n}.png')
                        plt.imsave(depth_map_save_path, depth_map.detach().cpu().numpy(), cmap='jet')
                    print(f"Saved depthmaps at iteration {bar.n} to {depth_map_save_dir}")
                loss, lr = global_alignment_iter(net, bar.n, niter, lr_base, lr_min, optimizer, schedule, 
                                                 temporal_smoothing_weight=temporal_smoothing_weight)
                bar.set_postfix_str(f'{lr=:g} loss={loss:g}')
                bar.update()
    else:
        for n in range(niter):
            loss, _ = global_alignment_iter(net, n, niter, lr_base, lr_min, optimizer, schedule, 
                                            temporal_smoothing_weight=temporal_smoothing_weight)
    return loss


def global_alignment_iter(net, cur_iter, niter, lr_base, lr_min, optimizer, schedule, temporal_smoothing_weight=0):
    t = cur_iter / niter
    if schedule == 'cosine':
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == 'linear':
        lr = linear_schedule(t, lr_base, lr_min)
    elif schedule.startswith('cycle'):
        try:
            num_cycles = int(schedule[5:])
        except ValueError:
            num_cycles = 2
        lr = cycled_linear_schedule(t, lr_base, lr_min, num_cycles=num_cycles)
    else:
        raise ValueError(f'bad lr {schedule=}')
    
    adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()

    if net.empty_cache:
        torch.cuda.empty_cache()
    
    loss = net(epoch=cur_iter)
    
    if net.empty_cache:
        torch.cuda.empty_cache()
    
    loss.backward()
    
    if net.empty_cache:
        torch.cuda.empty_cache()
    
    optimizer.step()
    
    return float(loss), lr



@torch.no_grad()
def clean_pointcloud( im_confs, K, cams, depthmaps, all_pts3d, 
                      tol=0.001, bad_conf=0, dbg=()):
    """ Method: 
    1) express all 3d points in each camera coordinate frame
    2) if they're in front of a depthmap --> then lower their confidence
    """
    assert len(im_confs) == len(cams) == len(K) == len(depthmaps) == len(all_pts3d)
    assert 0 <= tol < 1
    res = [c.clone() for c in im_confs]

    # reshape appropriately
    all_pts3d = [p.view(*c.shape,3) for p,c in zip(all_pts3d, im_confs)]
    depthmaps = [d.view(*c.shape) for d,c in zip(depthmaps, im_confs)]
    
    for i, pts3d in enumerate(all_pts3d):
        for j in range(len(all_pts3d)):
            if i == j: continue

            # project 3dpts in other view
            proj = geotrf(cams[j], pts3d)
            proj_depth = proj[:,:,2]
            u,v = geotrf(K[j], proj, norm=1, ncol=2).round().long().unbind(-1)

            # check which points are actually in the visible cone
            H, W = im_confs[j].shape
            msk_i = (proj_depth > 0) & (0 <= u) & (u < W) & (0 <= v) & (v < H)
            msk_j = v[msk_i], u[msk_i]

            # find bad points = those in front but less confident
            bad_points = (proj_depth[msk_i] < (1-tol) * depthmaps[j][msk_j]) & (res[i][msk_i] < res[j][msk_j])

            bad_msk_i = msk_i.clone()
            bad_msk_i[msk_i] = bad_points
            res[i][bad_msk_i] = res[i][bad_msk_i].clip_(max=bad_conf)

    return res
