# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# geometry utilitary functions
# --------------------------------------------------------
from typing import *
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import cKDTree as KDTree
from functools import partial

from utils.misc import invalid_to_zeros, invalid_to_nans
from utils.device import to_numpy

# decorator
from numbers import Number
import inspect
from functools import wraps


def get_device(args, kwargs):
    device = None
    for arg in (list(args) + list(kwargs.values())):
        if isinstance(arg, torch.Tensor):
            if device is None:
                device = arg.device
            elif device != arg.device:
                raise ValueError("All tensors must be on the same device.")
    return device


def get_args_order(func, args, kwargs):
    """
    Get the order of the arguments of a function.
    """
    names = inspect.getfullargspec(func).args
    names_idx = {name: i for i, name in enumerate(names)}
    args_order = []
    kwargs_order = {}
    for name, arg in kwargs.items():
        if name in names:
            kwargs_order[name] = names_idx[name]
            names.remove(name)
    for i, arg in enumerate(args):
        if i < len(names):
            args_order.append(names_idx[names[i]])
    return args_order, kwargs_order


def broadcast_args(args, kwargs, args_dim, kwargs_dim):
    spatial = []
    for arg, arg_dim in zip(args + list(kwargs.values()), args_dim + list(kwargs_dim.values())):
        if isinstance(arg, torch.Tensor) and arg_dim is not None:
            arg_spatial = arg.shape[:arg.ndim-arg_dim]
            if len(arg_spatial) > len(spatial):
                spatial = [1] * (len(arg_spatial) - len(spatial)) + spatial
            for j in range(len(arg_spatial)):
                if spatial[-j] < arg_spatial[-j]:
                    if spatial[-j] == 1:
                        spatial[-j] = arg_spatial[-j]
                    else:
                        raise ValueError("Cannot broadcast arguments.")
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and args_dim[i] is not None:
            args[i] = torch.broadcast_to(arg, [*spatial, *arg.shape[arg.ndim-args_dim[i]:]])
    for key, arg in kwargs.items():
        if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
            kwargs[key] = torch.broadcast_to(arg, [*spatial, *arg.shape[arg.ndim-kwargs_dim[key]:]])
    return args, kwargs, spatial

def batched(*dims):
    """
    Decorator that allows a function to be called with batched arguments.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, device=torch.device('cpu'), **kwargs):
            args = list(args)
            # get arguments dimensions
            args_order, kwargs_order = get_args_order(func, args, kwargs)
            args_dim = [dims[i] for i in args_order]
            kwargs_dim = {key: dims[i] for key, i in kwargs_order.items()}
            # convert to torch tensor
            device = get_device(args, kwargs) or device
            for i, arg in enumerate(args):
                if isinstance(arg, (Number, list, tuple)) and args_dim[i] is not None:
                    args[i] = torch.tensor(arg, device=device)
            for key, arg in kwargs.items():
                if isinstance(arg, (Number, list, tuple)) and kwargs_dim[key] is not None:
                    kwargs[key] = torch.tensor(arg, device=device)
            # broadcast arguments
            args, kwargs, spatial = broadcast_args(args, kwargs, args_dim, kwargs_dim)
            for i, (arg, arg_dim) in enumerate(zip(args, args_dim)):
                if isinstance(arg, torch.Tensor) and arg_dim is not None:
                    args[i] = arg.reshape([-1, *arg.shape[arg.ndim-arg_dim:]])
            for key, arg in kwargs.items():
                if isinstance(arg, torch.Tensor) and kwargs_dim[key] is not None:
                    kwargs[key] = arg.reshape([-1, *arg.shape[arg.ndim-kwargs_dim[key]:]])
            # call function
            results = func(*args, **kwargs)
            type_results = type(results)
            results = list(results) if isinstance(results, (tuple, list)) else [results]
            # restore spatial dimensions
            for i, result in enumerate(results):
                results[i] = result.reshape([*spatial, *result.shape[1:]])
            if type_results == tuple:
                results = tuple(results)
            elif type_results == list:
                results = list(results)
            else:
                results = results[0]
            return results
        return wrapper
    return decorator

@batched(0,0,0,0)
def intrinsics_from_focal_center(
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor]
) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix

    Args:
        focal_x (float | torch.Tensor): focal length in x axis
        focal_y (float | torch.Tensor): focal length in y axis
        cx (float | torch.Tensor): principal point in x axis
        cy (float | torch.Tensor): principal point in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    N = fx.shape[0]
    ret = torch.zeros((N, 3, 3), dtype=fx.dtype, device=fx.device)
    zeros, ones = torch.zeros(N, dtype=fx.dtype, device=fx.device), torch.ones(N, dtype=fx.dtype, device=fx.device)
    ret = torch.stack([fx, zeros, cx, zeros, fy, cy, zeros, zeros, ones], dim=-1).unflatten(-1, (3, 3))
    return ret

def intrinsics_from_fov_xy(
        fov_x: Union[float, torch.Tensor],
        fov_y: Union[float, torch.Tensor]
    ) -> torch.Tensor:
    """
    Get OpenCV intrinsics matrix from field of view in x and y axis

    Args:
        fov_x (float | torch.Tensor): field of view in x axis
        fov_y (float | torch.Tensor): field of view in y axis

    Returns:
        (torch.Tensor): [..., 3, 3] OpenCV intrinsics matrix
    """
    focal_x = 0.5 / torch.tan(fov_x / 2)
    focal_y = 0.5 / torch.tan(fov_y / 2)
    cx = cy = 0.5
    return intrinsics_from_focal_center(focal_x, focal_y, cx, cy)

def point_map_to_depth(points: torch.Tensor, mask: torch.Tensor = None, downsample_size: Tuple[int, int] = (64, 64)):
    """
    Recover the depth map and FoV from a point map with unknown z shift and focal.

    Note that it assumes:
    - the optical center is at the center of the map
    - the map is undistorted
    - the map is isometric in the x and y directions

    ### Parameters:
    - `points: torch.Tensor` of shape (..., H, W, 3)
    - `downsample_size: Tuple[int, int]` in (height, width), the size of the downsampled map. Downsampling produces approximate solution and is efficient for large maps.

    ### Returns:
    - `depth: torch.Tensor` of shape (..., H, W)
    - `fov_x: torch.Tensor` of shape (...)
    - `fov_y: torch.Tensor` of shape (...)
    - `shift: torch.Tensor` of shape (...), the z shift, making `depth = points[..., 2] + shift`
    """
    shape = points.shape
    height, width = points.shape[-3], points.shape[-2]
    diagonal = (height ** 2 + width ** 2) ** 0.5

    points = points.reshape(-1, *shape[-3:])
    mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
    uv = image_plane_uv(width, height, dtype=points.dtype, device=points.device)  # (H, W, 2)

    points_lr = F.interpolate(points.permute(0, 3, 1, 2), downsample_size, mode='nearest').permute(0, 2, 3, 1)
    uv_lr = F.interpolate(uv.unsqueeze(0).permute(0, 3, 1, 2), downsample_size, mode='nearest').squeeze(0).permute(1, 2, 0)
    mask_lr = None if mask is None else F.interpolate(mask.to(torch.float32).unsqueeze(1), downsample_size, mode='nearest').squeeze(1) > 0
    
    uv_lr_np = uv_lr.cpu().numpy()
    points_lr_np = points_lr.detach().cpu().numpy()
    mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
    optim_shift, optim_focal = [], []
    for i in range(points.shape[0]):
        points_lr_i_np = points_lr_np[i] if mask is None else points_lr_np[i][mask_lr_np[i]]
        uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
        optim_shift_i, optim_focal_i = solve_optimal_shift_focal(uv_lr_i_np, points_lr_i_np, ransac_iters=None)
        # optim_shift_i, optim_focal_i = solve_optimal_shift_focal(uv_lr_i_np, points_lr_i_np, ransac_iters=10)
        optim_shift.append(float(optim_shift_i))
        optim_focal.append(float(optim_focal_i))
    optim_shift = torch.tensor(optim_shift, device=points.device, dtype=points.dtype)
    optim_focal = torch.tensor(optim_focal, device=points.device, dtype=points.dtype)

    fov_x = 2 * torch.atan(width / diagonal / optim_focal)
    fov_y = 2 * torch.atan(height / diagonal / optim_focal)
    
    depth = (points[..., 2] + optim_shift[:, None, None]).reshape(shape[:-1])
    fov_x = fov_x.reshape(shape[:-3])
    fov_y = fov_y.reshape(shape[:-3])
    optim_shift = optim_shift.reshape(shape[:-3])

    return depth, fov_x, fov_y, optim_shift

def image_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height
    
    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv


def solve_optimal_shift_focal(uv: np.ndarray, xyz: np.ndarray, ransac_iters: int = None, ransac_hypothetical_size: float = 0.1, ransac_threshold: float = 0.1):
    "Solve `min |focal * xy / (z + shift) - uv|` with respect to shift and focal"
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)

    def fn(uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray):
        xy_proj = xy / (z + shift)[: , None]
        f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
        err = (f * xy_proj - uv).ravel()
        return err

    initial_shift = 0 #-z.min(keepdims=True) + 1.0

    if ransac_iters is None:
        solution = least_squares(partial(fn, uv, xy, z), x0=initial_shift, ftol=1e-3, method='lm')
        optim_shift = solution['x'].squeeze().astype(np.float32)
    else:
        best_err, best_shift = np.inf, None
        for _ in range(ransac_iters):
            maybe_inliers = np.random.choice(len(z), size=int(ransac_hypothetical_size * len(z)), replace=False)
            solution = least_squares(partial(fn, uv[maybe_inliers], xy[maybe_inliers], z[maybe_inliers]), x0=initial_shift, ftol=1e-3, method='lm')
            maybe_shift = solution['x'].squeeze().astype(np.float32)
            confirmed_inliers = np.linalg.norm(fn(uv, xy, z, maybe_shift).reshape(-1, 2), axis=-1) < ransac_threshold
            if confirmed_inliers.sum() > 10:
                solution = least_squares(partial(fn, uv[confirmed_inliers], xy[confirmed_inliers], z[confirmed_inliers]), x0=maybe_shift, ftol=1e-3, method='lm')
                better_shift = solution['x'].squeeze().astype(np.float32)
            else:
                better_shift = maybe_shift
            err = np.linalg.norm(fn(uv, xy, z, better_shift).reshape(-1, 2), axis=-1).clip(max=ransac_threshold).mean()
            if err < best_err:
                best_err, best_shift = err, better_shift
                initial_shift = best_shift
            
        optim_shift = best_shift

    xy_proj = xy / (z + optim_shift)[: , None]
    optim_focal = (xy_proj * uv).sum() / (xy_proj * xy_proj).sum()

    return optim_shift, optim_focal

def xy_grid(W, H, device=None, origin=(0, 0), unsqueeze=None, cat_dim=-1, homogeneous=False, **arange_kw):
    """ Output a (H,W,2) array of int32 
        with output[j,i,0] = i + origin[0]
             output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing='xy')
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


def depthmap_to_pts3d(depth, pseudo_focal, pp=None, **_):
    """
    Args:
        - depthmap (BxHxW array):
        - pseudo_focal: [B,H,W] ; [B,2,H,W] or [B,1,H,W]
    Returns:
        pointmap of absolute coordinates (BxHxWx3 array)
    """

    if len(depth.shape) == 4:
        B, H, W, n = depth.shape
    else:
        B, H, W = depth.shape
        n = None

    if len(pseudo_focal.shape) == 3:  # [B,H,W]
        pseudo_focalx = pseudo_focaly = pseudo_focal
    elif len(pseudo_focal.shape) == 4:  # [B,2,H,W] or [B,1,H,W]
        pseudo_focalx = pseudo_focal[:, 0]
        if pseudo_focal.shape[1] == 2:
            pseudo_focaly = pseudo_focal[:, 1]
        else:
            pseudo_focaly = pseudo_focalx
    else:
        raise NotImplementedError("Error, unknown input focal shape format.")

    assert pseudo_focalx.shape == depth.shape[:3]
    assert pseudo_focaly.shape == depth.shape[:3]
    grid_x, grid_y = xy_grid(W, H, cat_dim=0, device=depth.device)[:, None]

    # set principal point
    if pp is None:
        grid_x = grid_x - (W - 1) / 2
        grid_y = grid_y - (H - 1) / 2
    else:
        grid_x = grid_x.expand(B, -1, -1) - pp[:, 0, None, None]
        grid_y = grid_y.expand(B, -1, -1) - pp[:, 1, None, None]

    if n is None:
        pts3d = torch.empty((B, H, W, 3), device=depth.device)
        pts3d[..., 0] = depth * grid_x / pseudo_focalx
        pts3d[..., 1] = depth * grid_y / pseudo_focaly
        pts3d[..., 2] = depth
    else:
        pts3d = torch.empty((B, H, W, 3, n), device=depth.device)
        pts3d[..., 0, :] = depth * (grid_x / pseudo_focalx)[..., None]
        pts3d[..., 1, :] = depth * (grid_y / pseudo_focaly)[..., None]
        pts3d[..., 2, :] = depth
    return pts3d


def depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    z_cam = depthmap
    x_cam = (u - cu) * z_cam / fu
    y_cam = (v - cv) * z_cam / fv
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return X_cam, valid_mask


def distance_depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # z_cam = depthmap
    # x_cam = (u - cu) * z_cam / fu
    # y_cam = (v - cv) * z_cam / fv
    z_cam = np.ones_like(depthmap)
    x_cam = (u - cu) / fu
    y_cam = (v - cv) / fv
    # import pdb;pdb.set_trace()
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)
    X_cam_norm = np.linalg.norm(X_cam, axis=-1, keepdims=True)
    rescaled_X_cam = (depthmap[:,:,np.newaxis] / X_cam_norm) * X_cam

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return rescaled_X_cam, valid_mask

def kubric_distance_depthmap_to_camera_coordinates(depthmap, camera_intrinsics, pseudo_focal=None):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels.
    """
    camera_intrinsics = np.float32(camera_intrinsics)
    H, W = depthmap.shape

    # Compute 3D ray associated with each pixel
    # Strong assumption: there are no skew terms
    assert camera_intrinsics[0, 1] == 0.0
    assert camera_intrinsics[1, 0] == 0.0
    if pseudo_focal is None:
        fu = camera_intrinsics[0, 0]
        fv = camera_intrinsics[1, 1]
    else:
        assert pseudo_focal.shape == (H, W)
        fu = fv = pseudo_focal
    cu = camera_intrinsics[0, 2]
    cv = camera_intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    # z_cam = depthmap
    # x_cam = (u - cu) * z_cam / fu
    # y_cam = (v - cv) * z_cam / fv
    z_cam = -np.ones_like(depthmap)
    x_cam = (u + cu) / fu
    y_cam = (v + cv) / fv
    # import pdb;pdb.set_trace()
    X_cam = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)
    X_cam_norm = np.linalg.norm(X_cam, axis=-1, keepdims=True)
    rescaled_X_cam = (depthmap[:,:,np.newaxis] / X_cam_norm) * X_cam

    # Mask for valid coordinates
    valid_mask = (depthmap > 0.0)
    return rescaled_X_cam, valid_mask

def kubric_distance_depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = kubric_distance_depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask



def distance_depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = distance_depthmap_to_camera_coordinates(depthmap, camera_intrinsics)

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask


def depthmap_to_absolute_camera_coordinates(depthmap, camera_intrinsics, camera_pose, z_far=80, **kw):
    """
    Args:
        - depthmap (HxW array):
        - camera_intrinsics: a 3x3 matrix
        - camera_pose: a 4x3 or 4x4 cam2world matrix
    Returns:
        pointmap of absolute coordinates (HxWx3 array), and a mask specifying valid pixels."""
    X_cam, valid_mask = depthmap_to_camera_coordinates(depthmap, camera_intrinsics)
    if z_far > 0:
        valid_mask = valid_mask & (depthmap < z_far)

    large_depth_map_mask = depthmap > z_far

    X_world = X_cam # default
    if camera_pose is not None:
        # R_cam2world = np.float32(camera_params["R_cam2world"])
        # t_cam2world = np.float32(camera_params["t_cam2world"]).squeeze()
        R_cam2world = camera_pose[:3, :3]
        t_cam2world = camera_pose[:3, 3]

        # Express in absolute coordinates (invalid depth values)
        X_world = np.einsum("ik, vuk -> vui", R_cam2world, X_cam) + t_cam2world[None, None, :]

    return X_world, valid_mask, large_depth_map_mask


def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K


def normalize_pointcloud(pts1, pts2, norm_mode='avg_dis', valid1=None, valid2=None, ret_factor=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
    """
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    assert pts2 is None or (pts2.ndim >= 3 and pts2.shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split('_')

    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        nan_pts1, nnz1 = invalid_to_zeros(pts1, valid1, ndim=3)
        nan_pts2, nnz2 = invalid_to_zeros(pts2, valid2, ndim=3) if pts2 is not None else (None, 0)
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis':
            pass  # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            # actually warp input points before normalizing them
            log_dis = torch.log1p(all_dis)
            warp_factor = log_dis / all_dis.clip(min=1e-8)
            H1, W1 = pts1.shape[1:-1]
            pts1 = pts1 * warp_factor[:, :W1 * H1].view(-1, H1, W1, 1)
            if pts2 is not None:
                H2, W2 = pts2.shape[1:-1]
                pts2 = pts2 * warp_factor[:, W1 * H1:].view(-1, H2, W2, 1)
            all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz1 + nnz2 + 1e-8)
    else:
        # gather all points together (joint normalization)
        nan_pts1 = invalid_to_nans(pts1, valid1, ndim=3)
        nan_pts2 = invalid_to_nans(pts2, valid2, ndim=3) if pts2 is not None else None
        all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)

        if norm_mode == 'avg':
            norm_factor = all_dis.nanmean(dim=1)
        elif norm_mode == 'median':
            norm_factor = all_dis.nanmedian(dim=1).values.detach()
        elif norm_mode == 'sqrt':
            norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        else:
            raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts1.ndim:
        norm_factor.unsqueeze_(-1)

    res = pts1 / norm_factor
    if pts2 is not None:
        res = (res, pts2 / norm_factor)
    if ret_factor:
        res = res + (norm_factor,)
    return res


def normalize_pointcloud_all(pts_all, valid_all, norm_mode='avg_dis', ret_factor=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
    """

    norm_mode, dis_mode = norm_mode.split('_')
    nan_pts_all, nnz_all = [], 0
    for pts, valid in zip(pts_all, valid_all):
        assert pts.ndim >= 3 and pts.shape[-1] == 3
        nan_pts, nnz = invalid_to_zeros(pts, valid, ndim=3)
        nan_pts_all.append(nan_pts)
        nnz_all += nnz
    # import pdb;pdb.set_trace()
    all_pts = torch.cat(nan_pts_all, dim=0).reshape(1, -1, 3)



    if norm_mode == 'avg':
        # gather all points together (joint normalization)
        # nan_pts1, nnz1 = invalid_to_zeros(pts1, valid1, ndim=3)
        # nan_pts2, nnz2 = invalid_to_zeros(pts2, valid2, ndim=3) if pts2 is not None else (None, 0)
        # all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == 'dis':
            pass  # do nothing
        elif dis_mode == 'log1p':
            all_dis = torch.log1p(all_dis)
        elif dis_mode == 'warp-log1p':
            raise NotImplementedError
            # # actually warp input points before normalizing them
            # log_dis = torch.log1p(all_dis)
            # warp_factor = log_dis / all_dis.clip(min=1e-8)
            # H1, W1 = pts1.shape[1:-1]
            # pts1 = pts1 * warp_factor[:, :W1 * H1].view(-1, H1, W1, 1)
            # if pts2 is not None:
            #     H2, W2 = pts2.shape[1:-1]
            #     pts2 = pts2 * warp_factor[:, W1 * H1:].view(-1, H2, W2, 1)
            # all_dis = log_dis  # this is their true distance afterwards
        else:
            raise ValueError(f'bad {dis_mode=}')

        norm_factor = all_dis.sum(dim=1) / (nnz_all + 1e-8)
    else:
        raise NotImplementedError
        # # gather all points together (joint normalization)
        # nan_pts1 = invalid_to_nans(pts1, valid1, ndim=3)
        # nan_pts2 = invalid_to_nans(pts2, valid2, ndim=3) if pts2 is not None else None
        # all_pts = torch.cat((nan_pts1, nan_pts2), dim=1) if pts2 is not None else nan_pts1

        # # compute distance to origin
        # all_dis = all_pts.norm(dim=-1)

        # if norm_mode == 'avg':
        #     norm_factor = all_dis.nanmean(dim=1)
        # elif norm_mode == 'median':
        #     norm_factor = all_dis.nanmedian(dim=1).values.detach()
        # elif norm_mode == 'sqrt':
        #     norm_factor = all_dis.sqrt().nanmean(dim=1)**2
        # else:
        #     raise ValueError(f'bad {norm_mode=}')

    norm_factor = norm_factor.clip(min=1e-8)
    # while norm_factor.ndim < pts.ndim:
    #     norm_factor.unsqueeze_(-1)

    res = all_pts / norm_factor
    # if pts2 is not None:
    #     res = (res, pts2 / norm_factor)
    # if ret_factor:
    #     res = res + (norm_factor,)
    if ret_factor:
        return res, norm_factor
    return res


def normalize_pointcloud_all_marigold(pts_all, valid_all, norm_mode='avg_dis', ret_factor=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
        pts_all = [pts1, pts2, ...]
        pts1 [1, h ,w, 3]
        valid [1, h, w]
    """

    cated_pts = torch.cat(pts_all, dim=0) # [B, H, W, 3]
    cated_valid = torch.cat(valid_all, dim=0)
    cated_valid = cated_valid.unsqueeze(-1).repeat(1,1,1,3) # [B, H, W, 1]
    valid_pts = cated_pts[cated_valid].reshape(-1, 3)
    sorted, indices = torch.sort(valid_pts, dim=0)

    total_pts = sorted.shape[0]
    lower = int(total_pts * 0.02)
    upper = int(total_pts * 0.98)
    lower_bound = sorted[lower].reshape(1, 1, 1, 3)
    upper_bound = sorted[upper].reshape(1, 1, 1, 3)

    cated_pts = ((cated_pts - lower_bound) / (upper_bound - lower_bound) - 0.5) * 2
    cated_pts = cated_pts * cated_valid # set invalid points to 0
    print("lower bound:", lower_bound)
    print("upper bound:", upper_bound)

    return cated_pts


    norm_mode, dis_mode = norm_mode.split('_')
    nan_pts_all, nnz_all = [], 0
    for pts, valid in zip(pts_all, valid_all):
        assert pts.ndim >= 3 and pts.shape[-1] == 3
        nan_pts, nnz = invalid_to_zeros(pts, valid, ndim=3)
        nan_pts_all.append(nan_pts)
        nnz_all += nnz
    # import pdb;pdb.set_trace()
    all_pts = torch.cat(nan_pts_all, dim=0).reshape(1, -1, 3)

    all_dis = all_pts.norm(dim=-1)

    norm_factor = all_dis.sum(dim=1) / (nnz_all + 1e-8)

    norm_factor = norm_factor.clip(min=1e-8)

    res = all_pts / norm_factor
    if ret_factor:
        return res, norm_factor
    return res


def normalize_pointcloud_all_inverse_length(pts_all, valid_all, norm_mode='avg_dis', ret_factor=False, verbose=False, fix_lower_upper=False, inverse=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
        pts_all = [pts1, pts2, ...]
        pts1 [1, h ,w, 3]
        valid [1, h, w]
    """
    cated_pts = torch.cat(pts_all, dim=0) # [B, H, W, 3]
    cated_valid = torch.cat(valid_all, dim=0)
    if verbose:
        print('before normalization datapoint: ', cated_pts[0, 0, 0])
        print('before normalization data range', cated_pts.min(), cated_pts.max())
        print('before normalization valid datapoint', cated_valid[0, 0, 0])
    cated_valid = cated_valid.unsqueeze(-1) # [B, H, W, 1]
    all_dis = cated_pts.norm(dim=-1, keepdim=True) # [B, H, W, 1]

    normed_pts = cated_pts / (all_dis + 1e-8)

    # TODO: we can also try to inverse all the point 
    
    if inverse:
        normal_factor = (1.0 / (all_dis + 1e-8)) # [B, H, W, 1]
    else:
        normal_factor = (all_dis + 1e-8)

    # rescaled_normed_pts = (1.0 / (all_dis + 1e-8)) * normed_pts 


    valid_dis = normal_factor[cated_valid].reshape(-1, 1)

    # valid_pts = cated_pts[cated_valid].reshape(-1, 3)
    sorted, indices = torch.sort(valid_dis, dim=0)

    total_pts = sorted.shape[0]
    lower = int(total_pts * 0.02)
    upper = int(total_pts * 0.98)

    # lower = int(total_pts * 0.00)
    # upper = int(total_pts * 1.00) - 1

    if not fix_lower_upper:
        lower_bound = sorted[lower].reshape(1, 1, 1, 1) - 0.01
        upper_bound = sorted[upper].reshape(1, 1, 1, 1) + 0.01
    else:
        lower_bound = 0.00
        upper_bound = 0.25

    # normaled_normal_factor = ((normal_factor - lower_bound) / (upper_bound - lower_bound) - 0.5) * 2
    normaled_normal_factor = (normal_factor - lower_bound) / (upper_bound - lower_bound) # not exactly 0 to 1
    cated_pts = normaled_normal_factor * normed_pts
    cated_pts = cated_pts * cated_valid # set invalid points to 0
    # print("lower bound:", lower_bound)
    # print("upper bound:", upper_bound)
    if verbose:
        print('after normalization data range', cated_pts.min(), cated_pts.max())
        print('after normalization datapoint: ', cated_pts[0, 0, 0])
    return cated_pts


def normalize_pointcloud_bbox(pts_all, valid_all, norm_mode='avg_dis', ret_factor=False, verbose=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
        pts_all = [pts1, pts2, ...]
        pts1 [1, h ,w, 3]
        valid [1, h, w]
    """
    # pay attention here B is video length
    cated_pts = torch.cat(pts_all, dim=0) # [B, H, W, 3]
    B, H, W, _ = cated_pts.shape
    cated_valid = torch.cat(valid_all, dim=0)
    if verbose:
        print('before normalization datapoint: ', cated_pts[0, 0, 0])
        print('before normalization data range', cated_pts.min(), cated_pts.max())
        print('before normalization valid datapoint', cated_valid[0, 0, 0])
    cated_valid = cated_valid.unsqueeze(-1) # [B, H, W, 1]
    cated_valid_pts = cated_valid.repeat(1, 1, 1, 3)
    mean_pts = cated_pts[cated_valid_pts].reshape(-1, 3).mean(dim=0) # [3]
    # here we only consider centering the z axis
    mean_pts[0] = 0
    mean_pts[1] = 0
    cated_pts = cated_pts - mean_pts # TODO this is wrong


    # all_dis = cated_pts.norm(dim=-1, keepdim=True) # [B, H, W, 1]
    all_dis = torch.abs(cated_pts)
    all_dis = all_dis[cated_valid_pts].reshape(-1)
    sorted, indices = torch.sort(all_dis, dim=0)
    total_pts = sorted.shape[0]
    lower = int(total_pts * 0.02)
    upper = int(total_pts * 0.98)
    lower_bound = sorted[lower].reshape(1, 1, 1, 1) - 0.01
    upper_bound = sorted[upper].reshape(1, 1, 1, 1) + 0.01
    normed_pts = cated_pts / (upper_bound)

    if verbose:
        print('after normalization data range', normed_pts.min(), normed_pts.max())
        print('after normalization datapoint: ', normed_pts[0, 0, 0])

    return normed_pts


def cart_to_polar(xyz):

    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]  

    r = torch.norm(torch.stack([x, y, z], dim=-1), dim=-1)  

    theta = torch.atan2(y, x) # [-pi, pi]

    phi = torch.arccos(z / r) # [0, pi]

    return r, theta, phi

def polar_to_cart(r, theta, phi):
    
    x = r * torch.sin(phi) * torch.cos(theta)

    y = r * torch.sin(phi) * torch.sin(theta)

    z = r * torch.cos(phi)

    return torch.stack([x, y, z], dim=-1)


def normalize_pointcloud_inverse_polar(pts_all, valid_all, norm_mode='avg_dis', ret_factor=False, verbose=False, fix_lower_upper=False, inverse=False, upper_bound_ratio=0.98, upper_bound_threshold=2.0, clamp_min=-2.0, clamp_max=2.0, mode=None):
    # pay attention here B is video length
    cated_pts = torch.cat(pts_all, dim=0) # [B, H, W, 3]
    B, H, W, _ = cated_pts.shape
    cated_valid = torch.cat(valid_all, dim=0)
    if verbose:
        print('before normalization datapoint: ', cated_pts[0, 0, 0])
        print('before normalization data range', cated_pts.min(), cated_pts.max())
        print('before normalization valid datapoint', cated_valid[0, 0, 0])
    cated_valid = cated_valid.unsqueeze(-1) # [B, H, W, 1]
    cated_valid_pts = cated_valid.repeat(1, 1, 1, 3)
    valid_pts = cated_pts[cated_valid_pts].reshape(-1, 3) # TODO here set invalid far away point to infinite

    r, theta, phi = cart_to_polar(valid_pts)

    inv_r = 1.0 / (r + 1e-8)

    # valid_pts_z = valid_pts[:, 2]
    sorted, indices = torch.sort(inv_r, dim=0)
    total_pts = sorted.shape[0]
    polar_point = torch.zeros_like(cated_pts)
    if total_pts > 0:
        lower = int(total_pts * 0.02)
        upper = int(total_pts * upper_bound_ratio)
        lower_bound = sorted[lower] - 0.01
        upper_bound = sorted[upper] + 0.01
        if upper_bound > upper_bound_threshold:
            upper_bound = upper_bound_threshold
        normed_inv_r = (inv_r - 0.0) / (upper_bound - 0.0)
        normed_inv_r =( normed_inv_r - 0.5 ) * 2.0

        theta = theta / (np.pi)
        phi = ((phi / (np.pi)) - 0.5) * 2
        polar_point[cated_valid_pts] = torch.stack([normed_inv_r, theta, phi], dim=-1).reshape(-1)

    # set invalid points to 1
    polar_point[~cated_valid_pts] = 1.05
    
    if verbose:
        print('after normalization data range', polar_point.min(), polar_point.max())
        print('after normalization datapoint: ', polar_point[0, 0, 0])
    
    # check if cated_pts has inf or nan
    # print("check if cated_pts has inf or nan")
    if torch.isnan(polar_point).any() or torch.isinf(polar_point).any():
        print("polar_point has inf or nan")
        print("polar_point has inf or nan")


    # print("clip the output to be in the range of -1.1 to 1.1")
    polar_point = polar_point.clamp(clamp_min, clamp_max)

    return polar_point


def cart_to_polar_self(xyz):

    x, z, y = xyz[..., 0], xyz[..., 1], xyz[..., 2]  

    r = torch.norm(torch.stack([x, y, z], dim=-1), dim=-1)  

    theta = torch.atan2(y, x) # [-pi, pi]

    phi = torch.arccos(z / r) # [0, pi]

    return r, theta, phi

def polar_to_cart_self(r, theta, phi):
    
    x = r * torch.sin(phi) * torch.cos(theta)

    z = r * torch.sin(phi) * torch.sin(theta)

    y = r * torch.cos(phi)

    return torch.stack([x, y, z], dim=-1)


def normalize_pointcloud_inverse_polar_self(pts_all, valid_all, large_depth_list, norm_mode='avg_dis', ret_factor=False, verbose=False, fix_lower_upper=False, inverse=False, upper_bound_ratio=0.98, upper_bound_threshold=2.0, clamp_min=-2.0, clamp_max=2.0, mode=None):
    # pay attention here B is video length
    cated_pts = torch.cat(pts_all, dim=0) # [B, H, W, 3]
    B, H, W, _ = cated_pts.shape
    cated_valid = torch.cat(valid_all, dim=0)
    large_depth_list = torch.cat(large_depth_list, dim=0)
    cated_valid = cated_valid | large_depth_list
    if verbose:
        print('before normalization datapoint: ', cated_pts[0, 0, 0])
        print('before normalization data range', cated_pts.min(), cated_pts.max())
        print('before normalization valid datapoint', cated_valid[0, 0, 0])
    cated_valid = cated_valid.unsqueeze(-1) # [B, H, W, 1]
    cated_valid_pts = cated_valid.repeat(1, 1, 1, 3)
    valid_pts = cated_pts[cated_valid_pts].reshape(-1, 3) # TODO here set invalid far away point to infinite

    r, theta, phi = cart_to_polar_self(valid_pts)

    inv_r = 1.0 / (r + 1e-8)

    # valid_pts_z = valid_pts[:, 2]
    sorted, indices = torch.sort(inv_r, dim=0)
    total_pts = sorted.shape[0]
    polar_point = torch.zeros_like(cated_pts)
    if total_pts > 0:
        lower = int(total_pts * 0.02)
        upper = int(total_pts * upper_bound_ratio)
        lower_bound = sorted[lower] - 0.01
        upper_bound = sorted[upper] + 0.01
        if upper_bound > upper_bound_threshold:
            upper_bound = upper_bound_threshold
        normed_inv_r = (inv_r - 0.0) / (upper_bound - 0.0)
        normed_inv_r =( normed_inv_r - 0.5 ) * 2.0

        theta = theta / (np.pi)
        phi = ((phi / (np.pi)) - 0.5) * 2
        polar_point[cated_valid_pts] = torch.stack([normed_inv_r, theta, phi], dim=-1).reshape(-1)

    # set invalid points to 1
    polar_point[~cated_valid_pts] = 1.05
    
    if verbose:
        print('after normalization data range', polar_point.min(), polar_point.max())
        print('after normalization datapoint: ', polar_point[0, 0, 0])
    
    # check if cated_pts has inf or nan
    # print("check if cated_pts has inf or nan")
    if torch.isnan(polar_point).any() or torch.isinf(polar_point).any():
        print("polar_point has inf or nan")
        print("polar_point has inf or nan")


    # print("clip the output to be in the range of -1.1 to 1.1")
    polar_point = polar_point.clamp(clamp_min, clamp_max)

    return polar_point



def normalize_inverse_depth_bbox2(pts_all, valid_all, norm_mode='avg_dis', ret_factor=False, verbose=False, alpha=1.0, beta=1.0, lower_bound_ratio=0.02, upper_bound_ratio=0.98, clamp_min=-1.1, clamp_max=1.1, mode=None, return_st=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
        pts_all = [pts1, pts2, ...]
        pts1 [1, h ,w, 3]
        valid [1, h, w]
    """
    # pay attention here B is video length
    cated_pts = torch.cat(pts_all, dim=0) # [B, H, W, 1]
    B, H, W, _ = cated_pts.shape
    cated_valid = torch.cat(valid_all, dim=0)
    if verbose:
        print('before normalization datapoint: ', cated_pts[0, 0, 0])
        print('before normalization data range', cated_pts.min(), cated_pts.max())
        print('before normalization valid datapoint', cated_valid[0, 0, 0])
    cated_valid = cated_valid.unsqueeze(-1) # [B, H, W, 1]
    cated_pts = 1.0 / (cated_pts + 1e-8) # disparity
    cated_valid_pts = cated_valid.repeat(1, 1, 1, 1)
    valid_pts = cated_pts[cated_valid_pts].reshape(-1, 1)
    valid_pts_z = valid_pts[:, 0]
    sorted, indices = torch.sort(valid_pts_z, dim=0)

    total_pts = sorted.shape[0]
    s_ret=1
    t_ret=0
    if total_pts > 0:
        lower = int(total_pts * lower_bound_ratio)
        upper = int(total_pts * upper_bound_ratio)
        t_ret =  sorted[lower] - 0.01
        s_ret =  sorted[upper] - sorted[lower] + 0.02
        lower_bound = sorted[lower].reshape(1, 1, 1, 1) - 0.01
        upper_bound = sorted[upper].reshape(1, 1, 1, 1) + 0.01
        s = (upper_bound - lower_bound)
        lower_bound = lower_bound.repeat(1, 1, 1, 1)
        # lower_bound[0,0,0,0]= 0
        # lower_bound[0,0,0,1]= 0
        cated_pts = (cated_pts - lower_bound) / s
        cated_pts = (cated_pts * 2) - 1
        # alpha = 1.0
        # beta = 1.0
        # cated_pts[..., 0] = cated_pts[..., 0] * alpha
        # cated_pts[..., 1] = cated_pts[..., 1] * beta
        # print("min:", cated_pts.reshape(-1, 3).min(dim=0))
        # print("max:", cated_pts.reshape(-1, 3).max(dim=0))

    
    # set invalid points to 1
    # set invalid points to -1.05
    cated_pts[~cated_valid_pts] = -1.05

    # check if cated_pts has inf or nan
    # print("check if cated_pts has inf or nan")
    if torch.isnan(cated_pts).any() or torch.isinf(cated_pts).any():
        print("cated_pts has inf or nan")
        print("cated_pts has inf or nan")


    # print("clip the output to be in the range of -1.1 to 1.1")
    cated_pts = cated_pts.clamp(clamp_min, clamp_max)

    # if True:
    #     denormalized_pts = cated_pts.clone()
    #     denormalized_pts[..., 0] = denormalized_pts[..., 0] / alpha
    #     denormalized_pts[..., 1] = denormalized_pts[..., 1] / beta
    #     denormalized_pts[..., 2] = (denormalized_pts[..., 2] + 1) / 2


    if return_st:
        return cated_pts, s_ret, t_ret

    return cated_pts




def normalize_pointcloud_bbox2(pts_all, valid_all, norm_mode='avg_dis', ret_factor=False, verbose=False, alpha=1.0, beta=1.0, lower_bound_ratio=0.02, upper_bound_ratio=0.98, clamp_min=-2.0, clamp_max=2.0, mode=None, return_st=False):
    """ renorm pointmaps pts1, pts2 with norm_mode
        pts_all = [pts1, pts2, ...]
        pts1 [1, h ,w, 3]
        valid [1, h, w]
    """
    # pay attention here B is video length
    cated_pts = torch.cat(pts_all, dim=0) # [B, H, W, 3]
    B, H, W, _ = cated_pts.shape
    cated_valid = torch.cat(valid_all, dim=0)
    if verbose:
        print('before normalization datapoint: ', cated_pts[0, 0, 0])
        print('before normalization data range', cated_pts.min(), cated_pts.max())
        print('before normalization valid datapoint', cated_valid[0, 0, 0])
    cated_valid = cated_valid.unsqueeze(-1) # [B, H, W, 1]
    cated_valid_pts = cated_valid.repeat(1, 1, 1, 3)
    valid_pts = cated_pts[cated_valid_pts].reshape(-1, 3)
    valid_pts_z = valid_pts[:, 2]
    sorted, indices = torch.sort(valid_pts_z, dim=0)

    total_pts = sorted.shape[0]
    s_ret=1
    t_ret=0
    if total_pts > 0:
        lower = int(total_pts * lower_bound_ratio)
        upper = int(total_pts * upper_bound_ratio)
        t_ret =  sorted[lower] - 0.01
        s_ret =  sorted[upper] - sorted[lower] + 0.02
        lower_bound = sorted[lower].reshape(1, 1, 1, 1) - 0.01
        upper_bound = sorted[upper].reshape(1, 1, 1, 1) + 0.01
        s = (upper_bound - lower_bound)
        lower_bound = lower_bound.repeat(1, 1, 1, 3)
        lower_bound[0,0,0,0]= 0
        lower_bound[0,0,0,1]= 0
        cated_pts = (cated_pts - lower_bound) / s
        cated_pts[..., 2] = (cated_pts[..., 2] * 2) - 1
        # alpha = 1.0
        # beta = 1.0
        cated_pts[..., 0] = cated_pts[..., 0] * alpha
        cated_pts[..., 1] = cated_pts[..., 1] * beta
        # print("min:", cated_pts.reshape(-1, 3).min(dim=0))
        # print("max:", cated_pts.reshape(-1, 3).max(dim=0))

    
    # set invalid points to 1
    cated_pts[~cated_valid_pts] = 1.05

    # check if cated_pts has inf or nan
    # print("check if cated_pts has inf or nan")
    if torch.isnan(cated_pts).any() or torch.isinf(cated_pts).any():
        print("cated_pts has inf or nan")
        print("cated_pts has inf or nan")


    # print("clip the output to be in the range of -1.1 to 1.1")
    cated_pts = cated_pts.clamp(clamp_min, clamp_max)

    # if True:
    #     denormalized_pts = cated_pts.clone()
    #     denormalized_pts[..., 0] = denormalized_pts[..., 0] / alpha
    #     denormalized_pts[..., 1] = denormalized_pts[..., 1] / beta
    #     denormalized_pts[..., 2] = (denormalized_pts[..., 2] + 1) / 2


    if verbose:
        print('after normalization data range', normed_pts.min(), normed_pts.max())
        print('after normalization datapoint: ', normed_pts[0, 0, 0])
    if return_st:
        return cated_pts, s_ret, t_ret

    return cated_pts


def normalize_pointcloud_bbox2_center(pts_all, valid_all, norm_mode='avg_dis', ret_factor=False, verbose=False, alpha=1.0, beta=1.0, lower_bound_ratio=0.02, upper_bound_ratio=0.98, mode=None):
    """ renorm pointmaps pts1, pts2 with norm_mode
        pts_all = [pts1, pts2, ...]
        pts1 [1, h ,w, 3]
        valid [1, h, w]
    """
    # pay attention here B is video length
    cated_pts = torch.cat(pts_all, dim=0) # [B, H, W, 3]
    B, H, W, _ = cated_pts.shape
    cated_valid = torch.cat(valid_all, dim=0)
    if verbose:
        print('before normalization datapoint: ', cated_pts[0, 0, 0])
        print('before normalization data range', cated_pts.min(), cated_pts.max())
        print('before normalization valid datapoint', cated_valid[0, 0, 0])
    cated_valid = cated_valid.unsqueeze(-1) # [B, H, W, 1]
    cated_valid_pts = cated_valid.repeat(1, 1, 1, 3)
    valid_pts = cated_pts[cated_valid_pts].reshape(-1, 3)
    
    sorted, indices = torch.sort(valid_pts, dim=0)

    total_pts = sorted.shape[0]
    lower = int(total_pts * lower_bound_ratio)
    upper = int(total_pts * upper_bound_ratio)
    lower_bound = sorted[lower].reshape(1, 1, 1, 3) - 0.01
    upper_bound = sorted[upper].reshape(1, 1, 1, 3) + 0.01
    s = (upper_bound[..., [2]] - lower_bound[..., [2]])
    cated_pts = (cated_pts - lower_bound) / s
    cated_pts = (cated_pts * 2) - 1
    cated_pts[..., 0] = cated_pts[..., 0] * alpha
    cated_pts[..., 1] = cated_pts[..., 1] * beta
    cated_pts[~cated_valid_pts] = 1
    


    if verbose:
        print('after normalization data range', normed_pts.min(), normed_pts.max())
        print('after normalization datapoint: ', normed_pts[0, 0, 0])

    return cated_pts


def normalize_pointcloud_bbox2_layer(pts_all, valid_all, norm_mode='avg_dis', ret_factor=False, verbose=False, alpha=1.0, beta=1.0, lower_bound_ratio=0.02, upper_bound_ratio=0.98):
    """ renorm pointmaps pts1, pts2 with norm_mode
        pts_all = [pts1, pts2, ...]
        pts1 [1, h ,w, 3]
        valid [1, h, w]
    """
    # pay attention here B is video length
    cated_pts = torch.cat(pts_all, dim=0) # [B, H, W, 3]
    B, H, W, _ = cated_pts.shape
    cated_valid = torch.cat(valid_all, dim=0)
    if verbose:
        print('before normalization datapoint: ', cated_pts[0, 0, 0])
        print('before normalization data range', cated_pts.min(), cated_pts.max())
        print('before normalization valid datapoint', cated_valid[0, 0, 0])
    cated_valid = cated_valid.unsqueeze(-1) # [B, H, W, 1]
    cated_valid_pts = cated_valid.repeat(1, 1, 1, 3)
    valid_pts = cated_pts[cated_valid_pts].reshape(-1, 3)
    valid_pts_z = valid_pts[:, 2]
    sorted, indices = torch.sort(valid_pts_z, dim=0)

    total_pts = sorted.shape[0]
    lower = int(total_pts * lower_bound_ratio)
    upper = int(total_pts * upper_bound_ratio)
    lower_bound = sorted[lower].reshape(1, 1, 1, 1) - 0.01
    upper_bound = sorted[upper].reshape(1, 1, 1, 1) + 0.01
    s = (upper_bound - lower_bound)
    lower_bound = lower_bound.repeat(1, 1, 1, 3)
    lower_bound[0,0,0,0]= 0
    lower_bound[0,0,0,1]= 0
    cated_pts = (cated_pts - lower_bound) / s

    assert lower_bound_ratio == 0.00
    z_075_100 = cated_pts[..., [2]] > 0.75
    z_050_075 = torch.logical_and(cated_pts[..., [2]] > 0.50, cated_pts[..., [2]] < 0.75)
    z_025_050 = torch.logical_and(cated_pts[..., [2]] > 0.25, cated_pts[..., [2]] < 0.50)
    z_000_025 = torch.logical_and(cated_pts[..., [2]] > 0.00, cated_pts[..., [2]] < 0.25)
    z_075_100 = z_075_100.repeat(1, 1, 1, 3)
    z_050_075 = z_050_075.repeat(1, 1, 1, 3)
    z_025_050 = z_025_050.repeat(1, 1, 1, 3)
    z_000_025 = z_000_025.repeat(1, 1, 1, 3)
    z_075_100[..., 0] = 0
    z_075_100[..., 1] = 0
    z_050_075[..., 0] = 0
    z_050_075[..., 1] = 0
    z_025_050[..., 0] = 0
    z_025_050[..., 1] = 0
    z_000_025[..., 0] = 0
    z_000_025[..., 1] = 0


    # rescale 
    cated_pts[z_075_100] = cated_pts[z_075_100] * 1.0 + 2.75
    cated_pts[z_050_075] = cated_pts[z_050_075] * 2.0 + 2
    cated_pts[z_025_050] = cated_pts[z_025_050] * 4.0 + 1
    cated_pts[z_000_025] = cated_pts[z_000_025] * 8.0
    cated_pts[..., 2] = cated_pts[..., 2] / 3.75

    # # for x and y
    # x_pos_075_100 = cated_pts[..., [0]] > 0.75
    # x_pos_050_075 = torch.logical_and(cated_pts[..., [0]] > 0.50, cated_pts[..., [0]] < 0.75)
    # x_pos_025_050 = torch.logical_and(cated_pts[..., [0]] > 0.25, cated_pts[..., [0]] < 0.50)
    # x_pos_000_025 = torch.logical_and(cated_pts[..., [0]] > 0.00, cated_pts[..., [0]] < 0.25)






    cated_pts[..., 2] = (cated_pts[..., 2] * 2) - 1
    # alpha = 1.0
    # beta = 1.0
    cated_pts[..., 0] = cated_pts[..., 0] * alpha
    cated_pts[..., 1] = cated_pts[..., 1] * beta
    # print("min:", cated_pts.reshape(-1, 3).min(dim=0))
    # print("max:", cated_pts.reshape(-1, 3).max(dim=0))

    # set invalid points to 1
    cated_pts[~cated_valid_pts] = 1


    # print("clip the output to be in the range of -1.1 to 1.1")
    # cated_pts = cated_pts.clamp(-1.1, 1.1)

    # if True:
    #     denormalized_pts = cated_pts.clone()
    #     denormalized_pts[..., 0] = denormalized_pts[..., 0] / alpha
    #     denormalized_pts[..., 1] = denormalized_pts[..., 1] / beta
    #     denormalized_pts[..., 2] = (denormalized_pts[..., 2] + 1) / 2


    if verbose:
        print('after normalization data range', normed_pts.min(), normed_pts.max())
        print('after normalization datapoint: ', normed_pts[0, 0, 0])

    return cated_pts



@torch.no_grad()
def get_joint_pointcloud_depth(z1, z2, valid_mask1, valid_mask2=None, quantile=0.5):
    # set invalid points to NaN
    _z1 = invalid_to_nans(z1, valid_mask1).reshape(len(z1), -1)
    _z2 = invalid_to_nans(z2, valid_mask2).reshape(len(z2), -1) if z2 is not None else None
    _z = torch.cat((_z1, _z2), dim=-1) if z2 is not None else _z1

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(_z, dim=-1).values
    else:
        shift_z = torch.nanquantile(_z, quantile, dim=-1)
    return shift_z  # (B,)


@torch.no_grad()
def get_joint_pointcloud_center_scale(pts1, pts2, valid_mask1=None, valid_mask2=None, z_only=False, center=True):
    # set invalid points to NaN
    _pts1 = invalid_to_nans(pts1, valid_mask1).reshape(len(pts1), -1, 3)
    _pts2 = invalid_to_nans(pts2, valid_mask2).reshape(len(pts2), -1, 3) if pts2 is not None else None
    _pts = torch.cat((_pts1, _pts2), dim=1) if pts2 is not None else _pts1

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


def find_reciprocal_matches(P1, P2):
    """
    returns 3 values:
    1 - reciprocal_in_P2: a boolean array of size P2.shape[0], a "True" value indicates a match
    2 - nn2_in_P1: a int array of size P2.shape[0], it contains the indexes of the closest points in P1
    3 - reciprocal_in_P2.sum(): the number of matches
    """
    tree1 = KDTree(P1)
    tree2 = KDTree(P2)

    _, nn1_in_P2 = tree2.query(P1, workers=8)
    _, nn2_in_P1 = tree1.query(P2, workers=8)

    reciprocal_in_P1 = (nn2_in_P1[nn1_in_P2] == np.arange(len(nn1_in_P2)))
    reciprocal_in_P2 = (nn1_in_P2[nn2_in_P1] == np.arange(len(nn2_in_P1)))
    assert reciprocal_in_P1.sum() == reciprocal_in_P2.sum()
    return reciprocal_in_P2, nn2_in_P1, reciprocal_in_P2.sum()


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist
    return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))
