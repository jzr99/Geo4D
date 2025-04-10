from __future__ import annotations

import dataclasses
import os
import json
from pathlib import Path
from typing import Tuple, cast

import imageio.v3 as iio
import liblzfse
import numpy as np
import numpy as onp
import numpy.typing as onpt
import skimage.transform
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree

class Record3dLoader_Customized:
    """Helper for loading frames for Record3D captures."""

    def __init__(self, data_dir: Path, conf_threshold: float = 1.0, foreground_conf_threshold: float = 0.1, no_mask: bool = False, xyzw=True, init_conf=False):

        # Read metadata.
        intrinsics_path = data_dir / "pred_intrinsics.txt"
        intrinsics = np.loadtxt(intrinsics_path)

        self.K: onp.ndarray = np.array(intrinsics, np.float32).reshape(-1, 3, 3)
        fps = 30

        self.init_conf = init_conf

        poses_path = data_dir / "pred_traj.txt"
        poses = np.loadtxt(poses_path)
        self.T_world_cameras: onp.ndarray = np.array(poses, np.float32)
        self.T_world_cameras = np.concatenate(
            [
                # Convert TUM pose to SE3 pose
                Rotation.from_quat(self.T_world_cameras[:, 4:]).as_matrix() if not xyzw
                else Rotation.from_quat(np.concatenate([self.T_world_cameras[:, 5:], self.T_world_cameras[:, 4:5]], -1)).as_matrix(),
                self.T_world_cameras[:, 1:4, None],
            ],
            -1,
        )
        self.T_world_cameras = self.T_world_cameras.astype(np.float32)

        # Convert to homogeneous transformation matrices (ensure shape is (N, 4, 4))
        num_frames = self.T_world_cameras.shape[0]
        ones = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (num_frames, 1, 1))
        self.T_world_cameras = np.concatenate([self.T_world_cameras, ones], axis=1)

        self.fps = fps
        self.conf_threshold = conf_threshold
        self.foreground_conf_threshold = foreground_conf_threshold
        self.no_mask = no_mask

        # Read frames.
        self.rgb_paths = sorted(data_dir.glob("frame_*.png"), key=lambda p: int(p.stem.split("_")[-1]))
        self.depth_paths = sorted(data_dir.glob("frame_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
        if init_conf:
            self.init_conf_paths = sorted(data_dir.glob("init_conf_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
        else:
            self.init_conf_paths = []
        self.conf_paths = sorted(data_dir.glob("conf_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
        self.mask_paths = sorted(data_dir.glob("enlarged_dynamic_mask_*.png"), key=lambda p: int(p.stem.split("_")[-1]))

        # Remove the last frame since it does not have a ground truth dynamic mask
        self.rgb_paths = self.rgb_paths[:-1]

        # Align all camera poses by the first frame
        T0 = self.T_world_cameras[len(self.T_world_cameras) // 2]  # First camera pose (4x4 matrix)
        T0_inv = np.linalg.inv(T0)    # Inverse of the first camera pose

        # Apply T0_inv to all camera poses
        self.T_world_cameras = np.matmul(T0_inv[np.newaxis, :, :], self.T_world_cameras)


    def num_frames(self) -> int:
        return len(self.rgb_paths)

    def get_frame(self, index: int) -> Record3dFrame:

        # Read depth.
        depth = np.load(self.depth_paths[index])
        depth: onp.NDArray[onp.float32] = depth
        
        # Check if conf file exists, otherwise initialize with ones
        if len(self.conf_paths) == 0:
            conf = np.ones_like(depth, dtype=onp.float32)
        else:
            conf_path = self.conf_paths[index]
            if os.path.exists(conf_path):
                conf = np.load(conf_path)
                conf: onpt.NDArray[onp.float32] = conf
                # Clip confidence to avoid negative values
                conf = np.clip(conf, 0.0001, 99999)
            else:
                conf = np.ones_like(depth, dtype=onp.float32)

        # Check if init conf file exists, otherwise initialize with ones
        if len(self.init_conf_paths) == 0:  # If init conf is not available, use conf
            init_conf = conf
        else:
            init_conf_path = self.init_conf_paths[index]
            if os.path.exists(init_conf_path):
                init_conf = np.load(init_conf_path)
                init_conf: onpt.NDArray[onp.float32] = init_conf
                # Clip confidence to avoid negative values
                init_conf = np.clip(init_conf, 0.0001, 99999)
            else:
                init_conf = np.ones_like(depth, dtype=onp.float32)
        
        # Check if mask file exists, otherwise initialize with zeros
        if len(self.mask_paths) == 0:
            mask = np.ones_like(depth, dtype=onp.bool_)
        else:
            mask_path = self.mask_paths[index]
            if os.path.exists(mask_path):
                mask = iio.imread(mask_path) > 0
                mask: onpt.NDArray[onp.bool_] = mask
            else:
                mask = np.ones_like(depth, dtype=onp.bool_)

        if self.no_mask:
            mask = np.ones_like(mask).astype(np.bool_)

        # Read RGB.
        rgb = iio.imread(self.rgb_paths[index])
        # if 4 channels, remove the alpha channel
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        return Record3dFrame(
            K=self.K[index],
            rgb=rgb,
            depth=depth,
            mask=mask,
            conf=conf,
            init_conf=init_conf,
            T_world_camera=self.T_world_cameras[index],
            conf_threshold=self.conf_threshold,
            foreground_conf_threshold=self.foreground_conf_threshold,
        )


@dataclasses.dataclass
class Record3dFrame:
    """A single frame from a Record3D capture."""

    K: onpt.NDArray[onp.float32]
    rgb: onpt.NDArray[onp.uint8]
    depth: onpt.NDArray[onp.float32]
    mask: onpt.NDArray[onp.bool_]
    conf: onpt.NDArray[onp.float32]
    init_conf: onpt.NDArray[onp.float32]
    T_world_camera: onpt.NDArray[onp.float32]
    conf_threshold: float = 1.0
    foreground_conf_threshold: float = 0.1

    def get_point_cloud(
        self, downsample_factor: int = 1, bg_downsample_factor: int = 1,
    ) -> Tuple[onpt.NDArray[onp.float32], onpt.NDArray[onp.uint8], onpt.NDArray[onp.float32], onpt.NDArray[onp.uint8]]:
        rgb = self.rgb[::downsample_factor, ::downsample_factor]
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            onpt.NDArray[onp.bool_],
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        )
        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        img_wh = rgb.shape[:2][::-1]

        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), 2) + 0.5
        )
        grid = grid * downsample_factor
        conf_mask = self.conf > self.conf_threshold
        if self.init_conf is not None:
            fg_conf_mask = self.init_conf > self.foreground_conf_threshold
        else:
            fg_conf_mask = self.conf > self.foreground_conf_threshold
        # reshape the conf mask to the shape of the depth
        conf_mask = skimage.transform.resize(conf_mask, depth.shape, order=0)
        fg_conf_mask = skimage.transform.resize(fg_conf_mask, depth.shape, order=0)

        # Foreground points
        homo_grid = np.pad(grid[fg_conf_mask & mask], ((0, 0), (0, 1)), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], local_dirs)
        points = (T_world_camera[:3, 3] + dirs * depth[fg_conf_mask & mask, None]).astype(np.float32)
        point_colors = rgb[fg_conf_mask & mask]

        # Background points
        bg_homo_grid = np.pad(grid[conf_mask & ~mask], ((0, 0), (0, 1)), constant_values=1)
        bg_local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), bg_homo_grid)
        bg_dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], bg_local_dirs)
        bg_points = (T_world_camera[:3, 3] + bg_dirs * depth[conf_mask & ~mask, None]).astype(np.float32)
        bg_point_colors = rgb[conf_mask & ~mask]

        if bg_downsample_factor > 1 and bg_points.shape[0] > 0:
            indices = np.random.choice(
                bg_points.shape[0],
                size=bg_points.shape[0] // bg_downsample_factor,
                replace=False
            )
            bg_points = bg_points[indices]
            bg_point_colors = bg_point_colors[indices]

        return points, point_colors, bg_points, bg_point_colors
