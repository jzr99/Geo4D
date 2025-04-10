import cv2
import ipdb  # noqa: F401
import numpy as np
import torch
from pytorch3d.renderer import PerspectiveCameras, RayBundle

# from ray_diffusion.utils.normalize import intersect_skew_lines_high_dim
from utils.normalize import intersect_skew_lines_high_dim

from einops import rearrange

import torch.nn.functional as F


class Rays(object):
    def __init__(
        self,
        rays=None,
        origins=None,
        directions=None,
        moments=None,
        is_plucker=False,
        moments_rescale=1.0,
        ndc_coordinates=None,
        crop_parameters=None,
        num_patches_x=16,
        num_patches_y=16,
    ):
        """
        Ray class to keep track of current ray representation.

        Args:
            rays: (..., 6).
            origins: (..., 3).
            directions: (..., 3).
            moments: (..., 3).
            is_plucker: If True, rays are in plucker coordinates (Default: False).
            moments_rescale: Rescale the moment component of the rays by a scalar.
            ndc_coordinates: (..., 2): NDC coordinates of each ray.
        """
        if rays is not None:
            self.rays = rays
            self._is_plucker = is_plucker
        elif origins is not None and directions is not None:
            self.rays = torch.cat((origins, directions), dim=-1)
            self._is_plucker = False
        elif directions is not None and moments is not None:
            self.rays = torch.cat((directions, moments), dim=-1)
            self._is_plucker = True
        else:
            raise Exception("Invalid combination of arguments")

        if moments_rescale != 1.0:
            self.rescale_moments(moments_rescale)

        if ndc_coordinates is not None:
            self.ndc_coordinates = ndc_coordinates
        elif crop_parameters is not None:
            # (..., H, W, 2)
            xy_grid = compute_ndc_coordinates(
                crop_parameters,
                num_patches_x=num_patches_x,
                num_patches_y=num_patches_y,
            )[..., :2]
            xy_grid = xy_grid.reshape(*xy_grid.shape[:-3], -1, 2)
            self.ndc_coordinates = xy_grid
        else:
            self.ndc_coordinates = None

    def __getitem__(self, index):
        return Rays(
            rays=self.rays[index],
            is_plucker=self._is_plucker,
            ndc_coordinates=(
                self.ndc_coordinates[index]
                if self.ndc_coordinates is not None
                else None
            ),
        )

    def to_spatial(self, include_ndc_coordinates=False):
        """
        Converts rays to spatial representation: (..., H * W, 6) --> (..., 6, H, W)

        Returns:
            torch.Tensor: (..., 6, H, W)
        """
        rays = self.to_plucker().rays
        *batch_dims, P, D = rays.shape
        H = W = int(np.sqrt(P))
        assert H * W == P
        rays = torch.transpose(rays, -1, -2)  # (..., 6, H * W)
        rays = rays.reshape(*batch_dims, D, H, W)
        if include_ndc_coordinates:
            ndc_coords = self.ndc_coordinates.transpose(-1, -2)  # (..., 2, H * W)
            ndc_coords = ndc_coords.reshape(*batch_dims, 2, H, W)
            rays = torch.cat((rays, ndc_coords), dim=-3)
        return rays

    def rescale_moments(self, scale):
        """
        Rescale the moment component of the rays by a scalar. Might be desirable since
        moments may come from a very narrow distribution.

        Note that this modifies in place!
        """
        if self.is_plucker:
            self.rays[..., 3:] *= scale
            return self
        else:
            return self.to_plucker().rescale_moments(scale)

    @classmethod
    def from_spatial(cls, rays, moments_rescale=1.0, ndc_coordinates=None):
        """
        Converts rays from spatial representation: (..., 6, H, W) --> (..., H * W, 6)

        Args:
            rays: (..., 6, H, W)

        Returns:
            Rays: (..., H * W, 6)
        """
        *batch_dims, D, H, W = rays.shape
        rays = rays.reshape(*batch_dims, D, H * W)
        rays = torch.transpose(rays, -1, -2)
        return cls(
            rays=rays,
            is_plucker=True,
            moments_rescale=moments_rescale,
            ndc_coordinates=ndc_coordinates,
        )

    def to_point_direction(self, normalize_moment=True):
        """
        Convert to point direction representation <O, D>.

        Returns:
            rays: (..., 6).
        """
        if self._is_plucker:
            direction = torch.nn.functional.normalize(self.rays[..., :3], dim=-1)
            moment = self.rays[..., 3:]
            if normalize_moment:
                c = torch.linalg.norm(direction, dim=-1, keepdim=True)
                moment = moment / c
            points = torch.cross(direction, moment, dim=-1)
            return Rays(
                rays=torch.cat((points, direction), dim=-1),
                is_plucker=False,
                ndc_coordinates=self.ndc_coordinates,
            )
        else:
            return self

    def to_plucker(self):
        """
        Convert to plucker representation <D, OxD>.
        """
        if self.is_plucker:
            return self
        else:
            ray = self.rays.clone()
            ray_origins = ray[..., :3]
            ray_directions = ray[..., 3:]
            # Normalize ray directions to unit vectors
            ray_directions = ray_directions / ray_directions.norm(dim=-1, keepdim=True)
            plucker_normal = torch.cross(ray_origins, ray_directions, dim=-1)
            new_ray = torch.cat([ray_directions, plucker_normal], dim=-1)
            return Rays(
                rays=new_ray, is_plucker=True, ndc_coordinates=self.ndc_coordinates
            )

    def get_directions(self, normalize=True):
        if self.is_plucker:
            directions = self.rays[..., :3]
        else:
            directions = self.rays[..., 3:]
        if normalize:
            directions = torch.nn.functional.normalize(directions, dim=-1)
        return directions

    def get_origins(self):
        if self.is_plucker:
            origins = self.to_point_direction().get_origins()
        else:
            origins = self.rays[..., :3]
        return origins

    def get_moments(self):
        if self.is_plucker:
            moments = self.rays[..., 3:]
        else:
            moments = self.to_plucker().get_moments()
        return moments

    def get_ndc_coordinates(self):
        return self.ndc_coordinates

    @property
    def is_plucker(self):
        return self._is_plucker

    @property
    def device(self):
        return self.rays.device

    def __repr__(self, *args, **kwargs):
        ray_str = self.rays.__repr__(*args, **kwargs)[6:]  # remove "tensor"
        if self._is_plucker:
            return "PluRay" + ray_str
        else:
            return "DirRay" + ray_str

    def to(self, device):
        self.rays = self.rays.to(device)

    def clone(self):
        return Rays(rays=self.rays.clone(), is_plucker=self._is_plucker)

    @property
    def shape(self):
        return self.rays.shape

    def visualize(self):
        directions = torch.nn.functional.normalize(self.get_directions(), dim=-1).cpu()
        moments = torch.nn.functional.normalize(self.get_moments(), dim=-1).cpu()
        return (directions + 1) / 2, (moments + 1) / 2

    def to_ray_bundle(self, length=0.3, recenter=True):
        lengths = torch.ones_like(self.get_origins()[..., :2]) * length
        lengths[..., 0] = 0
        if recenter:
            centers, _ = intersect_skew_lines_high_dim(
                self.get_origins(), self.get_directions()
            )
            centers = centers.unsqueeze(1).repeat(1, lengths.shape[1], 1)
        else:
            centers = self.get_origins()
        return RayBundle(
            origins=centers,
            directions=self.get_directions(),
            lengths=lengths,
            xys=self.get_directions(),
        )


def cameras_to_rays(
    cameras,
    crop_parameters,
    use_half_pix=True,
    use_plucker=True,
    num_patches_x=16,
    num_patches_y=16,
):
    """
    Unprojects rays from camera center to grid on image plane.

    Args:
        cameras: Pytorch3D cameras to unproject. Can be batched.
        crop_parameters: Crop parameters in NDC (cc_x, cc_y, crop_width, scale).
            Shape is (B, 4).
        use_half_pix: If True, use half pixel offset (Default: True).
        use_plucker: If True, return rays in plucker coordinates (Default: False).
        num_patches_x: Number of patches in x direction (Default: 16).
        num_patches_y: Number of patches in y direction (Default: 16).
    """
    unprojected = []
    crop_parameters_list = (
        crop_parameters if crop_parameters is not None else [None for _ in cameras]
    )
    for camera, crop_param in zip(cameras, crop_parameters_list):
        xyd_grid = compute_ndc_coordinates(
            crop_parameters=crop_param,
            use_half_pix=use_half_pix,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
        )

        unprojected.append(
            camera.unproject_points(
                xyd_grid.reshape(-1, 3), world_coordinates=True, from_ndc=True
            )
        )
    unprojected = torch.stack(unprojected, dim=0)  # (N, P, 3)
    origins = cameras.get_camera_center().unsqueeze(1)  # (N, 1, 3)
    origins = origins.repeat(1, num_patches_x * num_patches_y, 1)  # (N, P, 3)
    directions = unprojected - origins
    rays = Rays(
        origins=origins,
        directions=directions,
        crop_parameters=crop_parameters,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
    )
    if use_plucker:
        return rays.to_plucker()
    return rays


def rays_to_cameras(
    rays,
    crop_parameters,
    num_patches_x=16,
    num_patches_y=16,
    use_half_pix=True,
    sampled_ray_idx=None,
    cameras=None,
    focal_length=(3.453,),
    ref_ray=None,
    ret_center=False,
):
    """
    If cameras are provided, will use those intrinsics. Otherwise will use the provided
    focal_length(s). Dataset default is 3.32.

    Args:
        rays (Rays): (N, P, 6)
        crop_parameters (torch.Tensor): (N, 4)
    """
    device = rays.device
    origins = rays.get_origins()
    directions = rays.get_directions()
    camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)

    if ref_ray is not None:
        I_patch_rays = ref_ray.get_directions()
        if len(focal_length) == 1:
            focal_length = focal_length * rays.shape[0]
        I_camera = PerspectiveCameras(focal_length=focal_length, device=device)
    else:
        # Retrieve target rays
        if cameras is None:
            if len(focal_length) == 1:
                focal_length = focal_length * rays.shape[0]
            I_camera = PerspectiveCameras(focal_length=focal_length, device=device)
        else:
            # Use same intrinsics but reset to identity extrinsics.
            I_camera = cameras.clone()
            I_camera.R[:] = torch.eye(3, device=device)
            I_camera.T[:] = torch.zeros(3, device=device)
        I_patch_rays = cameras_to_rays(
            cameras=I_camera,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            use_half_pix=use_half_pix,
            crop_parameters=crop_parameters,
        ).get_directions()

    if sampled_ray_idx is not None:
        I_patch_rays = I_patch_rays[:, sampled_ray_idx]

    # Compute optimal rotation to align rays
    R = torch.zeros_like(I_camera.R)
    for i in range(len(I_camera)):
        R[i] = compute_optimal_rotation_alignment(
            I_patch_rays[i],
            directions[i],
        )

    # Construct and return rotated camera
    cam = I_camera.clone()
    cam.R = R
    cam.T = -torch.matmul(R.transpose(1, 2), camera_centers.unsqueeze(2)).squeeze(2)
    if ret_center:
        return cam, camera_centers
    return cam


# https://www.reddit.com/r/learnmath/comments/v1crd7/linear_algebra_qr_to_ql_decomposition/
def ql_decomposition(A):
    P = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]], device=A.device).float()
    A_tilde = torch.matmul(A, P)
    Q_tilde, R_tilde = torch.linalg.qr(A_tilde)
    Q = torch.matmul(Q_tilde, P)
    L = torch.matmul(torch.matmul(P, R_tilde), P)
    d = torch.diag(L)
    Q[:, 0] *= torch.sign(d[0])
    Q[:, 1] *= torch.sign(d[1])
    Q[:, 2] *= torch.sign(d[2])
    L[0] *= torch.sign(d[0])
    L[1] *= torch.sign(d[1])
    L[2] *= torch.sign(d[2])
    return Q, L


def cameras_from_plucker(raydir, raymoment, ref_raymap=None):
    if len(raydir.shape) == 5:
        # b c t h w
        raydir = rearrange(raydir, "b c t h w -> (b t) h w c")
        raymoment = rearrange(raymoment, "b c t h w -> (b t) h w c")
        if ref_raymap is not None:
            ref_raymap = rearrange(ref_raymap, "b c t h w -> (b t) h w c")

    T, H, W, C = raydir.shape

    min_size = min(H, W)
    if H > W:
        crop = (H - W) // 2
        raydir = raydir[:,crop:-crop,:,:]
        raymoment = raymoment[:,crop:-crop,:,:]
        num_patches_x = W // 1
        num_patches_y = W // 1
        if ref_raymap is not None:
            ref_raymap = ref_raymap[:,crop:-crop,:,:]
    elif W > H:
        crop = (W - H) // 2
        raydir = raydir[:,:,crop:-crop,:]
        raymoment = raymoment[:,:,crop:-crop,:]
        num_patches_x = H // 1
        num_patches_y = H // 1
        if ref_raymap is not None:
            ref_raymap = ref_raymap[:,:,crop:-crop,:]


    # raydir = raydir / raydir[..., 2:3] # convert to raymap on image plane # never to this, this will change the ray direction!!!!!!! god
    raydir_lr = F.interpolate(raydir.permute(0, 3, 1, 2), (num_patches_x, num_patches_y), mode='nearest').permute(0, 2, 3, 1)
    raydir_lr = raydir_lr / torch.norm(raydir_lr, dim=-1, keepdim=True)

    if ref_raymap is not None:
        ref_raymap = F.interpolate(ref_raymap.permute(0, 3, 1, 2), (num_patches_x, num_patches_y), mode='nearest').permute(0, 2, 3, 1)
        ref_raymap = ref_raymap / torch.norm(ref_raymap, dim=-1, keepdim=True)

    raymoment = F.interpolate(raymoment.permute(0, 3, 1, 2), (num_patches_x, num_patches_y), mode='nearest').permute(0, 2, 3, 1)
    
    r = Rays(directions=raydir_lr.reshape(T, -1, C), moments=raymoment.reshape(T, -1, C))
    ref_ray = Rays(directions=raydir_lr.reshape(T, -1, C)[[0]].repeat(T, 1, 1), moments=raymoment.reshape(T, -1, C)[[0]].repeat(T, 1, 1))
    if ref_raymap is not None:
        ref_ray = Rays(directions=ref_raymap.reshape(T, -1, C)[[0]].repeat(T, 1, 1), moments=raymoment.reshape(T, -1, C)[[0]].repeat(T, 1, 1))
    # camera, center = rays_to_cameras_homography(r, None, num_patches_x=num_patches_x, num_patches_y=num_patches_y, reproj_threshold=0.002, ret_center=True)
    # camera, center = rays_to_cameras_homography_refray(r, None, num_patches_x=num_patches_x, num_patches_y=num_patches_y, reproj_threshold=0.002, ret_center=True, ref_ray=ref_ray)
    camera, center = rays_to_cameras(r, None, num_patches_x=num_patches_x, num_patches_y=num_patches_y, ret_center=True, ref_ray=ref_ray)
    return camera, center, r



def rays_to_cameras_homography_refray(
    rays,
    crop_parameters,
    num_patches_x=16,
    num_patches_y=16,
    use_half_pix=True,
    sampled_ray_idx=None,
    reproj_threshold=0.2,
    ret_center=False,
    ref_ray=None,
):
    """
    Args:
        rays (Rays): (N, P, 6)
        crop_parameters (torch.Tensor): (N, 4)
    """
    device = rays.device
    origins = rays.get_origins()
    directions = rays.get_directions()
    camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)

    if ref_ray is None:
        # Retrieve target rays
        I_camera = PerspectiveCameras(focal_length=[1] * rays.shape[0], device=device)
        I_patch_rays = cameras_to_rays(
            cameras=I_camera,
            num_patches_x=num_patches_x,
            num_patches_y=num_patches_y,
            use_half_pix=use_half_pix,
            crop_parameters=crop_parameters,
        ).get_directions()
    else:
        I_patch_rays = ref_ray.get_directions()

    if sampled_ray_idx is not None:
        I_patch_rays = I_patch_rays[:, sampled_ray_idx]

    # Compute optimal rotation to align rays
    Rs = []
    focal_lengths = []
    principal_points = []
    for i in range(rays.shape[-3]):
        R, f, pp = compute_optimal_rotation_intrinsics(
            I_patch_rays[i],
            directions[i],
            reproj_threshold=reproj_threshold,
        )
        Rs.append(R)
        focal_lengths.append(f)
        principal_points.append(pp)

    R = torch.stack(Rs)
    focal_lengths = torch.stack(focal_lengths)
    principal_points = torch.stack(principal_points)
    T = -torch.matmul(R.transpose(1, 2), camera_centers.unsqueeze(2)).squeeze(2)
    if ret_center:
        return PerspectiveCameras(
            R=R,
            T=T,
            focal_length=focal_lengths,
            principal_point=principal_points,
            device=device,
        ), camera_centers
    return PerspectiveCameras(
        R=R,
        T=T,
        focal_length=focal_lengths,
        principal_point=principal_points,
        device=device,
    )




def rays_to_cameras_homography(
    rays,
    crop_parameters,
    num_patches_x=16,
    num_patches_y=16,
    use_half_pix=True,
    sampled_ray_idx=None,
    reproj_threshold=0.2,
    ret_center=False,
):
    """
    Args:
        rays (Rays): (N, P, 6)
        crop_parameters (torch.Tensor): (N, 4)
    """
    device = rays.device
    origins = rays.get_origins()
    directions = rays.get_directions()
    camera_centers, _ = intersect_skew_lines_high_dim(origins, directions)

    # Retrieve target rays
    I_camera = PerspectiveCameras(focal_length=[1] * rays.shape[0], device=device)
    I_patch_rays = cameras_to_rays(
        cameras=I_camera,
        num_patches_x=num_patches_x,
        num_patches_y=num_patches_y,
        use_half_pix=use_half_pix,
        crop_parameters=crop_parameters,
    ).get_directions()

    if sampled_ray_idx is not None:
        I_patch_rays = I_patch_rays[:, sampled_ray_idx]

    # Compute optimal rotation to align rays
    Rs = []
    focal_lengths = []
    principal_points = []
    for i in range(rays.shape[-3]):
        R, f, pp = compute_optimal_rotation_intrinsics(
            I_patch_rays[i],
            directions[i],
            reproj_threshold=reproj_threshold,
        )
        Rs.append(R)
        focal_lengths.append(f)
        principal_points.append(pp)

    R = torch.stack(Rs)
    focal_lengths = torch.stack(focal_lengths)
    principal_points = torch.stack(principal_points)
    T = -torch.matmul(R.transpose(1, 2), camera_centers.unsqueeze(2)).squeeze(2)
    if ret_center:
        return PerspectiveCameras(
            R=R,
            T=T,
            focal_length=focal_lengths,
            principal_point=principal_points,
            device=device,
        ), camera_centers
    return PerspectiveCameras(
        R=R,
        T=T,
        focal_length=focal_lengths,
        principal_point=principal_points,
        device=device,
    )


def compute_optimal_rotation_alignment(A, B):
    """
    Compute optimal R that minimizes: || A - B @ R ||_F

    Args:
        A (torch.Tensor): (N, 3)
        B (torch.Tensor): (N, 3)

    Returns:
        R (torch.tensor): (3, 3)
    """
    # normally with R @ B, this would be A @ B.T
    H = B.T @ A
    U, _, Vh = torch.linalg.svd(H, full_matrices=True)
    s = torch.linalg.det(U @ Vh)
    S_prime = torch.diag(torch.tensor([1, 1, torch.sign(s)], device=A.device))
    return U @ S_prime @ Vh


def compute_optimal_rotation_intrinsics(
    rays_origin, rays_target, z_threshold=1e-4, reproj_threshold=0.2
):
    """
    Note: for some reason, f seems to be 1/f.

    Args:
        rays_origin (torch.Tensor): (N, 3)
        rays_target (torch.Tensor): (N, 3)
        z_threshold (float): Threshold for z value to be considered valid.

    Returns:
        R (torch.tensor): (3, 3)
        focal_length (torch.tensor): (2,)
        principal_point (torch.tensor): (2,)
    """
    device = rays_origin.device
    z_mask = torch.logical_and(
        torch.abs(rays_target) > z_threshold, torch.abs(rays_origin) > z_threshold
    )[:, 2]
    rays_target = rays_target[z_mask]
    rays_origin = rays_origin[z_mask]
    rays_origin = rays_origin[:, :2] / rays_origin[:, -1:]
    rays_target = rays_target[:, :2] / rays_target[:, -1:]

    A, _ = cv2.findHomography(
        rays_origin.cpu().numpy(),
        rays_target.cpu().numpy(),
        cv2.RANSAC,
        reproj_threshold,
    )
    A = torch.from_numpy(A).float().to(device)

    if torch.linalg.det(A) < 0:
        A = -A

    R, L = ql_decomposition(A)
    L = L / L[2][2]

    f = torch.stack((L[0][0], L[1][1]))
    pp = torch.stack((L[2][0], L[2][1]))
    return R, f, pp


def compute_ndc_coordinates(
    crop_parameters=None,
    use_half_pix=True,
    num_patches_x=16,
    num_patches_y=16,
    device=None,
):
    """
    Computes NDC Grid using crop_parameters. If crop_parameters is not provided,
    then it assumes that the crop is the entire image (corresponding to an NDC grid
    where top left corner is (1, 1) and bottom right corner is (-1, -1)).
    """
    if crop_parameters is None:
        cc_x, cc_y, width = 0, 0, 2
    else:
        if len(crop_parameters.shape) > 1:
            return torch.stack(
                [
                    compute_ndc_coordinates(
                        crop_parameters=crop_param,
                        use_half_pix=use_half_pix,
                        num_patches_x=num_patches_x,
                        num_patches_y=num_patches_y,
                    )
                    for crop_param in crop_parameters
                ],
                dim=0,
            )
        device = crop_parameters.device
        cc_x, cc_y, width, _ = crop_parameters

    dx = 1 / num_patches_x
    dy = 1 / num_patches_y
    if use_half_pix:
        min_y = 1 - dy
        max_y = -min_y
        min_x = 1 - dx
        max_x = -min_x
    else:
        min_y = min_x = 1
        max_y = -1 + 2 * dy
        max_x = -1 + 2 * dx

    y, x = torch.meshgrid(
        torch.linspace(min_y, max_y, num_patches_y, dtype=torch.float32, device=device),
        torch.linspace(min_x, max_x, num_patches_x, dtype=torch.float32, device=device),
        indexing="ij",
    )
    x_prime = x * width / 2 - cc_x
    y_prime = y * width / 2 - cc_y
    xyd_grid = torch.stack([x_prime, y_prime, torch.ones_like(x)], dim=-1)
    return xyd_grid