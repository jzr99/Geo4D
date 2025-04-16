import argparse, os, sys, glob
import datetime, time
from omegaconf import OmegaConf
from tqdm import tqdm
from einops import rearrange, repeat
from collections import OrderedDict
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from utils.funcs import load_video_batch

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config
import random
import numpy as np
import cv2
from dust3r.cloud_opt.optimizer_group import LightPointCloudGroupOptimizer


from dust3r.utils.vo_eval import eval_metrics, plot_trajectory
from dust3r.demo import get_3D_model_from_scene
from dust3r.depth_eval import depth_evaluation

import time

def post_optimization(view1, pred1, args, conf_optimize=False, init_method='group', lr=0.03, align=True, intrinsics=None, **kwargs):
    if init_method=='group':
        device = pred1[0]['pts3d'].device
        scene = LightPointCloudGroupOptimizer(view1, pred1, conf='id', conf_optimize=conf_optimize, verbose=True,
            shared_focal=not args.not_shared_focal and not args.use_gt_focal,
            flow_loss_weight=args.flow_loss_weight, flow_loss_fn=args.flow_loss_fn,
            depth_regularize_weight=args.depth_regularize_weight,
            num_total_iter=args.n_iter, temporal_smoothing_weight=args.temporal_smoothing_weight, motion_mask_thre=args.motion_mask_thre,
            flow_loss_start_epoch=args.flow_loss_start_epoch, flow_loss_thre=args.flow_loss_thre, translation_weight=args.translation_weight,
            sintel_ckpt=args.eval_dataset == 'sintel', use_self_mask=not args.use_gt_mask, sam2_mask_refine=args.sam2_mask_refine,
            empty_cache=len(view1) >= 80 and len(pred1['pts3d']) > 600, pxl_thre=args.pxl_thresh, **kwargs # empty cache to make it run on 48GB GPU
        ).to(device)
        lr = lr # 0.01
        if intrinsics is not None:
            scene.preset_focal([(intrinsics[i,0,0] + intrinsics[i, 1, 1]) / 2.0 for i in range(intrinsics.shape[0])], requires_grad=False)
        if align:
            loss = scene.compute_global_alignment(
                init=init_method, niter=args.n_iter, schedule=args.pose_schedule, lr=lr,
            )
    else:
        raise NotImplementedError
    return scene


def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        try:
            model.load_state_dict(state_dict, strict=True)
        except:
            ## rename the keys for 256x256 model
            new_pl_sd = OrderedDict()
            for k,v in state_dict.items():
                new_pl_sd[k] = v

            for k in list(new_pl_sd.keys()):
                if "framestride_embed" in k:
                    new_key = k.replace("framestride_embed", "fps_embedding")
                    new_pl_sd[new_key] = new_pl_sd[k]
                    del new_pl_sd[k]
            import pdb;pdb.set_trace()
            model.load_state_dict(new_pl_sd, strict=False)
            # model.load_state_dict(new_pl_sd, strict=True)
    else:
        # deepspeed
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]]=state_dict['module'][key]
        model.load_state_dict(new_pl_sd)
    print('>>> model checkpoint loaded.')
    return model


def denormalize_pc_bbox2(pc, alpha=1.0, beta=1.0):
    denormalized_pts = pc.clone()
    denormalized_pts[..., 0] = denormalized_pts[..., 0] / alpha
    denormalized_pts[..., 1] = denormalized_pts[..., 1] / beta
    denormalized_pts[..., 2] = (denormalized_pts[..., 2] + 1) / 2
    return denormalized_pts

def normalize_depth(depth):
    T, H, W = depth.shape
    depth = depth.reshape(-1)

    sorted, indices = torch.sort(depth, dim=0)

    total_pts = sorted.shape[0]
    lower = int(total_pts * 0.00)
    upper = int(total_pts * 1.00) - 1
    lower_bound = sorted[lower] - 0.01
    upper_bound = sorted[upper] + 0.01
    s = (upper_bound - lower_bound)
    depth = (depth - lower_bound) / s
    depth = depth * 2.0 - 1.0
    depth = depth.reshape(T, H, W, 1)
    depth = depth.repeat(1, 1, 1, 3)
    return depth, s, lower_bound


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def image_guided_synthesis(model, prompts, videos, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., \
                        unconditional_guidance_scale=1.0, cfg_img=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, pointmap_vae=None, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size

    if model.modality == 'img_vidpc':
        if model.cross_attention:
            img = videos[:,:,0,...]
            img = img
            ## img: b c h w
            img_emb = model.embedder(img) ## b l c
            img_emb = model.image_proj_model(img_emb)
        else:
            img = videos[:,:,0,...]
            img = img * 0
            ## img: b c h w
            img_emb = model.embedder(img) ## b l c
            img_emb = model.image_proj_model(img_emb)
    else:
        if model.cross_attention:
            bs = videos.shape[0]
            num_frames = videos.shape[2]
            img = rearrange(videos, 'b c t h w -> b t c h w')
            img = rearrange(img, 'b t c h w -> (b t) c h w')
            ## img: b c h w
            img_emb = model.embedder(img) ## b l c
            img_emb = rearrange(img_emb, '(b t) l c -> b t l c', b=bs, t=num_frames)
            img_emb = model.image_proj_model(img_emb)
            print("use cross attention")
        else:
            img = videos[:,:,0,...]
            img = img * 0
            ## img: b c h w
            img_emb = model.embedder(img) ## b l c
            img_emb = model.image_proj_model(img_emb)
    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        if loop or interp:
            raise NotImplementedError
        else:
            if model.modality == 'img_vidpc':
                img_cat_cond = z[:,:,:1,:,:]
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
            else:
                img_cat_cond = z[:,:,:,:,:]
        cond["c_concat"] = [img_cat_cond] # b c 1 h w
    
    if unconditional_guidance_scale != 1.0:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        if model.modality == 'img_vidpc':
            pass
        else:
            if model.cross_attention:
                uc_img_emb = rearrange(uc_img_emb, '(b t) l c -> b t l c', b=bs, t=num_frames)
        uc_img_emb = model.image_proj_model(uc_img_emb)
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    z0 = None
    cond_mask = None

    batch_variants = []
    for _ in range(n_samples):

        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            fs=fs,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            **kwargs
                                            )

        ## reconstruct from latent to pixel space
        # batch_images = model.decode_first_stage(samples)
        if model.modality == 'img_vidpc':
            # z_video, z_pc
            batch_images = model.decode_first_stage(samples[:,0:4])
            batch_pc = model.decode_first_stage_confhead(samples[:,4:])
            batch_images = torch.cat([batch_images, batch_pc], dim=1) # c = 3+4
            print('infer with vidpc vae')
        elif model.modality == 'multipc':
            batch_pc_0 = model.decode_first_stage_confhead(samples[:,0:4])
            batch_pc_1 = model.decode_first_stage_confhead(samples[:,4:8])
            batch_images = model.decode_first_stage(samples[:,8:])
            batch_images = torch.cat([batch_images, batch_pc_0, batch_pc_1], dim=1) # c = 3+4+4
            print('infer with multipc vae')
        elif model.modality == 'pc_ray':
            batch_pc_0 = model.decode_first_stage_confhead(samples[:,0:4])
            batch_raymap = model.decode_first_stage(samples[:,4:])
            batch_images = torch.cat([batch_pc_0, batch_raymap], dim=1) # c = 4 + 3
            print('infer with pc raymap')
        elif model.modality == 'pc_ray_cross_depth':
            if pointmap_vae is not None:
                # print('infer with pointmap vae')
                batch_pc_0 = decode_pm_confhead(samples[:, 0:4], model, pointmap_vae)
            else:
                batch_pc_0 = model.decode_first_stage_confhead(samples[:,0:4])
            batch_raymap = model.decode_first_stage(samples[:,4:8])
            batch_cross = model.decode_first_stage(samples[:,8:12])
            batch_depth = model.decode_first_stage(samples[:,12:])
            batch_depth = batch_depth.mean(dim=1, keepdim=True)
            batch_images = torch.cat([batch_pc_0, batch_raymap, batch_cross, batch_depth], dim=1)
        else:
            if model.perchannel_vae:
                batch_images = model.decode_core_confhead_perchannel(samples)
                print('infer with perchannel vae')
            else:
                if pointmap_vae is not None:
                    # print('infer with pointmap vae')
                    batch_images = decode_pm_confhead(samples[:, 0:4], model, pointmap_vae)
                else:
                    batch_images = model.decode_first_stage_confhead(samples[:,0:4])
                # batch_images = model.decode_first_stage_confhead(samples)
                print('infer with standard vae')
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)

def get_sky_mask(x_recon_reshape, sky_value=1.05, eps=0.05):
    # input [t h w c]
    min_val = sky_value-eps
    max_val = sky_value+eps
    sky_mask = (x_recon_reshape[..., 0] > min_val) & (x_recon_reshape[..., 0] < max_val) & (x_recon_reshape[..., 1] > min_val) & (x_recon_reshape[..., 1] < max_val) & (x_recon_reshape[..., 2] > min_val) & (x_recon_reshape[..., 2] < max_val)
    return sky_mask.unsqueeze(-1)


def get_far_away_mask(x_recon_reshape, far_away_value=1.5):
    # input [t h w c]
    far_away_mask = (abs(x_recon_reshape) > far_away_value).any(dim=-1)
    return far_away_mask.unsqueeze(-1)



def decode_pm_confhead(z, model, pointmap_vae):
    if model.encoder_type == "2d" and z.dim() == 5:
        b, _, t, _, _ = z.shape
        z = rearrange(z, 'b c t h w -> (b t) c h w')
        reshape_back = True
    else:
        reshape_back = False
        
    if not model.perframe_ae:    
        z = 1. / model.scale_factor * z
        results = pointmap_vae.decode_with_conf_adaptor(z)
    else:
        results = []
        for index in range(z.shape[0]):
            frame_z = 1. / model.scale_factor * z[index:index+1,:,:,:]
            frame_result = pointmap_vae.decode_with_conf_adaptor(frame_z)
            results.append(frame_result)
        results = torch.cat(results, dim=0)

    if reshape_back:
        results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
    return results

def run_inference(args, gpu_num, gpu_no):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())

    silent = config.postprocess.silent
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    model.perframe_ae = args.perframe_ae
    assert os.path.exists(args.ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)
    pointmap_vae = None
    # load fine-tuned vae decoder
    if 'vae_path' in config:
        pointmap_vae_config = config.pop("pointmap_vae_config", OmegaConf.create())
        pointmap_vae = instantiate_from_config(pointmap_vae_config)
        pointmap_vae = pointmap_vae.eval()
        pointmap_vae = pointmap_vae.cuda(gpu_no)
        from lvdm.basics import disabled_train
        pointmap_vae.train = disabled_train
        for param in pointmap_vae.parameters():
            param.requires_grad = False
        print(f'load vae path:', config['vae_path'])
        vae_weights = torch.load(config['vae_path'])
        vae_weights = vae_weights['state_dict']
        new_pl_sd = OrderedDict()
        for k,v in vae_weights.items():
            if k.startswith('model.'):
                k = k[6:]
                new_pl_sd[k] = v
        pointmap_vae.load_state_dict(new_pl_sd, strict=True)
    if pointmap_vae is None:
        orivae = True
    else:
        orivae = False

    model.eval()
    assert (args.width, args.height) in [(512, 384), (512, 320), (576, 256), (640, 192)], "Current implementation only support [input size = [(512, 384), (512, 320), (576, 256), (640, 192)]]"
    assert (args.height % 64 == 0) and (args.width % 64 == 0), "Error: image size [h,w] should be multiples of 64!"
    assert args.bs == 1, "Current implementation only support [batch size = 1]!"


    # load dataset from config
    # config.data.params.test.params.dataset = args.dataset
    # config.data.params.test.params.full_seq = args.full_seq
    # dataset = instantiate_from_config(config.data)
    # dataset.setup()
    # dataloader = dataset._test_dataloader()

    # dataset_name = args.dataset

    use_raymap = False
    use_crossmap = False
    use_inverse_depthmap = True
    use_traj = True
    config.postprocess.use_gt_focal=False

    seq_name = args.video_path.split('/')[-1].split('.')[0]

    vaedir = os.path.join(args.savedir, f'{seq_name}')

    os.makedirs(vaedir, exist_ok=True)
    save_dir = vaedir

    video_frames, fps_list = load_video_batch([args.video_path], frame_stride=args.frame_sampling_stride, video_size=(args.height, args.width), video_frames=args.max_video_frames)
    # import pdb;pdb.set_trace()
    B,C,T,H,W = video_frames.shape

    views = []
    for i in range(T):
        view = {
            'img': video_frames[0,:,i,:,:],
            'idx': (i,),
        }
        views.append(view)

    time_list = []
    time_for_each = 0
    total_frames = 0
    for idx, batch in enumerate(video_frames):
        fps = fps_list[idx]
        time_for_each = 0
        video = batch
        video = video.unsqueeze(0) # b c t h w
        B,C,T,H,W = video.shape
        channels = model.model.diffusion_model.out_channels
        n_frames = args.video_length
        # print(f'Inference with {n_frames} frames')
        noise_shape = [args.bs, channels, n_frames, H // 8, W // 8]
        seq = seq_name
        filenames = seq_name
        

        prompts = ['Output a video that assigns each 3D location in the world a consistent color.']
        videos_all = video.to("cuda") # b c t h w

        B,C,T,H,W = videos_all.shape
        intrinsics = None
        total_frames = total_frames + T

        slice_list = []
        stride = args.stride 
        for start in list(range(0, T-16 + 1, stride)):
            slice_list.append(slice(start, start+16, 1))
        if slice(T-16, T) not in slice_list:
            slice_list.append(slice(T-16, T, 1))
        print('slice_list:', slice_list)

        pred_list = []
        view_list = []
        npz_results_list = []
        pnt_valid_mask = torch.ones((T,H,W,1), device='cuda') > 0
        ref_raymap = None
        print("Diffusion Inference Start")
        for sl in tqdm(slice_list):
            videos = videos_all[:,:,sl,:,:].clone()
            # print(f'Inference with {sl.start} to {sl.stop} frames with step {sl.step}, with total {T} frames')
            
            view_list.append(views[sl])
            
            raymap = None
            crossmap = None
            inverse_depthmap = None
            traj = None
            # batch, variants, c, t, h, w
            batch_samples = image_guided_synthesis(model, prompts, videos, noise_shape, args.n_samples, args.ddim_steps, args.ddim_eta, \
                                args.unconditional_guidance_scale, args.cfg_img, fps, True, args.multiple_cond_cfg, args.loop, args.interp, args.timestep_spacing, args.guidance_rescale, pointmap_vae=pointmap_vae)

            assert batch_samples.shape[1] == 1, "only support variants size = 1"
            batch_samples = batch_samples[:,0]
            if batch_samples.shape[1] == 7 and model.modality == 'pc_ray' and use_raymap:
                raymap = batch_samples[:, 4:] 
                raymap = rearrange(raymap, 'b c t h w -> (b t) c h w')
                raymap = rearrange(raymap, 't c h w -> t h w c')
            elif model.modality == 'pc_ray_cross_depth':
                raymap = batch_samples[:, 4:7] 
                crossmap = batch_samples[:, 7:10]
                traj = raymap_to_camera_matrix(raymap, crossmap)
                raymap = rearrange(raymap, 'b c t h w -> (b t) c h w')
                raymap = rearrange(raymap, 't c h w -> t h w c')
                crossmap = rearrange(crossmap, 'b c t h w -> (b t) c h w')
                crossmap = rearrange(crossmap, 't c h w -> t h w c')
                inverse_depthmap = batch_samples[:, 10:11]
                inverse_depthmap = rearrange(inverse_depthmap, 'b c t h w -> (b t) c h w')
                inverse_depthmap = rearrange(inverse_depthmap, 't c h w -> t h w c')
                inverse_depthmap = (inverse_depthmap + 1.0) / 2.0
            else:
                raymap = None
            batch_samples = batch_samples[:, :4] # only keep point map

            
            x_recon = rearrange(batch_samples, 'b c t h w -> (b t) c h w')
            confidence = x_recon[:,[-1],:,:]
            softplus = torch.nn.Softplus()
            confidence = softplus(confidence)
            confidence = rearrange(confidence, 't c h w -> t h w c')
            if pointmap_vae is None:
                confidence = torch.ones_like(confidence)

            x_recon = x_recon[:,:-1,:,:] # 
            # x_reshape = rearrange(x, 't c h w -> t h w c')
            x_recon_reshape = rearrange(x_recon, 't c h w -> t h w c')
            
            
            invalid_pts = get_sky_mask(x_recon_reshape, sky_value=1.05, eps=0.35)
            far_away_mask = get_far_away_mask(x_recon_reshape, far_away_value=1.99)
            invalid_pts = invalid_pts | far_away_mask
            confidence[invalid_pts] = 999.0
            pnt_valid_mask[sl] =  pnt_valid_mask[sl] * (~invalid_pts)

            inverse_confidence = 1 / confidence
            inverse_confidence[invalid_pts] = 0.0
            x_recon = rearrange(x_recon, 't c h w -> t h w c')
            x_recon = denormalize_pc_bbox2(x_recon, alpha=2.0, beta=2.0)


            pred_pts = {'pts3d': x_recon, 'conf': inverse_confidence}
            if raymap is not None and use_raymap:
                pred_pts['raydir'] = raymap
            if crossmap is not None and use_crossmap:
                pred_pts['crossmap'] = crossmap 
            if inverse_depthmap is not None and use_inverse_depthmap:
                pred_pts['inverse_depthmap'] = inverse_depthmap
            if traj is not None and use_traj:
                pred_pts['traj'] = traj

            pred_list.append(pred_pts)
            # torch.cuda.empty_cache()
        
        


        scene = post_optimization(view_list, pred_list, config.postprocess, conf_optimize=True, init_method='group', lr=0.03, opt_raydir=True if use_raymap else False, intrinsics=intrinsics)



        depthmap = scene.get_depthmaps()
        depthmap = torch.stack(depthmap, dim=0)
        T, H, W = depthmap.shape


        os.makedirs(f'{save_dir}/{seq}', exist_ok=True)


        pred_traj = scene.get_tum_poses()

        outfile = get_3D_model_from_scene(
            outdir=f'{save_dir}/{seq}', silent=silent, scene=scene, min_conf_thr=2, as_pointcloud=True, mask_sky=False,
            clean_depth=False, transparent_cams=False, cam_size=0.01, save_name=seq, is_msk=False,
        )

        scene.save_tum_poses(f'{save_dir}/{seq}/pred_traj.txt')
        scene.save_focals(f'{save_dir}/{seq}/pred_focal.txt')
        scene.save_intrinsics(f'{save_dir}/{seq}/pred_intrinsics.txt')
        scene.save_depth_maps(f'{save_dir}/{seq}')
        scene.save_conf_maps(f'{save_dir}/{seq}')
        scene.save_init_conf_maps(f'{save_dir}/{seq}')
        scene.save_rgb_imgs(f'{save_dir}/{seq}')


from utils.rays import cameras_from_plucker


def raymap_to_camera_matrix(raymap, crossmap, ref_raymap=None):

    if ref_raymap is not None:
        ref_raymap = ref_raymap.cpu()
    pytorch_camera,center, rays = cameras_from_plucker(raymap.cpu(), crossmap.cpu(), ref_raymap)
    R = pytorch_camera.R # here R is already R_c2w
    T = pytorch_camera.T # here t is T_w2c
    num_frame = R.shape[0]
    P = torch.eye(4, device='cpu').repeat(num_frame, 1, 1)
    P[:,:3,:3] = R
    P[:,:3,3] = T
    R_c2w = P[:,:3,:3]
    T_c2w = -torch.bmm(R_c2w, P[:,:3,3].unsqueeze(-1)).squeeze(-1)
    P_c2w = torch.eye(4, device='cpu').repeat(num_frame, 1, 1)
    P_c2w[:,:3,:3] = R_c2w
    P_c2w[:,:3,:3] = R
    P_c2w[:,:3,3] = T_c2w
    return P_c2w.to(raymap.device)



def get_input(batch, k):
        x = batch[k]
        '''
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        '''
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_dir", type=str, default=None, help="a data dir containing videos and prompts")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt",)
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--dataset", type=str, default=None, help="Evaluation Dataset")
    parser.add_argument("--full_seq", action='store_true', default=False, help="Evaluation Dataset")
    parser.add_argument("--stride", type=int, default=4, help="Sliding window stride for video")
    parser.add_argument("--video_path", type=str, default="./data/demo/drift-turn.mp4", help="Input video path")
    parser.add_argument("--max_video_frames", type=int, default=-1, help="Input video max length, -1 means use all frames")
    parser.add_argument("--frame_sampling_stride", type=int, default=1, help="Input video sampling stride")
    
    ## currently not support looping video and generative frame interpolation
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--interp", action='store_true', default=False, help="generate generative frame interpolation or not")
    return parser


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@Geo4D cond-Inference: %s"%now)
    parser = get_parser()
    args = parser.parse_args()

    seed = args.seed
    if seed < 0:
        seed = random.randint(0, 2 ** 31)
    seed_everything(seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)
