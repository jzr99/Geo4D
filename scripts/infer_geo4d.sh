seed=123
name=geo4d_custom_seed${seed}_pc
epoch=model   
dir=geo4d

ckpt=checkpoints/$dir/$epoch.ckpt

config=configs/inference_geo4d.yaml

res_dir="results_demo"

CUDA_VISIBLE_DEVICES=$2 python3 scripts/evaluation/test_geo4d.py \
--video_path $1 \
--frame_sampling_stride 1 \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/${name}_${dir}_${epoch} \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 1.0 \
--ddim_steps 5 \
--ddim_eta 0.0 \
--text_input \
--video_length 16 \
--frame_stride 24 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae

# for better visual results of video depth, you can change the following parameters
# --unconditional_guidance_scale 7.5 
# --ddim_steps 50
# --ddim_eta 1.0