seed=123
name=geo4d_$1_seed${seed}_pc
epoch=model   
dir=geo4d

ckpt=checkpoints/$dir/$epoch.ckpt

config=configs/inference_geo4d.yaml

res_dir="results"

if [ "$1" == "sintel" ]; then
CUDA_VISIBLE_DEVICES=$2 python3 scripts/evaluation/infer_geo4d.py \
--dataset $1 \
--full_seq \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/${name}_${dir}_${epoch} \
--n_samples 1 \
--bs 1 --height 512 --width 256 \
--unconditional_guidance_scale 1.0 \
--ddim_steps 5 \
--ddim_eta 0.0 \
--text_input \
--video_length 16 \
--frame_stride 24 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae
else
CUDA_VISIBLE_DEVICES=$2 python3 scripts/evaluation/infer_geo4d.py \
--dataset $1 \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/${name}_${dir}_${epoch} \
--n_samples 1 \
--bs 1 --height 512 --width 256 \
--unconditional_guidance_scale 1.0 \
--ddim_steps 5 \
--ddim_eta 0.0 \
--text_input \
--video_length 16 \
--frame_stride 24 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae
fi


## multi-cond CFG: the <unconditional_guidance_scale> is s_txt, <cfg_img> is s_img
#--multiple_cond_cfg --cfg_img 7.5
#--loop