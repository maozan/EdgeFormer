#!/usr/bin/env bash

gpus=0

data_name=LEVIR
net_G=edm_cgm_fm
split=test1
vis_root=./vis
project_name=edm_cgm_fm
checkpoints_root=./checkpoints
checkpoint_name=best_ckpt.pt
img_size=256
dataset=CDDataset_edge
position_length=2
blocks=3


python eval_cd.py --dataset ${dataset} --blocks ${blocks} --position_length ${position_length} --split ${split} --net_G ${net_G} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


