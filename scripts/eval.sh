#!/usr/bin/env bash

gpus=0

data_name=LEVIR
net_G=HMNet_feature_lv_ca_fusion
split=test
vis_root=./vis
project_name=HMNet_feature_lv_ca_fusion_e4_CDDataset_edge_LEVIR_b16_lr0.0001_adamw_train_test_100_linear_ce
checkpoints_root=./checkpoints
checkpoint_name=best_ckpt.pt
img_size=256
dataset=CDDataset_edge


python eval_cd.py --dataset ${dataset} --split ${split} --net_G ${net_G} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


