#!/usr/bin/env bash

#GPUs
gpus=3

#Set paths
checkpoint_root=./checkpoints
vis_root=./vis
data_name=LEVIR
dataset=CDDataset_edge  # CDDataset_edge


img_size=256    
batch_size=8   
lr=0.0001         
max_epochs=200
embed_dim=256
position_length=2
blocks=3

net_G=edm_cgm_fm        # (HMNet_pure, no edge)、(HMNet_xx_lv, edge)

lr_policy=linear
optimizer=adamw                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=Flase
multi_scale_infer=False
shuffle_AB=False

#Initializing from pretrained weights
# pretrain=None

#Train and Validation splits
split=train         #trainval
split_val=test      #test
project_name=${net_G}_${blocks}_${position_length}_${dataset}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}

python main_cd.py --img_size ${img_size} --blocks ${blocks} --position_length ${position_length} --dataset ${dataset} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}
