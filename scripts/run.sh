#!/usr/bin/env bash

gpus=2
checkpoint_root=checkpoints
vis_root=vis
data_name=img6

img_size=256
batch_size=16
lr=0.0001
max_epochs=150
net_G=WaveHFD

lr_policy=linear
optimizer=adamw
split=train
split_val=val
project_name=15${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
