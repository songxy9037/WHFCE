#!/usr/bin/env bash

gpus=0

data_name=HCDv4
net_G=WHFCE
split=test
vis_root=vis
project_name=WaveHFD_GF

checkpoints_root=checkpoints
checkpoint_name=best_ckpt.pt
img_size=256


python eval_cd.py --split ${split} --net_G ${net_G} --img_size ${img_size} --vis_root ${vis_root} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}



