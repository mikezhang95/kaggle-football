#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

#LEVEL=academy_empty_goal_close
#LEVEL=academy_pass_and_shoot_with_keeper
#LEVEL=academy_counterattack_easy
LEVEL=academy_shoot_with_keeper


python3 -u test_ppo2.py \
    --level ${LEVEL} \
    --state extracted_stacked \
    --reward_experiment scoring \
    --policy MlpPolicy \
    --n_epochs 100 \
    --load_path ./outputs/${LEVEL} \
#     --render  
