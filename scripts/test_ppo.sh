#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

LEVEL=academy_empty_goal_close
#LEVEL=academy_pass_and_shoot_with_keeper
#LEVEL=academy_counterattack_easy

python3 -u test_ppo2.py \
    --level ${LEVEL} \
    --state extracted_stacked \
    --reward_experiment scoring,checkpoints \
    --policy ActorCriticPolicy \
    --n_epochs 10 \
    --load_path ./outputs/${LEVEL} \
    --render false 
