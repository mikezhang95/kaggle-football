#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

#LEVEL=academy_empty_goal_close
#LEVEL=academy_pass_and_shoot_with_keeper
#LEVEL=academy_counterattack_easy
LEVEL=academy_shoot_with_keeper

python3 -u run_ppo2.py \
    --level ${LEVEL} \
    --state extracted_stacked \
    --reward_experiment scoring \
    --policy MlpPolicy \
    --lr 5e-4 \
    --ent_coef 0.0 \
    --vf_coef 0.5 \
    --gamma 0.995 \
    --clip_range 0.1 \
    --max_grad_norm 0.5 \
    --num_timesteps 500000 \
    --num_envs 4 \
    --nsteps 500 \
    --n_epochs 10 \
    --save_interval 1000 \
    --seed 0 \
    --save_path ./outputs/${LEVEL} \
    > ./outputs/${LEVEL}/train.log
