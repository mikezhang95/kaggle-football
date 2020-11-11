#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3

LEVEL=academy_pass_and_shoot_with_keeper

python3 -u run_ppo2.py \
    --level ${LEVEL} \
    --state extracted_stacked \
    --reward_experiment scoring,checkpoints \
    --policy ActorCriticPolicy \
    --lr 1e-3 \
    --ent_coef 0.0 \
    --vf_coef 0.5 \
    --gamma 0.99 \
    --clip_range 0.2 \
    --max_grad_norm 0.5 \
    --num_timesteps 500000 \
    --num_envs 2 \
    --nsteps 512 \
    --n_epochs 10 \
    --save_interval 1000 \
    --seed 0 \
    --save_path ./outputs/${LEVEL} > ./outputs/${LEVEL}/train.log
