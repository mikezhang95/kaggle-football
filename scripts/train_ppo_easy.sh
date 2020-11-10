#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=-1

python3 -u run_ppo2.py \
    --level 11_vs_11_easy_stochastic \
    --state extracted_stacked \
    --reward_experiment scoring,checkpoints \
    --policy ActorCriticPolicy \
    --lr 1e-5 \
    --ent_coef 0.003 \
    --vf_coef 0.5 \
    --gamma 0.993 \
    --clip_range 0.08 \
    --max_grad_norm 0.64 \
    --num_timesteps 5000000 \
    --num_envs 2 \
    --nsteps 512 \
    --n_epochs 10 \
    --save_interval 100 \
    --seed 0 \
    --save_path ./outputs/ppo \
