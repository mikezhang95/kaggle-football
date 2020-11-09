#!/bin/bash

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISILBE_DEVICES=2,3

python3 -u run_ppo2.py \
  --level 11_vs_11_easy_stochastic \
  --state extracted_stacked \
  --reward_experiment scoring,checkpoints \
  --policy impala_cnn \
  --lr 0.000343 \
  --ent_coef 0.003 \
  --gamma 0.993 \
  --cliprange 0.08 \
  --max_grad_norm 0.64 \
  --num_timesteps 50000000 \
  --num_envs 4 \
  --nsteps 512 \
  --noptepochs 4 \
  --nminibatches 4 \
  --save_interval 100 \
  --seed 0 \
  --save_path /kaggle-football/outputs/ppo/ \
  "$@"
