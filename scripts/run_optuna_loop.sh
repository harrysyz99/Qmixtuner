#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

for i in $(seq 1 50); do
  echo "Executing command $i:"
  echo "CUDA_VISIBLE_DEVICES=0 python run_optuna.py --prune_model prune_log/llama7b-50/pytorch_model.bin --output_dir tune_log/llama7b-50_optuna --epoch 200"
  CUDA_VISIBLE_DEVICES=0 python run_optuna.py \
    --prune_model prune_log/llama7b-50/pytorch_model.bin \
    --output_dir "tune_log/llama7b-50_optuna" \
    --epoch 200
done
