prune_ckpt_path='llama7b-50'
export HF_ENDPOINT=https://hf-mirror.com
device=0

echo "[START] - Start Pruning Model"
CUDA_VISIBLE_DEVICES=$device python hf_prune.py --pruning_ratio 0.625 \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
      --block_attention_layer_start 4 --block_attention_layer_end 30 \
      --pruner_type taylor \
      --test_after_train \
      --device cpu  --eval_device cuda \
      --save_ckpt_log_name $prune_ckpt_path \
      --base_model 'baffo32/decapoda-research-llama-7B-hf' \
      --save_model
echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Tuning"
# CUDA_VISIBLE_DEVICES=$device python post_training_no_bits.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --output_dir tune_log/$tune_ckpt_path
# echo "[FINISH] - Finish Prune and Post-Training."      
# echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

# echo "[START] - Start Evaluating"
# CUDA_VISIBLE_DEVICES=$device bash scripts/evaluate.sh  baffo32/decapoda-research-llama-7B-hf tune_log/$tune_ckpt_path  prune_log/$prune_ckpt_path 200
# echo "[FINISH] - Finish Evaluating"




















# echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=0 python hf_prune.py --pruning_ratio 0.635 \
#       --block_wise \
#       --block_mlp_layer_start 4 --block_mlp_layer_end 30 \
#       --block_attention_layer_start 4 --block_attention_layer_end 30 \
#       --pruner_type taylor \
#       --test_after_train \
#       --device cpu  --eval_device cuda \
#       --save_ckpt_log_name $prune_ckpt_path \
#       --base_model 'baffo32/decapoda-research-llama-7B-hf' \
#       --save_model
# echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Pruning Model"
# CUDA_VISIBLE_DEVICES=0 python hf_prune.py --pruning_ratio 0.25 --device cpu  --eval_device cuda --block_wise --block_mlp_layer_start 4 --block_mlp_layer_end 30 --block_attention_layer_start 4 --block_attention_layer_end 30 --save_ckpt_log_name $prune_ckpt_path --pruner_type taylor --test_after_train --taylor param_first --save_model
# echo "[FINISH] - Finish Pruning Model"

# echo "[START] - Start Tuning"
# CUDA_VISIBLE_DEVICES=0 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --output_dir tune_log/$tune_ckpt_path --wandb_project llama_tune --num_epochs 2 --learning_rate 1e-4 --batch_size 64
# echo "[FINISH] - Finish Prune and Post-Training."
# echo "[INFO] - The pruned model is at {prune_log/$prune_ckpt_path/pytorch_model.bin}, and the recovery weight is at {tune_log/$tune_ckpt_path}/"

# CUDA_VISIBLE_DEVICES=1 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin \
#       --data_path yahma/alpaca-cleaned \
#       # --lora_r 8 \
#       --num_epochs 2 \ 
#       --learning_rate 1e-4 \ 
#       --batch_size 64 \
#       --output_dir tune_log/$tune_ckpt_path \ 
#       --wandb_project llama_tune \
#       --learning_rate 1e-4 \
#       --batch_size 512

# CUDA_VISIBLE_DEVICES=1 python post_training.py --prune_model prune_log/$prune_ckpt_path/pytorch_model.bin --data_path yahma/alpaca-cleaned --num_epochs 2 --learning_rate 1e-4 --batch_size 64 --output_dir tune_log/$tune_ckpt_path --batch_size 512

# echo "You can use the command:"
# echo "       python generate.py --model_type tune_prune_LLM --ckpt prune_log/$prune_ckpt_path/pytorch_model.bin --lora_ckpt tune_log/$tune_ckpt_path"
# echo "to use the pruned model"

# echo "[START] - Start Evaluating"
# CUDA_VISIBLE_DEVICES=0 bash scripts/evaluate.sh  lmsys/vicuna-7b-v1.5  prune_log/$prune_ckpt_path 
# echo "[FINISH] - Finish Evaluating"
