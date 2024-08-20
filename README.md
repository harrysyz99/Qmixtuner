# Qmixtuner


# README for Fine-Tuning Pruned Language Models

## Overview
This script is designed to fine-tune pruned large language models (LLMs) with additional features such as quantization and memory tracking. It integrates several libraries, including `transformers`, `torch`, `optuna`, and `bitsandbytes`, to handle various aspects like model optimization, dataset handling, and experimental trials.

The script supports:
- Loading and fine-tuning pruned models.
- Dynamic quantization with configurable bit precision.
- Memory usage tracking during training.
- Evaluation on multiple datasets using custom prompts.
- Integration with `optuna` for hyperparameter tuning.

## Installation

### Prerequisites
Ensure you have Python 3.8+ installed on your system. You will also need PyTorch with CUDA support if training on GPU.

### Dependencies
Install all required libraries using pip. You may want to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

## Advanced Usage

### Model Pruning
This section details how to prune a model using the provided bash script. Pruning is a process of reducing the model size by removing less important parameters, which can help decrease memory usage and improve inference speed without a significant loss in performance.

#### Pruning Script
The script `prune.sh` (assuming the filename) sets up and runs the model pruning process. It uses the `hf_prune.py` script which should be implemented to handle the pruning logic based on the provided arguments.

#### Pruning Command Explanation
- `prune_ckpt_path`: A variable to specify the checkpoint path for saving the pruned model.
- `HF_ENDPOINT`: Specifies the endpoint for accessing Hugging Face models, which can be useful if the default endpoint is slow or inaccessible.
- `device`: Specifies the GPU device ID to use for pruning. Set to `0` in this script.

```bash
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
Hyperparameter Tuning
After pruning, it is often beneficial to fine-tune the pruned model to regain or improve the performance. The following script automates the hyperparameter tuning using Optuna, a hyperparameter optimization framework.

Tuning Script
The script tune.sh (assuming the filename) is used to perform hyperparameter tuning on the pruned model. The script iteratively runs the run_optuna.py script with different settings.

Tuning Command Explanation
prune_ckpt_path: Path to the pruned model.
output_dir: Directory to save tuning logs and output models.
bash
Copy code
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
```

## Acknowledgements

We gratefully acknowledge the use of the LLM-Pruner framework in the development of our project. This tool has significantly contributed to the efficiency and effectiveness of our model pruning processes. We also extend our thanks to Optuna for providing a robust and versatile platform for hyperparameter optimization, enabling us to achieve optimal model performance through systematic and automated tuning. The support and resources from both these projects have been invaluable to our research and development efforts.

