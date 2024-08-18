import sys
sys.path.append('/root/autodl-tmp/LLM-Pruner/lm-evaluation-harness')

import os, gc
import argparse
import glob
import pandas as pd
import shutil
from datetime import datetime
import torch
from post_training_optuna import main as train
import main_optune

import optuna
from optuna.distributions import IntDistribution


def reset_cuda():
    gc.collect()
    torch.cuda.empty_cache()
    # torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


DS = ['arc_challenge', 'arc_easy', 'boolq', 'piqa', 'winogrande', 'openbookqa', 'hellaswag']

parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

# Model Type&Path
parser.add_argument('--base_model', type=str, default="baffo32/decapoda-research-llama-7B-hf", help='base model name')
parser.add_argument('--prune_model', type=str, help='prune model name')  # 🚀🚀🚀🚀🚀
parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
parser.add_argument('--cache_dataset', action="store_true", default=False)
parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory') # 🚀🚀🚀🚀🚀

# Training Hyperparameters
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
parser.add_argument('--num_epochs', type=int, default=2, help='number of epochs')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
parser.add_argument('--cutoff_len', type=int, default=256, help='cutoff length')
parser.add_argument('--val_set_size', type=int, default=2000, help='validation set size')
parser.add_argument('--prompt_template_name', type=str, default="alpaca", help="The prompt template to use, will default to alpaca.")
parser.add_argument('--no_instruction', action='store_true', default=False, help="Whether to use the instruction template or not.")

# Lora Configuration
# parser.add_argument('--lora_r', type=int, default=8, help='lora r')
parser.add_argument('--lora_alpha', type=int, default=16, help='lora alpha')
parser.add_argument('--lora_dropout', type=float, default=0.05, help='lora dropout')
parser.add_argument('--lora_target_modules', type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj", help='lora target modules')

# llm hyperparameters
parser.add_argument('--train_on_inputs', default=False, action="store_true", help='Train on inputs. If False, masks out inputs in loss')
parser.add_argument('--add_eos_token', default=False, action="store_true")
parser.add_argument('--group_by_length', default=False, action="store_true", help="faster, but produces an odd training loss curve")

# wandb params
parser.add_argument('--wandb_project', type=str, default="")
parser.add_argument('--resume_from_checkpoint', type=str, help="either training checkpoint or final adapter")

#ddp
parser.add_argument('--local_rank', type=int, default=-1)

############## eval
parser.add_argument("--model", default="hf-causal-experimental")

parser.add_argument("--tasks", default="hellaswag")
parser.add_argument("--provide_description", action="store_true")
parser.add_argument("--num_fewshot", type=int, default=0)
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--output_path", default=None)
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--epoch", type=int, default=200)
parser.add_argument("--no_cache", type=bool, default=True)
parser.add_argument("--decontamination_ngrams_path", default=None)
parser.add_argument("--description_dict_path", default=None)
parser.add_argument("--check_integrity", action="store_true")
args = parser.parse_args()
torch_version = int(torch.__version__.split('.')[1])
args.torch_version = torch_version

output_dir = args.output_dir

def objective(trial):
    reset_cuda()
    
    args.output_dir = f"{output_dir}_{trial.number}"
    args.output_path = f"{args.output_dir}/results.json"
    max_memory_reserved = train(args, trial)
    

    shutil.copy(f"{args.output_dir}/adapter_config.json", f"{args.output_dir}/checkpoint-{args.epoch}/adapter_config.json")
    shutil.move(f"{args.output_dir}/checkpoint-{args.epoch}/pytorch_model.bin", f"{args.output_dir}/checkpoint-{args.epoch}/adapter_model.bin")
    args.model_args = f"checkpoint={args.prune_model},peft={args.output_dir}/checkpoint-{args.epoch},config_pretrained={args.base_model}"
    results = main_optune.main(args)

    dataset_scores = {k:max(score for mn, score in v.items() if 'stderr' not in mn) for k, v in results['results'].items()}

    hellaswag = dataset_scores['hellaswag']

    return hellaswag, max_memory_reserved


if __name__ == "__main__":
    csvs = [csvfile for csvfile in glob.glob('optuna_results/*.csv')]
    sorted(csvs)
    if len(csvs) > 0:
        df_results = pd.read_csv(csvs[-1])
    else:
        df_results = pd.DataFrame()
    
    # create new study
    date_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    study = optuna.create_study(directions=["maximize", 'minimize'])

    if len(df_results)>0:
        for _, row in df_results.iterrows():
            trial = optuna.trial.create_trial(
                params={c.replace('params_', ""):int(row[c]) for c in df_results.columns if 'params_layer'in c},
                distributions={c.replace('params_', ""):IntDistribution(4, 8, step=4) for c in df_results.columns if 'params_layer'in c},
                values=[row[f'values_{i}'] for i in range(2)]
            )
            study.add_trial(trial)
    print(f'Total number of trials have been done = {len(study.trials)}')
    
    study.optimize(objective, n_trials=1)
    study.trials_dataframe().to_csv(f'optuna_results/optuna_trials_{date_str}.csv', index=False)
        
    print("Number of finished trials: ", len(study.trials))

# export HF_ENDPOINT=https://hf-mirror.com

# CUDA_VISIBLE_DEVICES=0 nohup python run_optuna.py --prune_model prune_log/llama7b-0.5/pytorch_model.bin --output_dir tune_log/llama7b_50_optuna --epoch 200  > llama7b-0.5-optuna.log 2>&1 &
    
# CUDA_VISIBLE_DEVICES=0 python run_optuna.py --prune_model prune_log/vicuna-7b_prune_50/pytorch_model.bin --output_dir tune_log/vicuna7b_50_optuna --epoch 10 --limit 5
