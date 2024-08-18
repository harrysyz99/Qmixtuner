'''
Refer to
https://github.com/tloen/alpaca-lora/blob/main/finetune.py
'''

import os
import sys
import argparse
from typing import List
from pathlib import Path
from inspect import signature
import numpy as np
from collections import Counter

import torch
from torch import nn
import transformers
from datasets import load_dataset

from accelerate import init_empty_weights
from peft import prepare_model_for_kbit_training
from LLMPruner.peft import (
    replace_lora_weights_loftq,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    LoftQConfig
    
)
from LLMPruner.utils.prompter import Prompter, ZeroPrompter
from LLMPruner.datasets.ppl_dataset import get_loaders
from transformers import TrainerCallback
from transformers.pytorch_utils import Conv1D
from transformers.utils.quantization_config import BitsAndBytesConfig
import json
import bitsandbytes as bnb
import optuna

device = "cuda" if torch.cuda.is_available() else "cpu"

BEGIN, END = [], []
class MemoryUsageCallback(TrainerCallback):
    def __init__(self, device):
        super().__init__()
        self.device = device
    
    def on_step_end(self, args, state, control, **kwargs):
        allocated, cached = self.get_memory_usage()
        END.append((allocated, cached))
        print(f">>>>>>>>> {END}")
        # print(f">>>>>>>>> Step {state.global_step} Memory Allocated: {allocated:.2f} GB, Memory Cached: {cached:.2f} GB")

    def on_step_begin(self, args, state, control, **kwargs):
        allocated, cached = self.get_memory_usage()
        BEGIN.append((allocated, cached))
        print(f"<<<<<<<<< {BEGIN}")
        # print(f"<<<<<<<<< Step {state.global_step} Memory Allocated: {allocated:.2f} GB, Memory Cached: {cached:.2f} GB")
    
    def get_memory_usage(self):
        allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 3)  # GB
        cached = torch.cuda.memory_reserved(self.device) / (1024 ** 3)  # GB
        return allocated, cached


def _replace_with_bnb_linear(
    model,
    bits_pattern_for_replace,
    modules_to_not_convert=None,
    current_key_name=None,
    quantization_config=None,
    has_been_replaced=False,
    module_path='',
):
    """
    Private method that wraps the recursion for module replacement.

    Returns the converted model and a boolean that indicates if the conversion has been successfull or not.
    """
    for name, module in model.named_children():
        current_full_path = f"{module_path}.{name}" if module_path else name
        # print('==========', current_full_path, '==========')
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if (isinstance(module, nn.Linear) or isinstance(module, Conv1D)) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            current_key_name_str = ".".join(current_key_name)
            if not any(
                (key + "." in current_key_name_str) or (key == current_key_name_str) for key in modules_to_not_convert
            ):
                # with init_empty_weights(): # 会导致权重初始化在meta上 🚀🚀🚀
                if current_full_path in bits_pattern_for_replace:
                    if isinstance(module, Conv1D):
                        in_features, out_features = module.weight.shape
                    else:
                        in_features = module.in_features
                        out_features = module.out_features

                    if bits_pattern_for_replace[current_full_path] == 8:
                        # print(f"{current_full_path} ### 8bit")
                        l8bit = bnb.nn.Linear8bitLt(
                            in_features,
                            out_features,
                            module.bias is not None,
                            has_fp16_weights=True, #quantization_config.llm_int8_has_fp16_weight,
                            threshold=quantization_config.llm_int8_threshold,
                            device='cpu',
                        )
                        l8bit.load_state_dict(module.state_dict())
                        model._modules[name] = l8bit
                        has_been_replaced = True
                    elif bits_pattern_for_replace[current_full_path] == 4:
                        extra_kwargs = (
                            {"quant_storage": quantization_config.bnb_4bit_quant_storage}
                            if "quant_storage" in list(signature(bnb.nn.Linear4bit).parameters)
                            else {}
                        )
                        # print(f"{current_full_path} ### 4bit")
                        l4bit =  bnb.nn.Linear4bit(
                            in_features,
                            out_features,
                            module.bias is not None,
                            quantization_config.bnb_4bit_compute_dtype,
                            compress_statistics=quantization_config.bnb_4bit_use_double_quant,
                            quant_type=quantization_config.bnb_4bit_quant_type,
                            device='cpu',
                            **extra_kwargs,
                        )
                        l4bit.load_state_dict(module.state_dict())
                        model._modules[name] = l4bit
                        has_been_replaced = True
                    # Store the module class in case we need to transpose the weight later
                    model._modules[name].source_cls = type(module)
                    # Force requires grad to False to avoid unexpected errors
                    model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = _replace_with_bnb_linear(
                module,
                bits_pattern_for_replace,
                modules_to_not_convert,
                current_key_name,
                quantization_config,
                has_been_replaced=has_been_replaced,
                module_path=current_full_path,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced


def replace_with_bnb_linear(model, bits_pattern_for_replace, modules_to_not_convert=None, current_key_name=None, quantization_config=None):
    modules_to_not_convert = ["lm_head"] if modules_to_not_convert is None else modules_to_not_convert
    model, has_been_replaced = _replace_with_bnb_linear(
        model, bits_pattern_for_replace, modules_to_not_convert, current_key_name, quantization_config
    )
    return model


def get_random_bits(num_layer):
    bits = [np.random.choice([4, 8], 1)[0] for i in range(num_layer)]
    print(Counter(bits))
    bits = np.array(bits).repeat(7)
    # TypeError: Object of type int64 is not JSON serializable
    return bits.tolist()


def get_linear_layer_names(model):
    return [(str(int(name.split('.')[0]) + 4) + "." + ".".join(name.split('.')[1:]))
            for name, module in model.model.layers[4:30].named_modules() if isinstance(module, torch.nn.Linear)]


def main(args, trial: optuna.trial.Trial=None):
    # Load Pruned Model
    print('Load Pruned Model ...')
    pruned_dict = torch.load(args.prune_model, map_location='cpu')
    tokenizer, model = pruned_dict['tokenizer'], pruned_dict['model']

    #############################################################
    # quantization 4bit or 8bit
    layer_names = get_linear_layer_names(model)
    assert len(layer_names) % 7 == 0
    if trial is None:
        bits_values = get_random_bits(int(len(layer_names) // 7))
    else:
        bits_values = []
        for l in range(int(len(layer_names) // 7)):
            sug_num_bits = trial.suggest_int(f"layer_{l}", 4, 8, 4)
            bits_values.extend([sug_num_bits] * 7)
    # Make sure there are enough bit values to match the layers
    if len(bits_values) < len(layer_names):
        raise ValueError("Not enough bit values provided for the number of layers.")
    bits_pattern = dict(zip(layer_names, bits_values))
    bits_pattern_for_replace = {"model.layers."+k : v for k, v in bits_pattern.items()}
    quantization_config_for_replace = BitsAndBytesConfig(
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type='nf4',
    )
    model = replace_with_bnb_linear(model, bits_pattern_for_replace, quantization_config=quantization_config_for_replace)
    
    for n, p in model.named_parameters():
        assert p.device == torch.device('cpu'), (n, p.dtype, p.device)

    print('Quantize Model ...')
    model.to(device)
    # for n, p in model.named_parameters():
    #     print(n, p.dtype, p.device)
    #############################################################

    gradient_accumulation_steps = args.batch_size // args.micro_batch_size
    if not args.no_instruction:
        prompter = Prompter(args.prompt_template_name)
    else:
        prompter = ZeroPrompter()

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=args.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < args.cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(data_point):
        if 'lamini' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                None,
                data_point["response"],
            )
        elif 'alpaca' in args.data_path.lower():
            full_prompt = prompter.generate_prompt(
                data_point["instruction"],
                data_point["input"],
                data_point["output"],
            )
        else:
            raise NotImplementedError

        tokenized_full_prompt = tokenize(full_prompt)
        if not args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"] if 'input' in data_point.keys() else None,
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if args.add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
        return tokenized_full_prompt

    def split_and_tokenizer(test_data, tokenizer, seq_len, field_name):
        test_ids = tokenizer("\n\n".join(test_data[field_name]), return_tensors='pt').input_ids[0]
        nsamples = test_ids.numel() // seq_len

        test_set = []
        for i in range(nsamples):
            batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
            test_set.append({
                'input_ids': batch,
                'labels': batch
            })
        return test_set


    # Set up the LoftQConfig with the dynamically created bits pattern
    loftq_config = LoftQConfig(
            loftq_bits=8,
            loftq_iter=1,
            bits_pattern=bits_pattern
        )
        
    # Continue setting up the LoraConfig and model
    config = LoraConfig(
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(","),
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        # init_lora_weights='pissa',
        # loftq_config=loftq_config
    )
    
    # gcks = {"use_reentrant": True}
    model = prepare_model_for_kbit_training(
        model, 
        use_gradient_checkpointing=False,
        # gradient_checkpointing_kwargs=gcks,
    )
    model = get_peft_model(model, config)
    # if hasattr(model, "enable_input_require_grads"):
    #     print("enable_input_require_grads")
    #     model.enable_input_require_grads()

    if device == 'cuda':
        model.bfloat16()
    # for n, p in model.named_parameters():
    #     print(n, p.dtype, p.device, p.requires_grad)
    model.print_trainable_parameters()

    # Load Train Dataset
    data = load_dataset(args.data_path)
    if args.cache_dataset and os.path.exists('datasets/cache/{}.bin'.format(args.data_path)):
        preprocess_data = torch.load('datasets/cache/{}.bin'.format(args.data_path))
        train_data, val_data = preprocess_data['train'], preprocess_data['val']
    else: 
        train_val = data["train"].train_test_split(
            test_size=args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_and_tokenize_prompt)
        )
        val_data = {
            args.data_path: train_val["test"].shuffle().map(generate_and_tokenize_prompt),
        }
        if args.cache_dataset and args.local_rank == 0:
            cache_file = 'datasets/cache/{}.bin'.format(args.data_path)
            cache_dir = '/'.join(cache_file.split('/')[:-1])
            directory = Path(cache_dir)
            directory.mkdir(parents=True, exist_ok=True)

            torch.save({
                'train': train_data, 'val': val_data
            }, cache_file)

    # Load Extra Validation Dataset
    if args.extra_val_dataset:
        from LLMPruner.datasets.ppl_dataset import get_wikitext2, get_ptb

        seq_len = 128
        for extra_dataset in args.extra_val_dataset.split(','):
            if 'wikitext2' in extra_dataset:
                _, test_data = get_wikitext2(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='text')
            if 'ptb' in extra_dataset:
                _, test_data = get_ptb(seq_len, None)
                test_data = split_and_tokenizer(test_data, tokenizer, seq_len, field_name='sentence')
            val_data[extra_dataset] = test_data

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=100,
            num_train_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            # fp16=True,
            # optim="adamw_torch",
            bf16=True,
            optim='paged_adamw_32bit',
            evaluation_strategy="steps",
            save_strategy="steps",
            logging_steps=10,
            logging_first_step=True,
            eval_steps=100,
            save_steps=100,
            max_steps = 200,
            # save_steps=10,
            # max_steps = 10, 
            output_dir=args.output_dir,
            save_total_limit=20,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=args.group_by_length,
            report_to="none",
            run_name=args.output_dir.split('/')[-1],
            metric_for_best_model="{}_loss".format(args.data_path),
            save_safetensors=False
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        callbacks=[MemoryUsageCallback(device)]  # Adding the callback here
    )
    model.config.use_cache = False
    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    trainer.train()

    model.state_dict = old_state_dict
    model.save_pretrained(args.output_dir, save_tensor=False)

    with open(os.path.join(args.output_dir, "memory_record.json"), 'w', encoding='utf-8') as f:
        json.dump({"begin": BEGIN, "end": END, "bits_pattern_for_replace": bits_pattern_for_replace}, f)
    
    _adapter_config_file = os.path.join(args.output_dir, "adapter_config.json")
    _adapter_config = json.load(open(_adapter_config_file))
    if 'loftq_config' in _adapter_config:
        _adapter_config["loftq_config"]['bits_pattern'] = bits_pattern
        with open(_adapter_config_file, 'w', encoding='utf-8') as f:
            json.dump(_adapter_config, f)

    return torch.cuda.max_memory_reserved() / (1024 ** 3) 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tuning Pruned LLM')

    # Model Type&Path
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--prune_model', type=str, help='prune model name')
    parser.add_argument('--data_path', type=str, default="yahma/alpaca-cleaned", help='data path')
    parser.add_argument('--cache_dataset', action="store_true", default=False)
    parser.add_argument('--extra_val_dataset', type=str, default=None, help='validation datasets. Split with ","')
    parser.add_argument('--output_dir', type=str, default="./lora-alpaca", help='output directory')

    # Training Hyperparameters
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--micro_batch_size', type=int, default=4, help='micro batch size')
    parser.add_argument('--num_epochs', type=int, default=5, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='learning rate')
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
   
    args = parser.parse_args()
    torch_version = int(torch.__version__.split('.')[1])
    args.torch_version = torch_version
    print(args)
    main(args)
