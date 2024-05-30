# Usage: deepspeed finetune.py --deepspeed <$PATH_TO_DEEPSPEED_CONFIG>

# Adapted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import logging
import pathlib
import os
import sys
import numpy as np
import random
from dataclasses import dataclass, field
from typing import List, Optional
import torch

from peft import (
    LoraConfig, 
    prepare_model_for_kbit_training
)
from transformers import (
    TrainingArguments,
    HfArgumentParser, 
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    Trainer, 
    BitsAndBytesConfig,
)
from transformers.integrations import deepspeed

from data_utils import rank0_print, make_supervised_data_module
from data_collator import DataCollatorWithPadding
from model import DialogModel


def set_seed(random_seed):
    """Set random seed."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_seed)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="yahma/llama-7b-hf")
    weight_beta: Optional[float] = field(default=1.0)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    cache_path: str = field(
        default="caches", metadata={"help": "Path to cache all data."}
    )
    padding_side: str = field(
        default="left",
        metadata={"help": "Padding side (right or left) for padding to max_length"}
    )
    truncation_side: str = field(
        default="left",
        metadata={"help": "Truncation_side (right or left) for input sequences"}
    )
    max_instruction_length: int = field(
        default=256,
        metadata={"help": "Maximum instruction length."}
    ) 
    max_utterance_length: int = field(
        default=72,
        metadata={"help": "Maximum utterance length."}
    )    
    max_rounds: int = field(
        default=10,
        metadata={"help": "Maximum number of dialog rounds."}
    )
    num_proc: int = field(
        default=8, metadata={"help": "Number of processes for data loading."}
    )


@dataclass
class FinetuningArguments(TrainingArguments):
    # The default arguments are defined in transformers.TrainingArguments
    # The following arguments are specified for reference
    output_dir: str = field(
        default=None, metadata={"help": "The output directory where the model checkpoints will be written."}
    )
    load_in_8bit: bool = False
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    num_train_epochs: float = 3.0
    evaluation_strategy: Optional[str] = "no"
    learning_rate: float = 2e-5
    weight_decay: float = 0.001
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    optim: str = "adamw_torch"
    fp16: bool = True
    bf16: bool = False
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 100
    save_strategy: Optional[str] = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    local_rank: int = field(default=0, metadata={"help": "Local rank of the process."})
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    random_seed: int = 42

@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def train():

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, FinetuningArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    set_seed(training_args.random_seed)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    if lora_args.q_lora:
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP and ZeRO3 are both currently incompatible with QLoRA."
            )
    
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map=device_map,
        torch_dtype=compute_dtype,
        quantization_config=BitsAndBytesConfig(  
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if lora_args.q_lora
        else None,
        trust_remote_code=True,
    )
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules,
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=training_args.gradient_checkpointing
    )
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    # Wrap the model with the DialogModel class
    model = DialogModel(model, lora_config, weight_beta=model_args.weight_beta)
    rank0_print(model)
    
    if training_args.deepspeed is not None and training_args.local_rank == 0:
        model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side=data_args.padding_side,
        truncation_side=data_args.truncation_side,
        trust_remote_code=True,
        use_fast=False,
        legacy=False,
    )
    # Set tokenizer's padding token and padding side
    if model.config.model_type == "llama" or model.config.model_type == "mistral":
        tokenizer.pad_token = tokenizer.unk_token

    rank0_print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
    rank0_print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
    rank0_print(f"Tokenizer unk_token_id: {tokenizer.unk_token_id}")
    rank0_print(f"Tokenizer bos_token_id: {tokenizer.bos_token_id}")
    rank0_print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, data_collator=data_collator, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    if training_args.local_rank == 0:
        model.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train()
