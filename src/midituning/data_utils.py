# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
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
import os
from typing import Dict
import torch.distributed as dist
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def rank0_print(*args):
    if dist.get_rank() == 0:
        print(*args)


class Preprocessor(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer,
                 max_instruction_length: int = 512, 
                 max_utterance_length: int = 128,
                 max_rounds: int = 10):
        self.tokenizer = tokenizer
        self.max_instruction_length = max_instruction_length
        self.max_utterance_length = max_utterance_length
        self.max_rounds = max_rounds
    
    def preprocess(self, source) -> Dict:
        roles = ("human", "gpt")
        source_instruct = source["instruction"]
        
        source_convs = source["conversations"]
        if source_convs[0]["from"] != roles[0]:
            # Skip the first one if it is not from human
            source_convs = source_convs[1:]
        
        role_ids = []
        messages = []
        for j, sentence in enumerate(source_convs):
            role = sentence["from"]
            message = sentence["value"]
            if message == '':
                message = '<none>'
            message += self.tokenizer.eos_token
            assert role == roles[j % 2], f"{source}"
            role_ids.append(roles.index(role))
            messages.append(message)
            if j >= self.max_rounds * 2 - 1:
                break
        
        # Tokenize instructions
        instruct_ids = self.tokenizer(
            source_instruct,
            return_tensors="pt",
            max_length=self.max_instruction_length,
            truncation=True,
        ).input_ids[0]
        instruct_mask = instruct_ids.ne(self.tokenizer.pad_token_id)

        # Tokenize conversations
        context_ids = []
        target_ids = []
        attention_mask = []
        for message in messages:
            message_ids = self.tokenizer(
                message,
                return_tensors="pt",
                max_length=self.max_utterance_length,
                truncation=True,
            ).input_ids[0]
            context_ids.append(message_ids)
            target_ids.append(message_ids.clone())
            attention_mask.append(message_ids.ne(self.tokenizer.pad_token_id))

        return dict(
            instruct_ids=instruct_ids,
            role_ids=role_ids,
            context_ids=context_ids,
            target_ids=target_ids,
            instruct_mask=instruct_mask,
            attention_mask=attention_mask,
        )
    
    def preprocess_infer(self, source) -> Dict:
        roles = ("human", "gpt")
        source_instruct = source["instruction"]
        source_convs = source["conversations"]
        
        # Remove the last system utterance
        assert source_convs[-1]["from"] == "gpt"
        source_convs[-1]["value"] = self.tokenizer.bos_token
        
        role_ids = []
        messages = []
        for j, sentence in enumerate(source_convs):
            role = sentence["from"]
            message = sentence["value"]
            if message == '':
                message = '<none>'
            if j != len(source_convs) - 1:
                message += self.tokenizer.eos_token
            assert role == roles[j % 2], f"{source}"
            role_ids.append(roles.index(role))
            messages.append(message)
            if j >= self.max_rounds * 2 - 1:
                break
        
        # Tokenize instructions
        instruct_ids = self.tokenizer(
            source_instruct,
            return_tensors="pt",
            max_length=self.max_instruction_length,
            truncation=True,
        ).input_ids[0]
        instruct_mask = instruct_ids.ne(self.tokenizer.pad_token_id)

        # Tokenize conversations
        context_ids = []
        target_ids = []
        attention_mask = []
        for message in messages:
            message_ids = self.tokenizer(
                message,
                return_tensors="pt",
                max_length=self.max_utterance_length,
                truncation=True,
            ).input_ids[0]
            context_ids.append(message_ids)
            target_ids.append(message_ids.clone())
            attention_mask.append(message_ids.ne(self.tokenizer.pad_token_id))

        return dict(
            instruct_ids=instruct_ids,
            role_ids=role_ids,
            context_ids=context_ids,
            target_ids=target_ids,
            instruct_mask=instruct_mask,
            attention_mask=attention_mask,
        )
    

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    dataset_cls = Preprocessor(tokenizer=tokenizer,
                               max_instruction_length=data_args.max_instruction_length,
                               max_utterance_length=data_args.max_utterance_length,
                               max_rounds=data_args.max_rounds)

    rank0_print(f"Loading training data from {data_args.data_path}")
    cache_dir = os.path.join(data_args.cache_path, "caches_midi")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    data_files = {"train": data_args.data_path}

    if data_args.eval_data_path:
        rank0_print(f"Loading eval data from {data_args.eval_data_path}")
        data_files["eval"] = data_args.eval_data_path
        
        dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)

        # num_proc > 1 will speed up the tokenization
        train_dataset = dataset["train"].map(dataset_cls.preprocess, num_proc=data_args.num_proc)
        eval_dataset = dataset["eval"].map(dataset_cls.preprocess, num_proc=data_args.num_proc)
    else:
        dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
        # num_proc > 1 will speed up the tokenization
        train_dataset = dataset["train"].map(dataset_cls.preprocess, num_proc=data_args.num_proc)
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)
