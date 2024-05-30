"""Inference for models."""
import json
import os
import fire
import numpy as np
import random
from typing import Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from accelerate import Accelerator

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig,
)

from peft import PeftConfig
from model import DialogModel, ADAPTER_SYSTEM
from data_utils import Preprocessor
from data_collator import DataCollatorWithPadding


def set_seed(random_seed):
    """Set random seed."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(random_seed)

def load_model(
    model_path: str,
    load_8bit: bool = False,
    bf16: bool = False,
    q_lora: bool = True,
    padding_side: str = "left",
    truncation_side: str = "left"
):
    """Load a model from saved path."""

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        if load_8bit:
            raise ValueError("8-bit quantization is not supported for multi-gpu inference.")
    
    from_pretrained_kwargs = {
        "device_map": device_map,
        "torch_dtype": torch.bfloat16 if bf16 else torch.float16,
        "load_in_8bit": load_8bit
    }
    
    if q_lora:
        from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=from_pretrained_kwargs["torch_dtype"] ,
        )
    else:
        from_pretrained_kwargs["quantization_config"] = None
    
    # Load model
    config_path = os.path.join(model_path, ADAPTER_SYSTEM)
    config = PeftConfig.from_pretrained(config_path)
    base_model_path = config.base_model_name_or_path
    if "peft" in base_model_path:
        raise ValueError(
            f"PeftModelAdapter cannot load a base model with 'peft' in the name: {config.base_model_name_or_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        padding_side=padding_side,
        truncation_side=truncation_side,
        rust_remote_code=True, 
        use_fast=False,
        legacy=False
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **from_pretrained_kwargs
    )
    model = DialogModel.from_pretrained(base_model, model_path)
    model.eval()

    return model, tokenizer


def get_response(output: str, bos_token: str, eos_token: str) -> str:
    response = output.split(eos_token)[0].strip()
    if bos_token in response:
        response = response.replace(bos_token, "").strip()
    return response

def run_generation(accelerator: Accelerator, dataloader, tokenizer, model, generation_config):
    model, dataloader = accelerator.prepare(model, dataloader)
    output_sequences = []

    for sample in tqdm(dataloader):
        unwrapped_model = accelerator.unwrap_model(model)
        with torch.inference_mode():
            model_kwargs = {
                "instruct_ids": sample.instruct_ids,
                "role_ids": sample.role_ids,
                "context_ids": sample.context_ids,
                "target_ids": sample.target_ids,
                "instruct_mask": sample.instruct_mask,
                "attention_mask": sample.attention_mask,
            }
            generated_tokens = unwrapped_model.generate(
                generation_config=generation_config,
                **model_kwargs
            ).sequences

            generated_tokens = accelerator.pad_across_processes(
                generated_tokens, dim=1, pad_index=tokenizer.pad_token_id)

            generated_tokens = accelerator.gather_for_metrics(generated_tokens)
        
        outputs = [tokenizer.decode(ids) for ids in generated_tokens]
        output_strs = [get_response(output, tokenizer.bos_token, tokenizer.eos_token) for output in outputs]
        output_sequences.extend(output_strs)

    return output_sequences


def main(
    model_path: str,
    test_data_path: str,
    test_unseen_data_path: Optional[str] = None,
    cache_path: str = "caches",
    output_dir: str = "results",
    
    load_8bit: bool = False,
    bf16: bool = False,
    q_lora: bool = True,
    padding_side: str = "left",   # left for inference
    truncation_side: str = "left",
    max_instruction_length: int = 256,
    max_utterance_length: int = 108,
    max_rounds: int = 10,
    num_proc: int = 8,   # num_proc > 1 will speed up tokenization

    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = -1,   # -1 means disable
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    random_seed: int = 42,
):
    set_seed(random_seed)

    accelerator = Accelerator()

    if accelerator.is_local_main_process:
        print(
            f"Model generation with params:\n"
            f"model_path: {model_path}\n"
            f"test_data_path: {test_data_path}\n"
            f"test_unseen_data_path: {test_unseen_data_path}\n"
            f"cache_path: {cache_path}\n"
            f"output_dir: {output_dir}\n"
            f"load_8bit: {load_8bit}\n"
            f"bf16: {bf16}\n"
            f"q_lora: {q_lora}\n"
            f"padding_side: {padding_side}\n"
            f"truncation_side: {truncation_side}\n"
            f"max_instruction_length: {max_instruction_length}\n"
            f"max_utterance_length: {max_utterance_length}\n"
            f"max_rounds: {max_rounds}\n"
            f"num_proc: {num_proc}\n"
            f"max_new_tokens: {max_new_tokens}\n"
            f"do_sample: {do_sample}\n"
            f"temperature: {temperature}\n"
            f"top_p: {top_p}\n"
            f"top_k: {top_k}\n"
            f"num_beams: {num_beams}\n"
            f"repetition_penalty: {repetition_penalty}\n"
            f"random_seed: {random_seed}\n"
        )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    with accelerator.main_process_first():
        # Model
        model, tokenizer = load_model(
            model_path,
            load_8bit=load_8bit,
            bf16=bf16,
            q_lora=q_lora,
            padding_side=padding_side,
            truncation_side=truncation_side
        )

        # Set tokenizer's padding token
        if model.config.model_type == "llama" or model.config.model_type == "mistral":
            tokenizer.pad_token = tokenizer.unk_token

        if accelerator.is_local_main_process:
            print(f"Loaded model from {model_path}")
            print(model)
            print(f"Tokenizer special tokens: {tokenizer.special_tokens_map}")
            print(f"Tokenizer pad_token_id: {tokenizer.pad_token_id}")
            print(f"Tokenizer unk_token_id: {tokenizer.unk_token_id}")
            print(f"Tokenizer bos_token_id: {tokenizer.bos_token_id}")
            print(f"Tokenizer eos_token_id: {tokenizer.eos_token_id}")

        generation_config = GenerationConfig(
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_new_tokens=1,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True
        )

        dataset_cls = Preprocessor(
            tokenizer=tokenizer,
            max_instruction_length=max_instruction_length,
            max_utterance_length=max_utterance_length,
            max_rounds=max_rounds
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        cache_dir = os.path.join(cache_path, "caches_midi")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        data_files = {"test": test_data_path}
        if test_unseen_data_path is not None:
            if os.path.exists(test_unseen_data_path):
                data_files["test_unseen"] = test_unseen_data_path
            else:
                print(f"Ignore: test_unseen_data_path {test_unseen_data_path} does not exist.")
            
        if accelerator.is_local_main_process:
            print(f"Loading data from {test_data_path}")
            if "test_unseen" in data_files:
                print(f"Loading data from {test_unseen_data_path}")

        load_data = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
        test_dataset = load_data["test"].map(dataset_cls.preprocess_infer, num_proc=num_proc)

        # Remove columns that aren't used by the model
        # otherwise, the dataset can't be converted to tensors
        signature_columns = ["instruct_ids", "role_ids", "context_ids", "target_ids", "instruct_mask", "attention_mask"]
        ignored_columns = list(set(test_dataset.column_names) - set(signature_columns))
        test_dataset = test_dataset.remove_columns(ignored_columns)

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=1,   # batch_size=1 for inference
            collate_fn=data_collator,
            shuffle=False,
        )

        if "test_unseen" in load_data:
            test_unseen_dataset = load_data["test_unseen"].map(dataset_cls.preprocess_infer, num_proc=num_proc)
            test_unseen_dataset = test_unseen_dataset.remove_columns(ignored_columns)
            test_unseen_dataloader = DataLoader(
                test_unseen_dataset, 
                batch_size=1,   # batch_size=1 for inference
                collate_fn=data_collator,
                shuffle=False,
            )
        else:
            test_unseen_dataloader = None
    
    
    if accelerator.is_local_main_process:
        print("Generating on the test set ...")

    results = run_generation(
        accelerator, 
        dataloader=test_dataloader,
        tokenizer=tokenizer,
        model=model,
        generation_config=generation_config
    )
    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        # Save test results
        save_gold = load_data["test"]
        save_path = os.path.join(output_dir, test_data_path.split('/')[-1].split('.')[0] + '_output.jsonl')
        with open(save_path, 'w', encoding='utf-8') as f:
            for item, res in zip(save_gold, results):
                result = {
                    "dialog_id": item["dialog_id"],
                    "turn_id": item["turn_id"],
                    "gold_response": item["conversations"][-1]["value"],
                    "generated_response": res,
                }
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        print("Saved {} resulted examples to {}.".format(len(results), save_path))
    

    if test_unseen_dataloader is not None:
        if accelerator.is_local_main_process:
            print("Generating on the test-unseen set ...")
        
        unseen_results = run_generation(
            accelerator, 
            dataloader=test_unseen_dataloader,
            tokenizer=tokenizer,
            model=model,
            generation_config=generation_config
        )
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            # Save test unseen results
            save_gold = load_data["test_unseen"]
            save_path = os.path.join(output_dir, test_unseen_data_path.split('/')[-1].split('.')[0] + '_output.jsonl')
            with open(save_path, 'w', encoding='utf-8') as f:
                for item, res in zip(save_gold, unseen_results):
                    result = {
                        "dialog_id": item["dialog_id"],
                        "turn_id": item["turn_id"],
                        "gold_response": item["conversations"][-1]["value"],
                        "generated_response": res,
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print("Saved {} resulted examples to {}.".format(len(unseen_results), save_path))    


if __name__ == "__main__":
    fire.Fire(main)
