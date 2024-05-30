#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import argparse
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
from tqdm import tqdm
import nltk
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, BertConfig

import sys
sys.path.append(os.path.join(os.getcwd()))
from src.detector.dataset import DetectionDataset
from src.detector.transform import DetectionTransform
from src.detector.encoder import DetectEncoder


def parse_data(gold_fp, eval_fp, max_history=4):
    print("Loading data from {}".format(gold_fp))
    all_dialogs = []
    with open(gold_fp, 'r', encoding='utf-8') as f:
        for line in f:
            dialog = json.loads(line.strip())
            all_dialogs.append(dialog)
    
    data_examples = []
    for dialog in all_dialogs:
        # parse a multi-turn dialog
        user_role = dialog["agents"][0]["name"]
        system_role = dialog["agents"][1]["name"]
        character = "[character]: <%s>\t[character persona]: %s" % (dialog["agents"][1]["name"], dialog["agents"][1]["persona"])
        setting = "[setting]: %s\t[setting category]: %s\t[setting description]: %s\t[setting background]: %s" % (
            dialog["setting"]["name"], dialog["setting"]["category"], dialog["setting"]["description"], dialog["setting"]["background"]
        )
        round_index = 1
        if max_history > 0:
            # only use the last max_history utterances
            for idx, role in enumerate(dialog["character"]):
                if role == system_role:
                    char_list = dialog["character"][max(0, idx - max_history): idx]
                    conv_list = dialog["conversation"][max(0, idx - max_history): idx]
                    conv_history = ""
                    for char, conv in zip(char_list, conv_list):
                        conv_history += "<%s>: " % char + conv + " "
                    context = [setting, character, conv_history]
                    gold_response = dialog["conversation"][idx]
                    
                    data_examples.append({
                        'dialog_id': dialog["dialog_id"],
                        'round_id': round_index,
                        'context': context,
                        'gold_response': gold_response,
                    })
                    round_index += 1
        else:
            # use all historical utterances
            conv_history = ""
            for idx, role in enumerate(dialog["character"]):
                if role == user_role:
                    conv_history += "<%s>: " % user_role + dialog["conversation"][idx] + " "
                else:
                    context = [setting, character, conv_history]
                    gold_response = dialog["conversation"][idx]        
                
                    data_examples.append({
                        'dialog_id': dialog["dialog_id"],
                        'round_id': round_index,
                        'context': context,
                        'gold_response': gold_response,
                    })
                    conv_history += "<%s>: " % role + dialog["conversation"][idx] + " "
                    round_index += 1
    print("Loaded %d examples" % len(data_examples))

    eval_examples = []
    with open(eval_fp, 'r', encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            eval_examples.append(example)
    if len(data_examples) != len(eval_examples):
        raise ValueError("The number of examples in {} ({}) and {} ({}) are not equal.".format(
            gold_fp, eval_fp, len(data_examples), len(eval_examples)))

    for idx in range(len(data_examples)):
        # check whether each example is aligned
        if data_examples[idx]['gold_response'] != eval_examples[idx]['gold_response']:
            raise ValueError("The gold response of example {} is not aligned.".format(idx))
        # use the generated response for evaluation
        data_examples[idx]['response'] = eval_examples[idx]['generated_response']
        data_examples[idx]['label'] = -100
    
    return data_examples

def calc_consist(eval_fp, gold_fp, model_dir, max_history=4, max_length=400, compute_gold=False):
    """ Calculate consistency score """
    data_examples = parse_data(gold_fp, eval_fp, max_history=max_history)

    tokenizer = BertTokenizerFast.from_pretrained(model_dir)
    transform = DetectionTransform(tokenizer, max_len=max_length)
    dataset = DetectionDataset(data_dir=None, detection_transform=transform, loaded_data=data_examples)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.custom_collate)

    # TODO: load model from model_dir
    bert_model = "pretrained/bert-base-uncased"
    bert_config = BertConfig.from_json_file(os.path.join(bert_model, 'config.json'))
    model = DetectEncoder(bert_config, pretrained_model_dir=bert_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    state_save_path = os.path.join(model_dir, 'detect_pytorch_model.bin')
    print('Loading parameters from', state_save_path)
    model.load_state_dict(torch.load(state_save_path, map_location=device))

    consist_probs = []
    gold_consist_probs = []

    print("Calculating consistency probs...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids_batch, input_masks_batch, _ = batch
            logits = model(input_ids_batch, input_masks_batch, labels=None)[0]
            pred_probs = F.softmax(logits, -1)   # estimated probs of the positive class (0: negative, 1: positive)
            pos_y_probs = pred_probs[:, 1].data.cpu().numpy()   
            consist_probs.extend([float(y_prob) for y_prob in pos_y_probs])
    assert len(consist_probs) == len(data_examples)

    if compute_gold:
        print("Calculating consistency probs of gold responses...")
        for idx in range(len(data_examples)):
            data_examples[idx]['response'] = data_examples[idx]['gold_response']  # use the gold response for evaluation
        gold_dataset = DetectionDataset(data_dir=None, detection_transform=transform, loaded_data=data_examples)
        gold_dataloader = DataLoader(gold_dataset, batch_size=8, shuffle=False, collate_fn=dataset.custom_collate)
        
        with torch.no_grad():
            for batch in tqdm(gold_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids_batch, input_masks_batch, _ = batch
                logits = model(input_ids_batch, input_masks_batch, labels=None)[0]
                pred_probs = F.softmax(logits, -1)   # estimated probs of the positive class (0: negative, 1: positive)
                pos_y_probs = pred_probs[:, 1].data.cpu().numpy()   
                gold_consist_probs.extend([float(y_prob) for y_prob in pos_y_probs])
        assert len(gold_consist_probs) == len(data_examples)
    
    save_name = eval_fp.split('/')[-1].split('.')[0].replace('_output', '') + '_consist_probs.jsonl'
    save_path = os.path.join(os.path.dirname(eval_fp), save_name)
    with open(save_path, 'w', encoding='utf-8') as fw:
        for idx, example in enumerate(data_examples):
            obj = {
                'dialog_id': example['dialog_id'],
                'round_id': example['round_id'],
                'consist_prob': consist_probs[idx],
            }
            if compute_gold:
                obj['gold_consist_prob'] = gold_consist_probs[idx]
            fw.write(json.dumps(obj, ensure_ascii=False) + '\n')
    print("Saved consistency probs to %s" % save_path)
    
    avg_prob = np.average(consist_probs) 

    return avg_prob


def calc_f1(hyps, refs):
    """ Calculate word-level f1 score """
    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    for response, golden_response in zip(hyps, refs):
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = hit_char_total / pred_char_total if pred_char_total > 0 else 0
    r = hit_char_total / golden_char_total if golden_char_total > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1


def calc_bleu(hyps, refs):
    """ Calculate bleu 1/2 """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method1,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method1,
                weights=[0.5, 0.5, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def calc_distinct(seqs):
    """ Calculate intra/inter distinct 1/2 """
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return (intra_dist1, intra_dist2, inter_dist1, inter_dist2)


def load_eval_data(eval_fp, lower_case=True):
    gold_examples, pred_examples = [], []
    print("Loading eval data from %s" % eval_fp)
    with open(eval_fp, 'r', encoding='utf-8') as fr:
        for idx, line in enumerate(fr):
            example = json.loads(line)
            gold_response = example["gold_response"]
            pred_response = example["generated_response"]
            if lower_case:
                gold_response = gold_response.lower()
                pred_response = pred_response.lower()
            # English word-level tokenization
            gold_words = nltk.word_tokenize(gold_response)  
            pred_words = nltk.word_tokenize(pred_response)
            gold_examples.append(gold_words)
            pred_examples.append(pred_words)
    assert len(gold_examples) == len(pred_examples)
    print("Loaded %d examples" % len(gold_examples))
    
    return (gold_examples, pred_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, required=True)
    parser.add_argument("--gold_file", type=str, default=None)
    parser.add_argument("--detector_model", type=str, default=None)
    parser.add_argument("--max_history", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=400)
    parser.add_argument("--compute_gold", type=bool, default=False)
    args = parser.parse_args()

    golds, preds = load_eval_data(args.eval_file, lower_case=True)
    output_str = ""

    # calculate consistency
    if args.gold_file is not None and args.detector_model is not None:
        consist_prob= calc_consist(
            args.eval_file, args.gold_file, args.detector_model, 
            max_history=args.max_history, max_length=args.max_length,
            compute_gold=args.compute_gold
        )
        output_str += "Consist.Prob.: %.3f\n" % consist_prob

    # calculate f1
    f1 = calc_f1(preds, golds)

    # calculate bleu
    bleu1, bleu2 = calc_bleu(preds, golds)

    # calculate distinct
    _, _, inter_dist1, inter_dist2 = calc_distinct(preds)

    output_str += "Word-F1: %.2f%%\n" % (f1 * 100)
    output_str += "BLEU1: %.3f\n" % bleu1
    output_str += "BLEU2: %.3f\n" % bleu2
    output_str += "DISTINCT1: %.3f\n" % inter_dist1
    output_str += "DISTINCT2: %.3f" % inter_dist2

    print(output_str)
