# -*- coding: utf-8 -*-
# This file is modified from https://github.com/chijames/Poly-Encoder
import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import random
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel, BertConfig, BertTokenizer, BertTokenizerFast
from transformers.optimization import get_linear_schedule_with_warmup

from dataset import DetectionDataset, SelectionDataset
from transform import DetectionTransform, SelectionSequentialTransform, SelectionJoinTransform
from encoder import DetectEncoder, DetectPolyEncoder
from sklearn import metrics
from curve import plot_calibration_curve, plot_roc_curve

import logging
logging.basicConfig(level=logging.ERROR)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def eval_detection_model(dataloader, model, device, architecture='detect', return_pos_probs=False):
    model.eval()
    eval_loss, nb_eval_steps= 0, 0
    nb_eval_examples, nb_hit_examples = 0, 0
    pos_probs, preds, trues = [], [], []

    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            if architecture == 'detect':
                input_ids_batch, input_masks_batch, labels_batch = batch
                logits, loss = model(input_ids_batch, input_masks_batch, labels_batch)
            else:
                context_token_ids_list_batch, context_input_masks_list_batch, \
                response_token_ids_list_batch, response_input_masks_list_batch, labels_batch = batch
                logits, loss = model(context_token_ids_list_batch, context_input_masks_list_batch,
                                response_token_ids_list_batch, response_input_masks_list_batch,
                                labels_batch)
        
        eval_loss += loss.item()
        pred_probs = F.softmax(logits, -1)
        if return_pos_probs:
            pos_y_probs = pred_probs[:, 1].data.cpu().numpy()   # estimated probs of the positive class (0: negative, 1: positive)
            pos_probs.extend([y_prob for y_prob in pos_y_probs])
        
        pred_labels = torch.argmax(pred_probs, -1)
        preds.extend([y_pred.item() for y_pred in pred_labels])
        true_labels = labels_batch.view(-1)
        trues.extend([y_true.item() for y_true in true_labels])

        num_hits = torch.eq(pred_labels, true_labels).sum().item()
        nb_eval_steps += 1
        nb_eval_examples += true_labels.size(0)
        nb_hit_examples += num_hits
        
    eval_loss = eval_loss / nb_eval_steps
    eval_acc = nb_hit_examples / nb_eval_examples
    eval_p = metrics.precision_score(trues, preds)
    eval_r = metrics.recall_score(trues, preds)
    eval_f1 = metrics.f1_score(trues, preds)

    if return_pos_probs:
        result = {
            'eval_examples': nb_eval_examples,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'eval_p': eval_p,
            'eval_r': eval_r,
            'eval_f1': eval_f1,
            'y_true': np.array(trues),
            'y_pred': np.array(preds),
            'y_pos_probs': np.array(pos_probs)
        }
    else:
        result = {
            'eval_examples': nb_eval_examples,
            'eval_loss': eval_loss,
            'eval_acc': eval_acc,
            'eval_p': eval_p,
            'eval_r': eval_r,
            'eval_f1': eval_f1
        }
    return result


def main(args):
    MODEL_CLASSES = {
        'bert': (BertConfig, BertTokenizerFast, BertModel),
    }
    ConfigClass, TokenizerClass, BertModelClass = MODEL_CLASSES[args.model_type]

    ## init dataset and bert model
    tokenizer = TokenizerClass.from_pretrained(args.bert_model)
    if args.architecture == 'detect':
        detection_transform = DetectionTransform(tokenizer, max_len=args.max_length)
    elif args.architecture == 'poly':
        context_transform = SelectionJoinTransform(tokenizer=tokenizer, max_len=args.max_contexts_length)
        response_transform = SelectionSequentialTransform(tokenizer=tokenizer, max_len=args.max_response_length)
    else:
        raise Exception('Unknown architecture.')

    if not args.eval:
        if args.architecture == 'detect':
            train_dataset = DetectionDataset(args.data_dir, detection_transform, data_split='train')
            val_dataset = DetectionDataset(args.data_dir, detection_transform, data_split='eval', sample_cnt=5000)
        else:
            train_dataset = SelectionDataset(args.data_dir, context_transform, response_transform, concat_transform=None, 
                                         data_split='train', mode=args.architecture)
            val_dataset = SelectionDataset(args.data_dir, context_transform, response_transform, concat_transform=None, 
                                       data_split='eval', sample_cnt=5000, mode=args.architecture)
        
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
                                      collate_fn=train_dataset.custom_collate, shuffle=True)
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    else: 
        # eval the performance of classifier on the validation set
        if args.architecture == 'detect':
            val_dataset = DetectionDataset(args.data_dir, detection_transform, data_split='eval')
        else:
            val_dataset = SelectionDataset(args.data_dir, context_transform, response_transform, concat_transform=None, 
                                       data_split='eval', mode=args.architecture)
        
    val_dataloader = DataLoader(val_dataset, batch_size=args.eval_batch_size, 
                                collate_fn=val_dataset.custom_collate, shuffle=False)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    shutil.copyfile(os.path.join(args.bert_model, 'vocab.txt'), os.path.join(args.output_dir, 'vocab.txt'))
    shutil.copyfile(os.path.join(args.bert_model, 'config.json'), os.path.join(args.output_dir, 'config.json'))
    if os.path.exists(os.path.join(args.bert_model, 'tokenizer.json')):
        shutil.copyfile(os.path.join(args.bert_model, 'tokenizer.json'), os.path.join(args.output_dir, 'tokenizer.json'))
    if os.path.exists(os.path.join(args.bert_model, 'tokenizer_config.json')):
        shutil.copyfile(os.path.join(args.bert_model, 'tokenizer_config.json'), os.path.join(args.output_dir, 'tokenizer_config.json'))
    log_wf = open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8')
    print(args, file=log_wf)

    state_save_path = os.path.join(args.output_dir, '{}_pytorch_model.bin'.format(args.architecture))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## build model
    bert_config = ConfigClass.from_json_file(os.path.join(args.bert_model, 'config.json'))
    
    if args.architecture == 'detect':
        model = DetectEncoder(bert_config, pretrained_model_dir=args.bert_model, num_labels=2)
    elif args.architecture == 'poly':
        model = DetectPolyEncoder(bert_config, pretrained_model_dir=args.bert_model, poly_m=args.poly_m, num_labels=2)
    else:
        raise Exception('Unknown architecture.')
    model.to(device)
    
    # evaluate the model
    if args.eval:
        print('Loading parameters from', state_save_path)
        model.load_state_dict(torch.load(state_save_path, map_location=device))
        print('Evaluating the model ...')
        test_result = eval_detection_model(val_dataloader, model, device, architecture=args.architecture, return_pos_probs=True)
        print('Eval result: ')
        print('Eval examples: %d | Loss: %.4f | Acc: %.4f | P: %.4f | R: %.4f | F1: %.4f' % (
            test_result['eval_examples'], test_result['eval_loss'], test_result['eval_acc'], test_result['eval_p'], test_result['eval_r'], test_result['eval_f1']))
        cache_path = os.path.join(args.output_dir, '{}_eval_cache.pkl'.format(args.architecture))
        with open(cache_path, 'wb') as f:
            pickle.dump(test_result, f)
        print('Save eval result to [%s]' % cache_path)
        if args.plot:
            plot_roc_curve(test_result['y_true'], test_result['y_pos_probs'])
            plot_calibration_curve(test_result['y_true'], test_result['y_pos_probs'], n_bins=10)
        exit()

    if args.plot:
        cache_path = os.path.join(args.output_dir, '{}_eval_cache.pkl'.format(args.architecture))
        if not os.path.exists(cache_path):
            if args.eval:
                print('Loading parameters from', state_save_path)
                model.load_state_dict(torch.load(state_save_path, map_location=device))
                print('Evaluating the model ...')
                test_result = eval_detection_model(val_dataloader, model, device, architecture=args.architecture, return_pos_probs=True)
                print('Eval result: ')
                print('Eval examples: %d | Loss: %.4f | Acc: %.4f | P: %.4f | R: %.4f | F1: %.4f' % (
                    test_result['eval_examples'], test_result['eval_loss'], test_result['eval_acc'], test_result['eval_p'], test_result['eval_r'], test_result['eval_f1']))
                cache_path = os.path.join(args.output_dir, '{}_eval_cache.pkl'.format(args.architecture))
                with open(cache_path, 'wb') as f:
                    pickle.dump(test_result, f)
                print('Save eval result to [%s]' % cache_path)
            else:
                raise FileNotFoundError('Cache file {}_eval_cache.pkl not found. Please run with --eval first.'.format(args.architecture))
        # plot calibration curve
        with open(cache_path, 'rb') as f:
            test_result = pickle.load(f)
        plot_roc_curve(test_result['y_true'], test_result['y_pos_probs'])
        plot_calibration_curve(test_result['y_true'], test_result['y_pos_probs'], n_bins=10)
        exit()
        
    no_decay = ["bias", "LayerNorm.weight"]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    print_freq = args.print_freq // args.gradient_accumulation_steps
    eval_freq = min(len(train_dataloader) // 2, args.eval_freq)
    eval_freq = eval_freq // args.gradient_accumulation_steps
    print('Print freq:', print_freq, "Eval freq:", eval_freq)

    epoch_start = 1
    global_step = 0
    best_eval_loss = float('inf')
    for epoch in range(epoch_start, int(args.num_train_epochs) + 1):
        tr_loss = 0
        nb_tr_steps = 0
        with tqdm(total=len(train_dataloader)//args.gradient_accumulation_steps) as bar:
            for step, batch in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                if args.architecture == 'detect':
                    input_ids_batch, input_masks_batch, labels_batch = batch
                    logits, loss = model(input_ids_batch, input_masks_batch, labels_batch)
                else:
                    context_token_ids_list_batch, context_input_masks_list_batch, \
                    response_token_ids_list_batch, response_input_masks_list_batch, labels_batch = batch
                    logits, loss = model(context_token_ids_list_batch, context_input_masks_list_batch,
                                 response_token_ids_list_batch, response_input_masks_list_batch,
                                 labels_batch)

                loss = loss / args.gradient_accumulation_steps
                
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    nb_tr_steps += 1
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if nb_tr_steps and nb_tr_steps % print_freq == 0:
                        bar.update(min(print_freq, nb_tr_steps))
                        print('Epoch: %d Global step: %d Loss: %.4f' % (epoch, global_step, tr_loss / nb_tr_steps))
                        log_wf.write('Epoch: %d Global step: %d Loss: %.4f\n' % (epoch, global_step, tr_loss / nb_tr_steps))

                    if global_step and global_step % eval_freq == 0:
                        val_result = eval_detection_model(val_dataloader, model, device, architecture=args.architecture)
                        print('Global step: %d VAL res:\n' % global_step, val_result)
                        log_wf.write('Global step: %d VAL res:\n' % global_step)
                        log_wf.write(str(val_result) + '\n')

                        if val_result['eval_loss'] < best_eval_loss:
                            best_eval_loss = val_result['eval_loss']
                            val_result['best_eval_loss'] = best_eval_loss
                            # save model
                            print('Saved to [%s]' % state_save_path)
                            log_wf.write('Saved to [%s]\n' % state_save_path)
                            torch.save(model.state_dict(), state_save_path)
                log_wf.flush()

        # add a eval step after each epoch
        val_result = eval_detection_model(val_dataloader, model, device, architecture=args.architecture)
        print('Epoch: %d Global step: %d VAL res:\n' % (epoch, global_step), val_result)
        log_wf.write('Global step: %d VAL res:\n' % global_step)
        log_wf.write(str(val_result) + '\n')

        if val_result['eval_loss'] < best_eval_loss:
            best_eval_loss = val_result['eval_loss']
            val_result['best_eval_loss'] = best_eval_loss
            # save model
            print('Saved to [%s]' % state_save_path)
            log_wf.write('Saved to [%s]\n' % state_save_path)
            torch.save(model.state_dict(), state_save_path)
        print('Epoch: %d Global step: %d Loss: %.4f' % (epoch, global_step, tr_loss / nb_tr_steps))
        log_wf.write('Epoch: %d Global step: %d Loss: %.4f' % (epoch, global_step, tr_loss / nb_tr_steps))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval", action="store_true", help="Whether to run evaluation.")
    parser.add_argument("--plot", action="store_true", help="Whether to plot calibration curve after evaluation.")
    parser.add_argument("--data_dir", default='data/light', type=str)
    parser.add_argument("--output_dir", required=True, type=str)

    parser.add_argument("--model_type", default='bert', type=str)
    parser.add_argument("--bert_model", default='pretrained/bert-base-uncased', type=str)
    parser.add_argument("--architecture", default='detect', type=str, help='[poly, detect]')
    
    parser.add_argument("--max_length", default=400, type=int)
    parser.add_argument("--max_contexts_length", default=256, type=int)
    parser.add_argument("--max_response_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
    parser.add_argument("--print_freq", default=100, type=int, help="Log frequency")
    parser.add_argument("--eval_freq", default=1000, type=int, help="Eval frequency")

    parser.add_argument("--poly_m", default=16, type=int, help="Number of m of polyencoder")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--warmup_steps", default=100, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--num_train_epochs", default=10.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, 
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level", type=str, default="O1", 
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html",)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    args = parser.parse_args()
    print(args)
    set_seed(args)

    main(args)
    