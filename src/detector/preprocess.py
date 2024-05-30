import os
import json
import argparse
import random
from tqdm import tqdm


def sample_responses_light(current_id, current_character, dialog_list, num_negative):
    candidates = []
    while len(candidates) < num_negative:
        cand_dialog = random.choice(dialog_list)
        if cand_dialog["dialog_id"] != current_id and \
            cand_dialog["agents"][0]["name"] != current_character and \
                cand_dialog["agents"][1]["name"] != current_character:
            cand_response = random.choice(cand_dialog["conversation"])
            if cand_response not in candidates:
                candidates.append(cand_response)
    return candidates

def parse_dataset_light(data_dir, data_mode, max_history=4, num_negative=1):
    assert data_mode in ['train', 'eval']
    if data_mode == 'train':
        print("Sampling positive and negative samples for training...")
        # since we train the classifier to identify the consistency between the context and a dialogue model's generated response, 
        # we use all the data (except validation set) for training
        data_splits = ['train', 'test', 'test_unseen']
        save_fp = os.path.join(data_dir, 'detector_train.jsonl')
    else:
        print("Sampling positive and negative samples for evaluation...")
        data_splits = ['valid']
        save_fp = os.path.join(data_dir, 'detector_eval.jsonl')
        
    data_samples = []
    for split in data_splits:
        data_fp = os.path.join(data_dir, 'light_{}.jsonl'.format(split))
        print("Loading data from {}".format(data_fp))
        all_dialogs = []
        with open(data_fp, 'r', encoding='utf-8') as f:
            for line in f:
                dialog = json.loads(line.strip())
                all_dialogs.append(dialog)
        
        for dialog in tqdm(all_dialogs):
            # parse a multi-round dialog
            user_role = dialog["agents"][0]["name"]
            system_role = dialog["agents"][1]["name"]
            character = "[character]: <%s>\t[character persona]: %s" % (dialog["agents"][1]["name"], dialog["agents"][1]["persona"])
            setting = "[setting]: %s\t[setting category]: %s\t[setting description]: %s\t[setting background]: %s" % (
                dialog["setting"]["name"], dialog["setting"]["category"], dialog["setting"]["description"], dialog["setting"]["background"]
            )
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
                        
                        pos_response = dialog["conversation"][idx]
                        neg_responses = []
                        for jdx, role in enumerate(dialog["character"]):
                            if role == user_role and (not dialog["conversation"][jdx] in conv_list):
                                neg_responses.append(dialog["conversation"][jdx])
                       
                        if len(neg_responses) > num_negative:
                            neg_responses = random.sample(neg_responses, num_negative)
                        elif len(neg_responses) < num_negative:
                            neg_responses += sample_responses_light(dialog["dialog_id"], system_role, all_dialogs, num_negative - len(neg_responses))
                        assert len(neg_responses) == num_negative
                        data_samples.append({
                            'context': context,
                            'pos_response': pos_response,
                            'neg_responses': neg_responses
                        })
            else:
                # use all historical utterances
                conv_history = ""
                for idx, role in enumerate(dialog["character"]):
                    if role == user_role:
                        conv_history += "<%s>: " % user_role + dialog["conversation"][idx] + " "
                    else:
                        context = [setting, character, conv_history]
                        pos_response = dialog["conversation"][idx]
                        neg_responses = []
                        for role, response in zip(dialog["character"], dialog["conversation"]):
                            if role == user_role:
                                neg_responses.append(response)
                        
                        if len(neg_responses) > num_negative:
                            neg_responses = random.sample(neg_responses, num_negative)
                        elif len(neg_responses) < num_negative:
                            neg_responses += sample_responses_light(dialog["dialog_id"], system_role, all_dialogs, num_negative - len(neg_responses))
                        assert len(neg_responses) == num_negative
                        data_samples.append({
                            'context': context,
                            'pos_response': pos_response,
                            'neg_responses': neg_responses
                        })
                        conv_history += "<%s>: " % role + dialog["conversation"][idx] + " "

    with open(save_fp, 'w', encoding='utf-8') as f:
        for sample in data_samples:
            line = json.dumps(sample, ensure_ascii=False)
            f.write(line + '\n')
    print("Total samples: {}\tEach sample with positive(s):negatives {}:{}.".format(len(data_samples), 1, num_negative))
    print("Saved to {}".format(save_fp))


def sample_responses_topdial(current_id, current_target, current_turn, dialog_list, num_negative):
    cand_dialogs = []
    for cand_dialog in dialog_list:
        if cand_dialog["target"][0] == current_target[0] and \
            cand_dialog["target"][1] != current_target[1] and \
            cand_dialog["id"] != current_id:
            # sampling hard negatives from responses with the same target action but different target topics
            cand_dialogs.append(cand_dialog)
    if len(cand_dialogs) == 0:
        print("current_id: {}, current_target: {}".format(current_id, current_target))
    assert len(cand_dialogs) > 0
    
    candidates = []
    while len(candidates) < num_negative:
        cand_dialog = random.choice(cand_dialogs)
        cand_convs = cand_dialog["conversation"]
        if len(cand_convs) >= current_turn + 1:
            if "system" in cand_convs[current_turn].keys():
                cand_response = cand_convs[current_turn]["system"]
            else:
                cand_response = cand_convs[current_turn-1]["system"]
            if cand_response not in candidates:
                candidates.append(cand_response)

    assert len(candidates) == num_negative
    return candidates

def parse_dataset_topdial(data_dir, data_mode, max_history=4, num_negative=1):
    assert data_mode in ['train', 'eval']
    if data_mode == 'train':
        print("Sampling positive and negative samples for training...")
        # since we train the classifier to identify the consistency between the context and a dialogue model's generated response, 
        # we use all the data (except validation set) for training
        data_splits = ['train', 'test_seen', 'test_unseen']
        save_fp = os.path.join(data_dir, 'detector_train.jsonl')
    else:
        print("Sampling positive and negative samples for evaluation...")
        data_splits = ['dev']
        save_fp = os.path.join(data_dir, 'detector_eval.jsonl')
        
    data_samples = []
    for split in data_splits:
        data_fp = os.path.join(data_dir, 'dialogue_{}.jsonl'.format(split))
        print("Loading data from {}".format(data_fp))
        all_dialogs = []
        with open(data_fp, 'r', encoding='utf-8') as f:
            for line in f:
                dialog = json.loads(line.strip())
                all_dialogs.append(dialog)
        
        for dialog in tqdm(all_dialogs):
            user_profile = "[user profile]: "
            for k, v in dialog["user_profile"].items():
                user_profile += "%s: %s\t" % (k, v)
            target = "[system target]: <%s, %s> " % (dialog["target"][0], dialog["target"][1])
            
            # parse a multi-round dialog
            all_roles, all_convs = [], []
            is_system_first = False
            if "system" in dialog["conversation"][0].keys():
                is_system_first = True
                all_roles.append("user")
                all_convs.append("<none>")
            for idx, conv in enumerate(dialog["conversation"]):
                for k, v in conv.items():
                    all_roles.append(k)
                    all_convs.append(v)
            if max_history > 0:
                # only use the last max_history utterances
                for idx, role in enumerate(all_roles):
                    if role == "system":
                        char_list = all_roles[max(0, idx - max_history): idx]
                        conv_list = all_convs[max(0, idx - max_history): idx]
                        conv_history = ""
                        for char, conv in zip(char_list, conv_list):
                            conv_history += "<%s>: " % char + conv + " "
                        context = [user_profile, target, conv_history]
                        
                        pos_response = all_convs[idx]
                        current_turn = idx - 1 if is_system_first else idx
                        neg_responses = sample_responses_topdial(dialog["id"], dialog["target"], current_turn, all_dialogs, num_negative)
                        
                        data_samples.append({
                            'context': context,
                            'pos_response': pos_response,
                            'neg_responses': neg_responses
                        })
            else:
                # use all historical utterances
                conv_history = ""
                for idx, role in enumerate(all_roles):
                    if role == "user":
                        conv_history += "<user>: %s " % all_convs[idx]
                    else:
                        context = [user_profile, target, conv_history]
                        pos_response = all_convs[idx]
                        neg_responses = sample_responses_topdial(dialog["id"], dialog["target"], all_dialogs, num_negative)
                        data_samples.append({
                            'context': context,
                            'pos_response': pos_response,
                            'neg_responses': neg_responses
                        })
                        conv_history += "<system>: %s " % all_convs[idx]

    with open(save_fp, 'w', encoding='utf-8') as f:
        for sample in data_samples:
            line = json.dumps(sample, ensure_ascii=False)
            f.write(line + '\n')
    print("Total samples: {}\tEach sample with positive(s):negatives {}:{}.".format(len(data_samples), 1, num_negative))
    print("Saved to {}".format(save_fp))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data/light', type=str)
    parser.add_argument("--max_history", default=4, type=int, help="Max number of history utterances to use, 0 denotes all to use.")
    parser.add_argument("--num_negative_train", default=10, type=int, help="Number of negative responses for each context during training.")
    parser.add_argument("--num_negative_eval", default=10, type=int, help="Number of negative responses for each context during evaluation.")
    parser.add_argument("--random_seed", default=42, type=int, help="Random seed for data sampling.")
    args = parser.parse_args()
    print(args)
    random.seed(args.random_seed)

    if "light" in args.data_dir.lower():
        parse_dataset_light(args.data_dir, 'train', max_history=args.max_history, num_negative=args.num_negative_train)
        parse_dataset_light(args.data_dir, 'eval', max_history=args.max_history, num_negative=args.num_negative_eval)
    elif "topdial" in args.data_dir.lower():
        parse_dataset_topdial(args.data_dir, 'train', max_history=args.max_history, num_negative=args.num_negative_train)
        parse_dataset_topdial(args.data_dir, 'eval', max_history=args.max_history, num_negative=args.num_negative_eval)
    else:
        raise ValueError("Unknown dataset: {}".format(args.data_dir))
