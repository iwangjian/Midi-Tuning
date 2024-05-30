import os
import json
import torch
from torch.utils.data import Dataset


class DetectionDataset(Dataset):
    def __init__(self, data_dir, detection_transform, data_split='eval', sample_cnt=None, loaded_data=None):
        self.detection_transform = detection_transform
        self.data_source = []

        if loaded_data is not None:
            self.data_source = loaded_data
        else:
            self._load_data(data_dir, data_split=data_split, sample_cnt=sample_cnt)
    
    def _load_data(self, data_dir, data_split='eval', sample_cnt=None):
        data_fp = os.path.join(data_dir, 'detector_{}.jsonl'.format(data_split))
        print("Loading data from {}".format(data_fp))
        with open(data_fp, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                sample = json.loads(line)
                context = sample['context']
                pos_response = sample['pos_response']
                neg_responses = sample['neg_responses']
                instance = {
                    'context': context,
                    'response': pos_response,
                    'label': 1
                }
                self.data_source.append(instance)
                for neg_response in neg_responses:
                    instance = {
                        'context': context,
                        'response': neg_response,
                        'label': 0
                    }
                    self.data_source.append(instance)
                if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                    break

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        instance = self.data_source[index]
        context, response, label = instance['context'], instance['response'], instance['label']
        input_ids, input_masks = self.detection_transform(context, response)
        return (input_ids, input_masks, label)
    
    def custom_collate(self, batch):
        input_ids_list_batch, input_masks_list_batch, labels_batch = [], [], []
        for sample in batch:
            input_ids_list, input_masks_list = sample[:2]
            input_ids_list_batch.append(input_ids_list)
            input_masks_list_batch.append(input_masks_list)
            labels_batch.append(sample[-1])
        long_tensors = [input_ids_list_batch, input_masks_list_batch]
        input_ids_batch, input_masks_batch = (
            torch.tensor(t, dtype=torch.long) for t in long_tensors)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)
        
        return input_ids_batch, input_masks_batch, labels_batch
    

class SelectionDataset(Dataset):
    def __init__(self, data_dir, context_transform, response_transform, concat_transform, data_split='eval', sample_cnt=None, mode='poly'):
        self.context_transform = context_transform
        self.response_transform = response_transform
        self.concat_transform = concat_transform
        self.mode = mode

        self.data_source = []

        self._load_data(data_dir, data_split=data_split, sample_cnt=sample_cnt)
    
    def _load_data(self, data_dir, data_split='eval', sample_cnt=None):
        data_fp = os.path.join(data_dir, 'detector_{}.jsonl'.format(data_split))
        print("Loading data from {}".format(data_fp))
        if data_split == 'eval':
            with open(data_fp, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    sample = json.loads(line)
                    context = sample['context']
                    pos_response = sample['pos_response']
                    neg_responses = sample['neg_responses']
                    group = {
                        'context': context,
                        'responses': [pos_response],
                        'labels': [1]
                    }
                    for neg_response in neg_responses:
                        group['responses'].append(neg_response)
                        group['labels'].append(0)
                    self.data_source.append(group)
                    if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                        break
        else:
            with open(data_fp, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    sample = json.loads(line)
                    context = sample['context']
                    pos_response = sample['pos_response']
                    neg_responses = sample['neg_responses']
                    for neg_response in neg_responses:
                        group = {
                            'context': context,
                            'responses': [pos_response, neg_response],
                            'labels': [1, 0]
                        }
                        self.data_source.append(group)
                        if sample_cnt is not None and len(self.data_source) >= sample_cnt:
                            break

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        group = self.data_source[index]
        context, responses, labels = group['context'], group['responses'], group['labels']
        if self.mode == 'cross':
            transformed_text = self.concat_transform(context, responses)
            ret = transformed_text, labels
        else:
            transformed_context = self.context_transform(context)  # [token_ids],[seg_ids],[masks]
            transformed_responses = self.response_transform(responses)  # [token_ids],[seg_ids],[masks]
            ret = transformed_context, transformed_responses, labels

        return ret

    def custom_collate(self, batch):
        if self.mode == 'cross':
            text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = [], [], []
            labels_batch = []
            for sample in batch:
                text_token_ids_list, text_input_masks_list, text_segment_ids_list = sample[0]

                text_token_ids_list_batch.append(text_token_ids_list)
                text_input_masks_list_batch.append(text_input_masks_list)
                text_segment_ids_list_batch.append(text_segment_ids_list)

                labels_batch.append(sample[1])

            long_tensors = [text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch]

            text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch = (
                torch.tensor(t, dtype=torch.long) for t in long_tensors)

            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
            return text_token_ids_list_batch, text_input_masks_list_batch, text_segment_ids_list_batch, labels_batch

        else:
            contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
            responses_token_ids_list_batch, responses_input_masks_list_batch = [], [], [], []
            labels_batch = []
            for sample in batch:
                (contexts_token_ids_list, contexts_input_masks_list), (responses_token_ids_list, responses_input_masks_list) = sample[:2]

                contexts_token_ids_list_batch.append(contexts_token_ids_list)
                contexts_input_masks_list_batch.append(contexts_input_masks_list)

                responses_token_ids_list_batch.append(responses_token_ids_list)
                responses_input_masks_list_batch.append(responses_input_masks_list)

                labels_batch.append(sample[-1])

            long_tensors = [contexts_token_ids_list_batch, contexts_input_masks_list_batch,
                                            responses_token_ids_list_batch, responses_input_masks_list_batch]

            contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
            responses_token_ids_list_batch, responses_input_masks_list_batch = (
                torch.tensor(t, dtype=torch.long) for t in long_tensors)

            labels_batch = torch.tensor(labels_batch, dtype=torch.long)
            return contexts_token_ids_list_batch, contexts_input_masks_list_batch, \
                          responses_token_ids_list_batch, responses_input_masks_list_batch, labels_batch
