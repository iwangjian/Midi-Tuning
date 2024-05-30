import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class DetectPolyEncoder(nn.Module):
    def __init__(self, config, pretrained_model_dir, poly_m=16, num_labels=2):
        super().__init__()
        self.bert_context = BertModel.from_pretrained(pretrained_model_dir)
        self.bert_response = BertModel.from_pretrained(pretrained_model_dir)
        self.poly_m = poly_m
        self.poly_code_embeddings = nn.Embedding(self.poly_m, config.hidden_size)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.detector_head = nn.Linear(config.hidden_size, self.num_labels)
        # https://github.com/facebookresearch/ParlAI/blob/master/parlai/agents/transformer/polyencoder.py#L355
        torch.nn.init.normal_(self.poly_code_embeddings.weight, config.hidden_size ** -0.5)
    
    def resize_token_embeddings(self, new_num_tokens=None):
        self.bert_context.resize_token_embeddings(new_num_tokens, pad_to_multiple_of=None)
        self.bert_response.resize_token_embeddings(new_num_tokens, pad_to_multiple_of=None)

    def dot_attention(self, q, k, v):
        # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
        # k=v: [bs, length, dim] or [bs, poly_m, dim]
        attn_weights = torch.matmul(q, k.transpose(2, 1)) # [bs, poly_m, length]
        attn_weights = F.softmax(attn_weights, -1)
        output = torch.matmul(attn_weights, v) # [bs, poly_m, dim]
        return output

    def forward(self, context_input_ids, context_input_masks, responses_input_ids, responses_input_masks, labels=None):
        batch_size, res_cnt, seq_length = responses_input_ids.shape

        # context encoder
        ctx_out = self.bert_context(context_input_ids, context_input_masks)[0]  # [bs, length, dim]
        poly_code_ids = torch.arange(self.poly_m, dtype=torch.long).to(context_input_ids.device)
        poly_code_ids = poly_code_ids.unsqueeze(0).expand(batch_size, self.poly_m)
        poly_codes = self.poly_code_embeddings(poly_code_ids) # [bs, poly_m, dim]
        embs = self.dot_attention(poly_codes, ctx_out, ctx_out) # [bs, poly_m, dim]

        # responses encoder
        responses_input_ids = responses_input_ids.view(-1, seq_length)   # [bs*res_cnt, seq_length]
        responses_input_masks = responses_input_masks.view(-1, seq_length)  # [bs*res_cnt, seq_length]
        cand_emb = self.bert_response(responses_input_ids, responses_input_masks)[0][:,0,:] # [bs*res_cnt, dim]
        cand_emb = cand_emb.view(batch_size, res_cnt, -1) # [bs, res_cnt, dim]

        # merge
        ctx_emb = self.dot_attention(cand_emb, embs, embs) # [bs*res_cnt, dim]
        ctx_emb = ctx_emb.reshape(-1, ctx_emb.shape[-1])  # [bs*res_cnt, dim]
        
        dropout_output = self.dropout(ctx_emb)
        logits  = self.detector_head(dropout_output)  # [bs*res_cnt, num_labels]
 
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
            return (logits, loss)
        else:
            return (logits,)


class DetectEncoder(nn.Module):
    def __init__(self, config, pretrained_model_dir, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_dir)
        self.num_labels = num_labels
        self.dropout = nn.Dropout(0.2)
        self.detector_head = nn.Linear(config.hidden_size, self.num_labels)
    
    def resize_token_embeddings(self, new_num_tokens=None):
        self.bert.resize_token_embeddings(new_num_tokens, pad_to_multiple_of=None)

    def forward(self, text_input_ids, text_input_masks, labels=None):
        encoded_states, pooled_output = self.bert(text_input_ids, text_input_masks, return_dict=False)
        
        # use pooled_output or cls_states?
        cls_states = encoded_states[:,0,:]
        dropout_output = self.dropout(cls_states)
        #dropout_output = self.dropout(pooled_output)   
        
        logits  = self.detector_head(dropout_output)  # [bs, num_labels]

        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits.view(-1, self.num_labels), labels.view(-1))
            return (logits, loss)
        else:
            return (logits,)