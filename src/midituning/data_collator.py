from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers.trainer_pt_utils import LabelSmoother
import torch

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

@dataclass
class DataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizer
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        feature_keys = features[0].keys()
        batch = {}
        for key in feature_keys:
            feature = [f[key] for f in features]
            batch_size = len(feature)
            if key in ["context_ids", "target_ids", "attention_mask"]:
                max_turn_length = max([len(x) for x in feature])
                seq_lengths = []
                for i, x in enumerate(feature):
                    seq_lengths.append(max([len(y) for y in x]))
                max_seq_length = max(seq_lengths)

                if key == "attention_mask":
                    batch_feature = torch.zeros(
                        (batch_size, max_turn_length, max_seq_length), 
                        dtype=torch.bool,
                    )
                    for i, feature in enumerate(feature):
                        for j, x in enumerate(feature):
                            if self.tokenizer.padding_side == "left":
                                batch_feature[i, j, -len(x):] = torch.tensor(x, dtype=torch.bool)
                            else:
                                batch_feature[i, j, :len(x)] = torch.tensor(x, dtype=torch.bool)
                elif key == "target_ids":
                    batch_feature = torch.full(
                        (batch_size, max_turn_length, max_seq_length), IGNORE_TOKEN_ID, 
                        dtype=torch.long,
                    )
                    for i, feature in enumerate(feature):
                        for j, x in enumerate(feature):
                            if self.tokenizer.padding_side == "left":
                                batch_feature[i, j, -len(x):] = torch.tensor(x, dtype=torch.long)
                            else:
                                batch_feature[i, j, :len(x)] = torch.tensor(x, dtype=torch.long)
                else:
                    batch_feature = torch.full(
                        (batch_size, max_turn_length, max_seq_length), self.tokenizer.pad_token_id, 
                        dtype=torch.long,
                    )
                    for i, feature in enumerate(feature):
                        for j, x in enumerate(feature):
                            if self.tokenizer.padding_side == "left":
                                batch_feature[i, j, -len(x):] = torch.tensor(x, dtype=torch.long)
                            else:
                                batch_feature[i, j, :len(x)] = torch.tensor(x, dtype=torch.long)
            else:
                max_seq_length = max([len(x) for x in feature])
                batch_feature = torch.full(
                    (batch_size, max_seq_length), self.tokenizer.pad_token_id, 
                    dtype=torch.long,
                )
                for i, x in enumerate(feature):
                    if self.tokenizer.padding_side == "left":
                        batch_feature[i, -len(x):] = torch.tensor(x, dtype=torch.long)
                    else:
                        batch_feature[i, :len(x)] = torch.tensor(x, dtype=torch.long)
            batch[key] = batch_feature
        
        batch = BatchEncoding(data=batch, tensor_type=self.return_tensors)

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        
        return batch