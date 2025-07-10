import json
import torch
from ..data_interface import register_dataset
from transformers import T5Tokenizer
from ..lmdb_dataset import *

@register_dataset
class ProtT5TokenClassificationDataset(LMDBDataset):
    def __init__(self,
                 tokenizer: str,
                 max_length: int = 512,
                 **kwargs):
        """
        Args:
            tokenizer: Path to tokenizer
            max_length: Max length of sequence
            **kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer)
        self.max_length = max_length
        self.ignore_index = -100

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq = entry['seq'][::2] 
        label = entry["label"]
        
        return seq, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, raw_labels = tuple(zip(*batch))
        
        # Prepare sequences for ProtT5 tokenizer (add spaces)
        seqs_with_spaces = tuple(" ".join(seq) for seq in seqs)
        
        inputs_encoding = self.tokenizer.batch_encode_plus(
            seqs_with_spaces, 
            padding='max_length', 
            truncation=True, 
            return_tensors='pt', 
            max_length=self.max_length
        )

        target_length = inputs_encoding['input_ids'].shape[1]

        processed_labels = []
        for label_list in raw_labels:
            truncated_label_list = label_list[:target_length - 1]

            padding_size = target_length - len(truncated_label_list)
            padded_label_list = truncated_label_list + [self.ignore_index] * padding_size
            
            processed_labels.append(padded_label_list)

        labels_tensor = torch.tensor(processed_labels, dtype=torch.long)
        
        return {"inputs": inputs_encoding}, {"labels": labels_tensor}
