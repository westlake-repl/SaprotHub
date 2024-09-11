import torch
import json
import random

from ..data_interface import register_dataset
from transformers import T5Tokenizer
from ..lmdb_dataset import *
from ..lmdb_dataset import *
from utils.others import setup_seed
import re


@register_dataset
class ProtT5RegressionDataset(LMDBDataset):
	def __init__(self,
				 tokenizer: str,
				 max_length: int = 1024,
				 min_clip: [float, float] = None,
				 mix_max_norm: [float, float] = None,
				 mask_struc_ratio: float = None,
				 plddt_threshold: float = None,
				 **kwargs):
		"""
		
		Args:
			tokenizer: ESM tokenizer

			max_length: Maximum length of the sequence

			min_clip: [given_value, clip_value]
					  Set the fitness value to a fixed value if it is less than a given value
			
			mix_max_norm: [min_norm, max_norm]
						  Normalize the fitness value to [0, 1] by min-max normalization

			mask_struc_ratio: Ratio of masked structure tokens, replace structure tokens with "#"
			
			plddt_threshold: If not None, mask structure tokens with pLDDT < threshold

			**kwargs:
		"""
		
		super().__init__(**kwargs)
		self.tokenizer = T5Tokenizer.from_pretrained(tokenizer)
		self.max_length = max_length
		self.min_clip = min_clip
		self.mix_max_norm = mix_max_norm
		self.mask_struc_ratio = mask_struc_ratio
		self.plddt_threshold = plddt_threshold
	
	def __getitem__(self, index):
		entry = json.loads(self._get(index))
		seq = entry['seq'][::2]
		seq = " ".join(seq)
	
		if self.min_clip is not None:
			given_min, clip_value = self.min_clip
			if entry['fitness'] < given_min:
				entry['fitness'] = clip_value
		
		if self.mix_max_norm is not None:
			min_norm, max_norm = self.mix_max_norm
			entry['fitness'] = (entry['fitness'] - min_norm) / (max_norm - min_norm)
				
		label = entry['fitness']

		return seq, label
	
	def __len__(self):
		return int(self._get("length"))
	
	def collate_fn(self, batch):
		seqs, labels = tuple(zip(*batch))
		labels = torch.tensor(labels)
		labels = {"labels": labels}
		
		encoder_info = self.tokenizer.batch_encode_plus(seqs, return_tensors='pt', padding=True, max_length=self.max_length, truncation=True)
		inputs = {"inputs": encoder_info}
		
		return inputs, labels