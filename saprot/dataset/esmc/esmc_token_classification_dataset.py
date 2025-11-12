import json
import torch

from ..data_interface import register_dataset
from ..lmdb_dataset import LMDBDataset
from .sa_utils import normalize_to_amino_acids

try:
    from esm.sdk.api import ESMProtein
    from esm.models.esmc import ESMC
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_dataset
class ESMCTokenClassificationDataset(LMDBDataset):
    def __init__(self,
            model_name: str = "esmc_300m",
            max_length: int = 1024,
            sa_debug: bool = True,
            **kwargs):
        """
        Args:
            model_name: name of the model
            max_length: max length of sequence
            **kwargs:
        """
        super().__init__(**kwargs)

        temp_model = ESMC.from_pretrained(model_name)
        self.tokenizer = temp_model.tokenizer
        del temp_model

        self.model_name = model_name
        self.max_length = max_length
        self._sa_to_aa_warned = False
        self._sa_debug = sa_debug
        self._prefetch_sa_warning()

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        original_seq = entry['seq']
        seq, converted = normalize_to_amino_acids(original_seq)
        if converted and not self._sa_to_aa_warned:
            self._emit_sa_warning(original_seq, seq)

        if len(seq) > self.max_length:
            seq = seq[:self.max_length]

        label_core = entry["label"][:len(seq)]
        # Add -1 to match special tokens convention if needed; we will align length in loss
        label = [-1] + label_core + [-1]
        label = torch.tensor(label, dtype=torch.long)

        return seq, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        from data.data_transform import pad_sequences

        seqs, label_ids = tuple(zip(*batch))
        label_ids = pad_sequences(label_ids, constant_value=-1)
        labels = {"labels": label_ids}

        proteins = [ESMProtein(sequence=s) for s in seqs]
        inputs = {"inputs": {"proteins": proteins}}

        return inputs, labels

    def _prefetch_sa_warning(self):
        if self._sa_to_aa_warned:
            return
        try:
            total = int(self._get("length"))
        except Exception:
            total = 0

        for idx in range(min(total, 10)):
            try:
                entry = json.loads(self._get(idx))
            except Exception:
                continue
            original_seq = entry.get("seq", "")
            converted_seq, converted = normalize_to_amino_acids(original_seq)
            if converted:
                self._emit_sa_warning(original_seq, converted_seq)
                break

    def _emit_sa_warning(self, original_seq: str, converted_seq: str):
        if not self._sa_to_aa_warned:
            if self._sa_debug:
                preview_len = 120
                original_preview = original_seq[:preview_len]
                converted_preview = converted_seq[:preview_len]
                print("[ESMCTokenClassificationDataset] SA sample detected and converted.",
                      f"Original: {original_preview}",
                      f"Converted: {converted_preview}",
                      sep="\n")
            self._sa_to_aa_warned = True


