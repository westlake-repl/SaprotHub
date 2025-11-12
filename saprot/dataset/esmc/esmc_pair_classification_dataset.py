import torch
import json

from ..lmdb_dataset import LMDBDataset
from ..data_interface import register_dataset
from .sa_utils import normalize_to_amino_acids

try:
    from esm.sdk.api import ESMProtein
    from esm.models.esmc import ESMC
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_dataset
class ESMCPairClassificationDataset(LMDBDataset):
    def __init__(self,
                 model_name: str = "esmc_300m",
                 tokenizer: str = None,
                 max_length: int = 1024,
                 plddt_threshold: float = None,
                 sa_debug: bool = False,
                 **kwargs):
        """
        Args:
            model_name: name of the model
            max_length: max length of sequence
            plddt_threshold: if not None, mask structure tokens with pLDDT < threshold
            **kwargs:
        """
        super().__init__(**kwargs)

        model_ref = tokenizer or model_name
        temp_model = ESMC.from_pretrained(model_ref)
        self.tokenizer = temp_model.tokenizer
        del temp_model

        self.model_name = model_name
        self.tokenizer_name = model_ref
        self.max_length = max_length
        self.plddt_threshold = plddt_threshold
        self._sa_to_aa_warned = False
        self._sa_debug = sa_debug
        self._prefetch_sa_warning()

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        original_seq_1 = entry['seq_1']
        original_seq_2 = entry['seq_2']
        seq_1, conv_1 = normalize_to_amino_acids(original_seq_1)
        seq_2, conv_2 = normalize_to_amino_acids(original_seq_2)

        if (conv_1 or conv_2) and not self._sa_to_aa_warned:
            self._emit_sa_warning(original_seq_1, seq_1, original_seq_2, seq_2)

        if len(seq_1) > self.max_length:
            seq_1 = seq_1[:self.max_length]
        if len(seq_2) > self.max_length:
            seq_2 = seq_2[:self.max_length]

        return seq_1, seq_2, int(entry["label"])

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs_1, seqs_2, label_ids = tuple(zip(*batch))

        label_ids = torch.tensor(label_ids, dtype=torch.long)
        labels = {"labels": label_ids}

        proteins_1 = [ESMProtein(sequence=s) for s in seqs_1]
        proteins_2 = [ESMProtein(sequence=s) for s in seqs_2]

        inputs = {
            "inputs_1": {"proteins": proteins_1},
            "inputs_2": {"proteins": proteins_2}
        }

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
            original_seq_1 = entry.get("seq_1", "")
            original_seq_2 = entry.get("seq_2", "")
            converted_1, conv_1 = normalize_to_amino_acids(original_seq_1)
            converted_2, conv_2 = normalize_to_amino_acids(original_seq_2)
            if conv_1 or conv_2:
                self._emit_sa_warning(original_seq_1, converted_1, original_seq_2, converted_2)
                break

    def _emit_sa_warning(self, orig_1: str, conv_1: str, orig_2: str, conv_2: str):
        if not self._sa_to_aa_warned:
            if self._sa_debug:
                preview_len = 120
                orig_preview_1 = orig_1[:preview_len]
                conv_preview_1 = conv_1[:preview_len]
                orig_preview_2 = orig_2[:preview_len]
                conv_preview_2 = conv_2[:preview_len]
                print("[ESMCPairClassificationDataset] SA sample detected and converted.",
                      f"Seq1 original: {orig_preview_1}",
                      f"Seq1 converted: {conv_preview_1}",
                      f"Seq2 original: {orig_preview_2}",
                      f"Seq2 converted: {conv_preview_2}",
                      sep="\n")
            self._sa_to_aa_warned = True


