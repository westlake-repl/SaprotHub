import torch
import json
from ..data_interface import register_dataset
from ..lmdb_dataset import LMDBDataset
from .sa_utils import normalize_to_amino_acids

try:
    from esm.sdk.api import ESMProtein
    from esm.models.esmc import ESMC
except ImportError:
    raise ImportError("Please install esm package first: pip install esm")


@register_dataset
class ESMCRegressionDataset(LMDBDataset):
    def __init__(self,
                 model_name: str = "esmc_300m",
                 max_length: int = 1024,
                 min_clip: [float, float] = None,
                 mix_max_norm: [float, float] = None,
                 sa_debug: bool = True,
                 **kwargs):
        """
        Args:
            model_name: name of the model
            max_length: maximum length of the sequence
            min_clip: [given_value, clip_value]
                      Set the fitness value to a fixed value if it is less than a given value
            mix_max_norm: [min_norm, max_norm]
                      Normalize the fitness value to [0, 1] by min-max normalization
            **kwargs:
        """
        super().__init__(**kwargs)

        temp_model = ESMC.from_pretrained(model_name)
        self.tokenizer = temp_model.tokenizer
        del temp_model
        
        self.model_name = model_name
        self.max_length = max_length
        self.min_clip = min_clip
        self.mix_max_norm = mix_max_norm
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

        if self.min_clip is not None:
            given_min, clip_value = self.min_clip
            if entry["fitness"] < given_min:
                entry["fitness"] = clip_value

        if self.mix_max_norm is not None:
            min_norm, max_norm = self.mix_max_norm
            entry["fitness"] = (entry["fitness"] - min_norm) / (max_norm - min_norm)

        label = entry["fitness"]
        return seq, label

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, labels = tuple(zip(*batch))

        labels = {"labels": torch.tensor(labels, dtype=torch.float32)}

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
                print("[ESMCRegressionDataset] SA sample detected and converted.",
                      f"Original: {original_preview}",
                      f"Converted: {converted_preview}",
                      sep="\n")
            self._sa_to_aa_warned = True