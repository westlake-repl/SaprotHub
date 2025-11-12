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
class ESMCClassificationDataset(LMDBDataset):
    def __init__(self,
                 model_name: str = "esmc_300m",
                 use_bias_feature: bool = False,
                 max_length: int = 1024,
                 preset_label: int = None,
                 **kwargs):
        """
        Args:
            model_name: name of the model
            use_bias_feature: if True, structure information will be used
            max_length: max length of sequence
            preset_label: if not None, all labels will be set to this value
            **kwargs:
        """
        super().__init__(**kwargs)

        temp_model = ESMC.from_pretrained(model_name)
        self.tokenizer = temp_model.tokenizer
        del temp_model

        self.model_name = model_name
        self.max_length = max_length
        self.use_bias_feature = use_bias_feature
        self.preset_label = preset_label
        self._sa_to_aa_warned = False

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq, converted = normalize_to_amino_acids(entry['seq'])
        if converted and not self._sa_to_aa_warned:
            print("[ESMCClassificationDataset] Detected SA sequences. "
                  "Converted them to plain amino-acid sequences for ESMC.")
            self._sa_to_aa_warned = True

        # Truncate plain sequence for ESMC
        if len(seq) > self.max_length:
            seq = seq[:self.max_length]

        coords = entry.get('coords', None) if self.use_bias_feature else None
        if coords is not None:
            coords = {k: v[:self.max_length] for k, v in coords.items()}

        label = entry["label"] if self.preset_label is None else self.preset_label

        return seq, label, coords

    def __len__(self):
        return int(self._get("length"))

    def collate_fn(self, batch):
        seqs, label_ids, coords = tuple(zip(*batch))

        label_ids = torch.tensor(label_ids, dtype=torch.long)
        labels = {"labels": label_ids}

        # Build ESMProtein objects expected by ESMC
        proteins = [ESMProtein(sequence=seq) for seq in seqs]

        # Note: training_step calls self(**inputs), so we wrap under key 'inputs'
        model_inputs = {"proteins": proteins}
        inputs = {"inputs": model_inputs}

        if self.use_bias_feature and coords[0] is not None:
            inputs["coords"] = coords

        return inputs, labels


