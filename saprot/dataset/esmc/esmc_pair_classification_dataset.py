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

    def __getitem__(self, index):
        entry = json.loads(self._get(index))
        seq_1, conv_1 = normalize_to_amino_acids(entry['seq_1'])
        seq_2, conv_2 = normalize_to_amino_acids(entry['seq_2'])

        if (conv_1 or conv_2) and not self._sa_to_aa_warned:
            print("[ESMCPairClassificationDataset] Detected SA sequences. "
                  "Converted them to plain amino-acid sequences for ESMC.")
            self._sa_to_aa_warned = True

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


