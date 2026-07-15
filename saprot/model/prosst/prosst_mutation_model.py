import re
from typing import Sequence

import torch

from ..model_interface import register_model
from .base import ProSSTBaseModel


MUTATION_RE = re.compile(r"^([A-Z])([0-9]+)([A-Z])$")


@register_model
class ProSSTMutationModel(ProSSTBaseModel):
    """ProSST zero-shot mutation scorer.

    The score follows the official ProSST convention:
    log P(mutant amino acid) - log P(wild-type amino acid), summed over all
    substitutions in a mutation string such as ``H87Y:V162M``.
    """

    def __init__(self, test_result_path: str = None, **kwargs):
        self.test_result_path = test_result_path
        super().__init__(task="lm", **kwargs)

    def initialize_metrics(self, stage):
        return {}

    def forward(self, inputs):
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            ss_input_ids=inputs["ss_input_ids"],
            return_dict=True,
        )
        log_probs = torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1)
        return {
            "log_probs": log_probs,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }

    @staticmethod
    def _parse_mutation(mutant: str, sequence_length: int):
        parsed = []
        for item in str(mutant).split(":"):
            item = item.strip().upper()
            match = MUTATION_RE.match(item)
            if match is None:
                raise ValueError(
                    f"Invalid mutation '{item}'. Expected format like H87Y or H87Y:V162M."
                )

            wt, pos, mt = match.groups()
            idx = int(pos) - 1
            if idx < 0 or idx >= sequence_length:
                raise ValueError(
                    f"Mutation '{item}' position is out of range for sequence length "
                    f"{sequence_length}."
                )
            parsed.append((wt, idx, mt))

        return parsed

    def _check_wt_residue(self, input_ids, wt: str, idx: int):
        vocab = self.tokenizer.get_vocab()
        wt_id = vocab.get(wt)
        if wt_id is None:
            raise ValueError(f"Mutation uses amino acid outside ProSST vocab: {wt}")

        residue_input_id = input_ids[idx + 1].item()
        if residue_input_id != wt_id:
            decoded = self.tokenizer.decode([residue_input_id]).strip()
            raise ValueError(
                f"Mutation WT amino acid mismatch at position {idx + 1}: "
                f"input sequence has '{decoded}', not '{wt}'."
            )

    def score_from_outputs(self, outputs, mutants: Sequence[str]):
        log_probs = outputs["log_probs"]
        input_ids = outputs["input_ids"].to(log_probs.device)
        attention_mask = outputs["attention_mask"].to(log_probs.device)
        vocab = self.tokenizer.get_vocab()
        if len(mutants) != log_probs.shape[0]:
            raise ValueError(
                f"Received {len(mutants)} mutation strings for batch size "
                f"{log_probs.shape[0]}."
            )
        scores = []

        for batch_idx, mutant in enumerate(mutants):
            score = log_probs.new_tensor(0.0)
            sequence_length = int(attention_mask[batch_idx].sum().item()) - 2
            for wt, idx, mt in self._parse_mutation(mutant, sequence_length):
                if mt not in vocab:
                    raise ValueError(
                        f"Mutation uses amino acid outside ProSST vocab: {wt}->{mt}"
                    )
                self._check_wt_residue(input_ids[batch_idx], wt, idx)
                score = score + (
                    log_probs[batch_idx, idx, vocab[mt]]
                    - log_probs[batch_idx, idx, vocab[wt]]
                )
            scores.append(score)

        return torch.stack(scores)

    def score_batch(self, inputs, mutants: Sequence[str]):
        return self.score_from_outputs(self.forward(inputs), mutants)

    def loss_func(self, stage, outputs, labels):
        mutants = labels["mutants"]
        scores = self.score_from_outputs(outputs, mutants)

        target = labels.get("labels", labels.get("fitness", labels.get("scores")))
        if target is None:
            loss = scores.sum() * 0.0
        else:
            target = target.to(scores)
            loss = torch.nn.functional.mse_loss(scores, target)

        if stage == "test" and self.test_result_path is not None:
            if not hasattr(self, "test_mutants"):
                self.test_mutants = []
                self.test_scores = []
                self.test_targets = []
            self.test_mutants.extend([str(mutant) for mutant in mutants])
            self.test_scores.append(scores.detach().cpu())
            if target is not None:
                self.test_targets.append(target.detach().cpu())

        return loss

    def on_test_epoch_start(self):
        super().on_test_epoch_start()
        self.test_mutants = []
        self.test_scores = []
        self.test_targets = []

    def on_test_epoch_end(self):
        if self.test_result_path is not None:
            scores = torch.cat(self.test_scores, dim=0) if self.test_scores else torch.tensor([])
            targets = torch.cat(self.test_targets, dim=0) if self.test_targets else None

            with open(self.test_result_path, "w", encoding="utf-8") as handle:
                header = "mutant,score"
                if targets is not None:
                    header += ",target"
                handle.write(header + "\n")
                for idx, (mutant, score) in enumerate(zip(self.test_mutants, scores)):
                    row = f"{mutant},{score.item()}"
                    if targets is not None:
                        row += f",{targets[idx].item()}"
                    handle.write(row + "\n")

        log_dict = self.get_log_dict("test")
        if self.test_outputs:
            log_dict["test_loss"] = torch.mean(torch.stack(self.test_outputs))
        self.output_test_metrics(log_dict)
