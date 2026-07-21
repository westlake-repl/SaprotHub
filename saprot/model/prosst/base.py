import os
import json
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Dict, Optional

import torch
from easydict import EasyDict
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from ..abstract_model import AbstractModel
from .specs import resolve_structure_vocab_size


class ProSSTBaseModel(AbstractModel):
    def __init__(
        self,
        task: str,
        config_path: str = "AI4Protein/ProSST-2048",
        structure_vocab_size: Optional[int] = None,
        extra_config: dict = None,
        load_pretrained: bool = True,
        gradient_checkpointing: bool = False,
        lora_kwargs: dict = None,
        **kwargs,
    ):
        assert task in [
            "classification",
            "regression",
            "token_classification",
            "pair_classification",
            "pair_regression",
            "lm",
            "base",
        ]
        self.task = task
        self.config_path = config_path
        self.structure_vocab_size = resolve_structure_vocab_size(
            config_path,
            structure_vocab_size,
        )
        self.extra_config = extra_config or {}
        self.load_pretrained = load_pretrained
        self.gradient_checkpointing = gradient_checkpointing
        self.lora_kwargs = lora_kwargs

        for path_key in ["save_path", "from_checkpoint"]:
            if isinstance(kwargs.get(path_key), str):
                kwargs[path_key] = kwargs[path_key].replace("\\", "/")

        super().__init__(**kwargs)

        if self.lora_kwargs is not None:
            self.lora_kwargs = EasyDict(lora_kwargs)
            self._init_lora()

        self.valid_metrics_list = {"step": []}

    def _init_lora(self):
        from peft import LoraConfig, PeftModel, get_peft_model

        is_trainable = getattr(self.lora_kwargs, "is_trainable", False)
        config_list = getattr(self.lora_kwargs, "config_list", [])
        assert self.lora_kwargs.num_lora >= len(config_list), (
            "The number of LoRA models should be greater than or equal to the "
            "number of weight files."
        )

        for i in range(self.lora_kwargs.num_lora):
            adapter_name = f"adapter_{i}" if self.lora_kwargs.num_lora > 1 else "default"

            if i < len(config_list):
                entry = config_list[i]
                lora_config_path = (
                    entry.get("lora_config_path", entry)
                    if isinstance(entry, dict)
                    else getattr(entry, "lora_config_path", entry)
                )
                if i == 0:
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        lora_config_path,
                        adapter_name=adapter_name,
                        is_trainable=is_trainable,
                    )
                else:
                    self.model.load_adapter(
                        lora_config_path,
                        adapter_name=adapter_name,
                        is_trainable=is_trainable,
                    )
                continue

            target_modules = getattr(self.lora_kwargs, "target_modules", None)
            if target_modules is None:
                target_modules = [
                    "query",
                    "key",
                    "value",
                    "ss_proj",
                    "ss_q_proj",
                    "pos_proj",
                    "pos_q_proj",
                    "intermediate.dense",
                    "output.dense",
                ]

            lora_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                target_modules=target_modules,
                modules_to_save=["classifier"],
                inference_mode=False,
                r=getattr(self.lora_kwargs, "r", 8),
                lora_dropout=getattr(self.lora_kwargs, "lora_dropout", 0.0),
                lora_alpha=getattr(self.lora_kwargs, "lora_alpha", 16),
            )

            if i == 0:
                self.model = get_peft_model(self.model, lora_config, adapter_name=adapter_name)
            else:
                self.model.add_adapter(adapter_name, lora_config)

        if self.lora_kwargs.num_lora > 1:
            def lora_forward(func):
                def forward(*args, **kwargs):
                    logits_list = []
                    ori_shape = None
                    for i in range(self.lora_kwargs.num_lora):
                        adapter_name = f"adapter_{i}"
                        self.model.set_adapter(adapter_name)
                        logits = func(*args, **kwargs)
                        logits_list.append(logits)
                        if ori_shape is None:
                            ori_shape = logits.shape

                    logits = torch.stack(logits_list, dim=0)
                    if len(ori_shape) == 2 and ori_shape[-1] > 1:
                        logits = logits.permute(1, 0, 2)
                        preds = logits.argmax(dim=-1)
                        preds = torch.mode(preds, dim=1).values
                        dummy_logits = torch.zeros(ori_shape).to(logits)
                        for i, pred in enumerate(preds):
                            dummy_logits[i, pred] = 1.0
                    else:
                        dummy_logits = logits.mean(dim=0)

                    return dummy_logits.detach()

                return forward

            self.forward = lora_forward(self.forward)

        if self.gradient_checkpointing:
            self._disable_peft_input_require_grads()
        if hasattr(self.model, "print_trainable_parameters"):
            self.model.print_trainable_parameters()
        self.init_optimizers()

    def _disable_peft_input_require_grads(self):
        """Remove PEFT's reentrant-checkpoint compatibility hook.

        ColabProSST checkpoints encoder layers with ``use_reentrant=False``, so
        gradients reach LoRA parameters without forcing frozen embedding
        outputs to require gradients. Leaving PEFT's hook active turns the
        output into a leaf tensor and conflicts with ProSST's in-place token
        dropout mask.
        """
        get_base_model = getattr(self.model, "get_base_model", None)
        base_model = get_base_model() if callable(get_base_model) else self.model
        hook = getattr(base_model, "_require_grads_hook", None)
        if hook is None:
            return

        hook.remove()
        delattr(base_model, "_require_grads_hook")

    def initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config_path,
            trust_remote_code=True,
        )
        config = AutoConfig.from_pretrained(
            self.config_path,
            trust_remote_code=True,
        )
        for key, value in self.extra_config.items():
            setattr(config, key, value)

        if self.load_pretrained:
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.config_path,
                trust_remote_code=True,
                **self.extra_config,
            )
        else:
            self.model = AutoModelForMaskedLM.from_config(
                config,
                trust_remote_code=True,
            )

        if self.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def initialize_metrics(self, stage: str) -> Dict:
        return {}

    def _enable_gradient_checkpointing(self):
        if hasattr(self.model, "prosst") and hasattr(self.model.prosst, "encoder"):
            encoder = self.model.prosst.encoder
            self._patch_encoder_gradient_checkpointing(encoder)
            encoder.gradient_checkpointing = True
        elif hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    @staticmethod
    def _patch_encoder_gradient_checkpointing(encoder):
        if getattr(encoder, "_colabprosst_checkpointing_patched", False):
            return

        def forward(
            self,
            hidden_states,
            attention_mask,
            output_hidden_states=True,
            output_attentions=False,
            query_states=None,
            relative_pos=None,
            ss_hidden_states=None,
            return_dict=True,
        ):
            attention_mask = self.get_attention_mask(attention_mask)
            relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None
            if isinstance(hidden_states, Sequence):
                next_kv = hidden_states[0]
            else:
                next_kv = hidden_states
            rel_embeddings = self.get_rel_embedding()

            for i, layer_module in enumerate(self.layer):
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    def custom_forward(*inputs):
                        return layer_module(
                            inputs[0],
                            inputs[1],
                            query_states=inputs[2],
                            relative_pos=inputs[3],
                            rel_embeddings=inputs[4],
                            output_attentions=output_attentions,
                            ss_hidden_states=inputs[5],
                        )

                    hidden_states = torch.utils.checkpoint.checkpoint(
                        custom_forward,
                        next_kv,
                        attention_mask,
                        query_states,
                        relative_pos,
                        rel_embeddings,
                        ss_hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = layer_module(
                        next_kv,
                        attention_mask,
                        query_states=query_states,
                        relative_pos=relative_pos,
                        rel_embeddings=rel_embeddings,
                        output_attentions=output_attentions,
                        ss_hidden_states=ss_hidden_states,
                    )

                if output_attentions:
                    hidden_states, att_m = hidden_states

                if query_states is not None:
                    query_states = hidden_states
                    if isinstance(hidden_states, Sequence):
                        next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
                else:
                    next_kv = hidden_states

                if output_attentions:
                    all_attentions = all_attentions + (att_m,)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, all_hidden_states, all_attentions]
                    if v is not None
                )
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_attentions,
            )

        encoder.forward = types.MethodType(forward, encoder)
        encoder._colabprosst_checkpointing_patched = True

    def _get_classifier(self):
        if hasattr(self.model, "classifier"):
            return self.model.classifier
        if hasattr(self.model, "base_model"):
            base_model = self.model.base_model
            if hasattr(base_model, "model"):
                base_model = base_model.model
            if hasattr(base_model, "classifier"):
                return base_model.classifier
        raise AttributeError("Cannot find ProSST classifier head.")

    def _mean_pool(self, hidden_states, inputs):
        attention_mask = inputs["attention_mask"].to(hidden_states.device).bool()
        input_ids = inputs["input_ids"].to(hidden_states.device)

        residue_mask = attention_mask.clone()
        for token_id in [
            self.tokenizer.pad_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.eos_token_id,
        ]:
            if token_id is not None:
                residue_mask = residue_mask & (input_ids != token_id)

        residue_mask = residue_mask.unsqueeze(-1).to(hidden_states.dtype)
        denom = residue_mask.sum(dim=1).clamp(min=1.0)
        return (hidden_states * residue_mask).sum(dim=1) / denom

    def get_token_representations(self, inputs):
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            ss_input_ids=inputs["ss_input_ids"],
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]

    def get_pooled_representations(self, inputs):
        hidden_states = self.get_token_representations(inputs)
        return self._mean_pool(hidden_states, inputs)

    def forward(self, inputs):
        return self.get_pooled_representations(inputs)

    def loss_func(self, stage: str, outputs, labels) -> torch.Tensor:
        raise NotImplementedError

    def _checkpoint_metadata(self) -> dict:
        metadata = {
            "base_model": self.config_path,
            "structure_vocab_size": self.structure_vocab_size,
            "task": self.task,
        }
        if hasattr(self, "num_labels"):
            metadata["num_labels"] = int(self.num_labels)
        return {"colabprosst": metadata}

    def save_checkpoint(
        self,
        save_path: str,
        save_info: dict = None,
        save_weights_only: bool = True,
    ) -> None:
        checkpoint_info = self._checkpoint_metadata()
        if save_info is not None:
            checkpoint_info.update(save_info)

        if not self.lora_kwargs:
            raise RuntimeError(
                "ColabProSST downstream training requires LoRA. Configure "
                "lora_kwargs before saving an adapter."
            )

        try:
            if hasattr(self.trainer.strategy, "deepspeed_engine"):
                save_path = os.path.dirname(save_path)
        except Exception:
            pass

        self.model.save_pretrained(save_path)
        metadata = dict(checkpoint_info["colabprosst"])
        metadata["checkpoint_format"] = "peft_adapter"
        active_adapter = getattr(self.model, "active_adapter", "default")
        peft_config = getattr(self.model, "peft_config", {}).get(
            active_adapter
        )
        if peft_config is not None:
            metadata["lora"] = {
                "r": int(peft_config.r),
                "lora_alpha": int(peft_config.lora_alpha),
                "lora_dropout": float(peft_config.lora_dropout),
            }
        metadata_path = Path(save_path) / "colabprosst.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

    def load_lora_adapter(self, adapter_path: str) -> None:
        if not self.lora_kwargs:
            raise RuntimeError(
                "Cannot load a LoRA adapter into a non-PEFT ProSST model."
            )
        from peft.utils import load_peft_weights, set_peft_model_state_dict

        active_adapter = getattr(self.model, "active_adapter", "default")
        adapter_state = load_peft_weights(
            adapter_path,
            device=str(self.device),
        )
        set_peft_model_state_dict(
            self.model,
            adapter_state,
            adapter_name=active_adapter,
        )

    def output_test_metrics(self, log_dict):
        print("=" * 100)
        print("Test Result:")
        for key, value in log_dict.items():
            print(f"{key}: {value.item() if hasattr(value, 'item') else value}")
        print("=" * 100)

    def plot_valid_metrics_curve(self, log_dict):
        return None
