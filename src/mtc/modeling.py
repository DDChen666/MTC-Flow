from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


def mask_mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 1.5, weight: Optional[torch.Tensor] = None, reduction: str = "mean") -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(weight=weight, reduction="none")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = self.cross_entropy(logits, target)
        with torch.no_grad():
            pt = torch.softmax(logits, dim=-1).gather(1, target.view(-1, 1)).squeeze(1).clamp(min=1e-6, max=1.0)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class MultiTaskClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        num_primary: int,
        num_secondary: int,
        primary_weight: Optional[torch.Tensor] = None,
        secondary_weight: Optional[torch.Tensor] = None,
        use_focal: bool = False,
        focal_gamma: float = 1.5,
        dropout: float = 0.1,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)
        self._supports_token_type_ids = "token_type_ids" in self.encoder.forward.__code__.co_varnames
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(f"Model {model_name} does not expose hidden_size; ensure a transformer encoder is used")
        self.dropout = nn.Dropout(dropout)
        self.primary_head = nn.Linear(hidden_size, num_primary)
        self.secondary_head = nn.Linear(hidden_size, num_secondary)

        if use_focal:
            self.primary_loss = FocalLoss(gamma=focal_gamma, weight=primary_weight)
            self.secondary_loss = FocalLoss(gamma=focal_gamma, weight=secondary_weight)
        else:
            self.primary_loss = nn.CrossEntropyLoss(weight=primary_weight) if primary_weight is not None else nn.CrossEntropyLoss()
            self.secondary_loss = nn.CrossEntropyLoss(weight=secondary_weight) if secondary_weight is not None else nn.CrossEntropyLoss()

        self.alpha = alpha
        self.beta = beta

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        labels_secondary: Optional[torch.Tensor] = None,
    ) -> dict:
        encoder_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if token_type_ids is not None and self._supports_token_type_ids:
            encoder_kwargs["token_type_ids"] = token_type_ids
        outputs = self.encoder(**encoder_kwargs)
        pooled_output = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else mask_mean_pooling(outputs.last_hidden_state, attention_mask)
        pooled_output = self.dropout(pooled_output)
        primary_logits = self.primary_head(pooled_output)
        secondary_logits = self.secondary_head(pooled_output)

        loss = None
        if labels is not None and labels_secondary is not None:
            primary_loss = self.primary_loss(primary_logits, labels)
            secondary_loss = self.secondary_loss(secondary_logits, labels_secondary)
            loss = self.alpha * primary_loss + self.beta * secondary_loss

        return {
            "loss": loss,
            "logits_primary": primary_logits,
            "logits_secondary": secondary_logits,
        }
