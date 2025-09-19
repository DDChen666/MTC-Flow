from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .labels import TaskLabelSchema


@dataclass
class DatasetFields:
    text_field: str
    primary_field: str
    secondary_field: str


class ReviewDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        fields: DatasetFields,
        label_schema: TaskLabelSchema,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fields = fields
        self.label_schema = label_schema
        self.items: List[Dict[str, Any]] = []

        text_key = fields.text_field
        primary_key = fields.primary_field
        secondary_key = fields.secondary_field

        for row in rows:
            text = str(row.get(text_key, "")).strip()
            if not text:
                continue
            primary = str(row.get(primary_key, "")).strip()
            secondary = str(row.get(secondary_key, "")).strip()
            if primary not in label_schema.primary.label2id or secondary not in label_schema.secondary.label2id:
                continue
            self.items.append({
                "text": text,
                "primary": primary,
                "secondary": secondary,
            })

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.items[index]
        encoded = self.tokenizer(
            item["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        batch = {key: torch.tensor(val) for key, val in encoded.items()}
        batch["labels"] = torch.tensor(self.label_schema.primary.label2id[item["primary"]], dtype=torch.long)
        batch["labels_secondary"] = torch.tensor(self.label_schema.secondary.label2id[item["secondary"]], dtype=torch.long)
        return batch
