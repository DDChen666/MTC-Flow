from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

LOGGER = logging.getLogger(__name__)


def _deduplicate(labels: Iterable[str]) -> List[str]:
    seen = set()
    unique: List[str] = []
    for label in labels:
        if label not in seen:
            unique.append(label)
            seen.add(label)
        else:
            LOGGER.warning("Duplicate label detected and ignored: %s", label)
    return unique


@dataclass
class LabelMapping:
    labels: List[str]
    label2id: Dict[str, int] = field(init=False)
    id2label: Dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.labels = _deduplicate(self.labels)
        if not self.labels:
            raise ValueError("Label list cannot be empty")
        self.label2id = {label: idx for idx, label in enumerate(self.labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        LOGGER.debug("Initialized LabelMapping with labels: %s", self.labels)

    def to_dict(self) -> Dict[str, List[str]]:
        return {"labels": self.labels}


@dataclass
class TaskLabelSchema:
    primary: LabelMapping
    secondary: LabelMapping

    @classmethod
    def from_lists(cls, primary: List[str], secondary: List[str]) -> "TaskLabelSchema":
        return cls(primary=LabelMapping(primary), secondary=LabelMapping(secondary))

    def to_serializable(self) -> Dict[str, Dict[str, List[str]]]:
        return {
            "PRIMARY_LABELS": self.primary.labels,
            "SECONDARY_LABELS": self.secondary.labels,
            "PRIMARY2ID": self.primary.label2id,
            "SECONDARY2ID": self.secondary.label2id,
        }
