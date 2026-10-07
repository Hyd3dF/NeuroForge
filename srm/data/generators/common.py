"""Shared helpers for deterministic synthetic generators (02 §14)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from srm.data.io import Dataset
from srm.data.sef import Extraction, RecordBase, SourceRecord

_ONSETS = ("b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z", "br", "dr", "gl", "kr", "pl", "st", "tr", "th", "sh")
_VOWELS = ("a", "e", "i", "o", "u", "ai", "ea", "io", "ou")
_CODAS = ("", "", "", "n", "r", "s", "l", "m", "x", "th")


class NameFactory:
    """Pronounceable, unique, deterministic names."""

    def __init__(self, rng: np.random.Generator) -> None:
        self.rng = rng
        self.used: set[str] = set()

    def make(self, syllables: tuple[int, int] = (2, 3)) -> str:
        for _ in range(10_000):
            n = int(self.rng.integers(syllables[0], syllables[1] + 1))
            parts = []
            for _s in range(n):
                parts.append(
                    _ONSETS[self.rng.integers(len(_ONSETS))]
                    + _VOWELS[self.rng.integers(len(_VOWELS))]
                    + _CODAS[self.rng.integers(len(_CODAS))]
                )
            name = "".join(parts).capitalize()
            if name not in self.used:
                self.used.add(name)
                return name
        raise RuntimeError("name space exhausted")


@dataclass
class GenContext:
    dataset_id: str
    seed: int
    generator: str
    version: str
    rng: np.random.Generator = field(init=False)
    names: NameFactory = field(init=False)
    _ordinals: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.names = NameFactory(self.rng)
        self.dataset = Dataset(
            dataset_id=self.dataset_id, generator=self.generator, generator_version=self.version, seed=self.seed
        )

    def rid(self, kind: str) -> str:
        n = self._ordinals.get(kind, 0)
        self._ordinals[kind] = n + 1
        return f"{self.dataset_id}:{kind}:{n}"

    def extraction(self) -> Extraction:
        return Extraction(method="generator", method_version=f"{self.generator}-{self.version}", confidence=1.0)

    def add(self, record: RecordBase) -> RecordBase:
        return self.dataset.add(record)

    def source(self, name: str, source_type: str, trust_class: str | None = None, root: str | None = None,
               category: str = "synthetic") -> str:
        sid = f"{self.dataset_id}:source:{name}"
        self.add(
            SourceRecord(
                record_id=sid, source_type=source_type, trust_class=trust_class or source_type,
                title=name, root_source_id=root, data_category=category, extraction=self.extraction(),
            )
        )
        return sid

    def choice(self, seq: list[Any] | tuple[Any, ...]) -> Any:
        return seq[int(self.rng.integers(len(seq)))]
