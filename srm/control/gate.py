"""Thalamic Gate (07 §2): salience competition, deduplication and admission into the Workspace."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from srm.interface import codes as C


@dataclass
class BufferItem:
    record_id: int
    source: str  # producing process
    error: float = 0.0
    precision: float = 1.0
    relevance: float = 0.0
    novelty: float = 0.0
    trust: float = 0.5
    code: np.ndarray | None = None
    salience: float = 0.0


@dataclass
class Gate:
    weights: tuple[float, float, float, float] = (0.5, 1.5, 0.5, 0.5)
    bias: float = -1.0
    theta_dup: float = 0.75
    admitted_log: list[int] = field(default_factory=list)

    def salience(self, item: BufferItem, gain_ach: float = 1.0) -> float:
        w1, w2, w3, w4 = self.weights
        z = w1 * item.precision * abs(item.error) + w2 * item.relevance + w3 * item.novelty + w4 * item.trust + self.bias
        return gain_ach / (1.0 + math.exp(-z))

    def admit(self, items: list[BufferItem], free_slots: int, gain_ach: float = 1.0, B: int = 64) -> list[BufferItem]:
        for it in items:
            it.salience = self.salience(it, gain_ach)
        chosen: list[BufferItem] = []
        for it in sorted(items, key=lambda x: -x.salience):
            if len(chosen) >= free_slots:
                break
            if it.code is not None and any(
                c.code is not None and C.overlap(it.code, c.code) >= self.theta_dup * B for c in chosen
            ):
                continue  # lateral inhibition between near-duplicates
            chosen.append(it)
        self.admitted_log.extend(i.record_id for i in chosen)
        return chosen
