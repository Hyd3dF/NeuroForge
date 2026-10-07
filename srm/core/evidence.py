"""Evidence algebra (05 §2).

``b = e⁺/(e⁺+e⁻+W)``, ``d = e⁻/(e⁺+e⁻+W)``, ``u = W/(e⁺+e⁻+W)``.  Evidence events
that share a root source are combined by **max**, not sum (independence rule).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

from srm.interface.messages import EvidenceEvent


@dataclass(frozen=True)
class Opinion:
    b: float
    d: float
    u: float

    @staticmethod
    def of(e_plus: float, e_minus: float, W: float) -> "Opinion":
        total = e_plus + e_minus + W
        return Opinion(e_plus / total, e_minus / total, W / total)


def belief(e_plus: float, e_minus: float, W: float) -> float:
    return e_plus / (e_plus + e_minus + W)


@dataclass
class EvidenceLedger:
    """Per-target accumulator: ``root → (max Δe⁺, max Δe⁻)`` plus an inherited component."""

    roots: dict[str, tuple[float, float]] = field(default_factory=dict)
    inherited_plus: float = 0.0

    def add(self, root: str, delta_plus: float, delta_minus: float) -> bool:
        """Apply an event; returns True if totals changed."""
        if delta_plus < 0 or delta_minus < 0:
            raise ValueError("evidence deltas must be non-negative")
        old = self.roots.get(root, (0.0, 0.0))
        new = (max(old[0], delta_plus), max(old[1], delta_minus))
        if new != old:
            self.roots[root] = new
            return True
        return False

    def add_event(self, event: EvidenceEvent) -> bool:
        return self.add(event.source_root_id, event.delta_plus, event.delta_minus)

    @property
    def e_plus(self) -> float:
        return sum(p for p, _ in self.roots.values()) + self.inherited_plus

    @property
    def e_minus(self) -> float:
        return sum(m for _, m in self.roots.values())

    def opinion(self, W: float) -> Opinion:
        return Opinion.of(self.e_plus, self.e_minus, W)

    def n_roots_supporting(self) -> int:
        return sum(1 for p, _ in self.roots.values() if p > 0)

    def merge(self, other: "EvidenceLedger") -> "EvidenceLedger":
        """Union by root with max per root (used when pooling evidence of equal hypotheses)."""
        out = EvidenceLedger(dict(self.roots), max(self.inherited_plus, other.inherited_plus))
        for root, (p, m) in other.roots.items():
            out.add(root, p, m)
        return out

    def to_dict(self) -> dict[str, object]:
        return {"roots": {k: list(v) for k, v in self.roots.items()}, "inherited_plus": self.inherited_plus}

    @staticmethod
    def from_dict(d: dict[str, object]) -> "EvidenceLedger":
        roots = {k: (float(v[0]), float(v[1])) for k, v in dict(d.get("roots", {})).items()}  # type: ignore[arg-type]
        return EvidenceLedger(roots, float(d.get("inherited_plus", 0.0)))  # type: ignore[arg-type]


def t_norm(values: Iterable[float], kind: str = "min") -> float:
    vals = list(values)
    if not vals:
        return 1.0
    if kind == "min":
        return min(vals)
    if kind == "product":
        return math.prod(vals)
    raise ValueError(f"unknown t-norm {kind}")


def inherited_evidence(b_inh: float, W: float, e_max: float) -> float:
    """``e⁺ = W · b_inh / (1 − b_inh)`` capped at ``e_max`` (05 §2.3)."""
    if b_inh <= 0.0:
        return 0.0
    if b_inh >= 1.0:
        return e_max
    return min(e_max, W * b_inh / (1.0 - b_inh))


def modality_factor(modality: str, hedge: float, factors: dict[str, float], hedge_slope: float) -> float:
    """``μ`` from 02 §10.2; hedged statements scale by ``1 − slope·hedge``."""
    base = factors.get(modality, 0.0)
    if modality == "hedged":
        return max(0.0, base * (1.0 - hedge_slope * hedge))
    return base


def initial_evidence(kappa: float, trust: float, confidence: float, mu: float) -> float:
    """``Δe⁺ = κ · t(s) · c · μ`` (02 §10.3)."""
    return kappa * trust * confidence * mu


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(1.0 - eps, max(eps, p))
    return math.log(p / (1.0 - p))
