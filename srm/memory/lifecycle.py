"""Lifecycle transitions (04 §9).  CONSOLIDATED/STABLE are set by sleep (08 §5)."""

from __future__ import annotations

from dataclasses import dataclass

from srm.config.build_config import BuildConfig
from srm.interface.messages import Lifecycle


@dataclass
class LifecycleInputs:
    lifecycle: Lifecycle
    n_independent_roots: int
    belief: float
    disbelief: float
    e_plus: float
    e_minus: float
    use_count: int
    verified: bool = False
    unresolved_contradiction: bool = False
    contradiction_resolved: bool = False


def next_lifecycle(x: LifecycleInputs, cfg: BuildConfig) -> Lifecycle:
    e, m = cfg.epistemics, cfg.memory
    lc = x.lifecycle
    if lc in (Lifecycle.DEPRECATED, Lifecycle.DECAYED):
        return lc
    if x.disbelief >= e.theta_retract and x.e_minus > x.e_plus:
        return Lifecycle.DEPRECATED
    if x.unresolved_contradiction or (x.e_plus >= e.theta_conflict_mass and x.e_minus >= e.theta_conflict_mass):
        return Lifecycle.CONTESTED
    if lc == Lifecycle.CONTESTED:
        if not x.contradiction_resolved:
            return lc
        lc = Lifecycle.NEW
    if lc == Lifecycle.NEW and (x.n_independent_roots >= 2 or (x.verified and x.belief >= e.theta_commit)):
        lc = Lifecycle.CORROBORATED
    if lc == Lifecycle.CORROBORATED and x.use_count >= m.n_use:
        lc = Lifecycle.USED
    return lc


def plasticity_for(lifecycle: Lifecycle, cfg: BuildConfig) -> float:
    return cfg.memory.plasticity_by_stage[lifecycle.value]
