"""Support: best-supported, not most probable (06 §4).

``Support(h) = llr(h) − λ_c·C(h) + λ_v·V(h) − λ_mdl·DL(h) + min(π_cap, λ_p·log p_gen(h))``

The evidence term is the calibrated log-odds of the hypothesis's pooled evidence
ledger, in which events sharing a root source were already combined by max (the
grouping in 06 §4).  The generator probability is only a capped proposal prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from srm.config.build_config import BuildConfig
from srm.core.evidence import EvidenceLedger, logit
from srm.interface.messages import EpistemicState
from srm.interface.values import Value


@dataclass
class HypothesisEval:
    hyp_id: str
    binding: Value | None
    claim_ids: list[str] = field(default_factory=list)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    tainted: bool = False
    grounded: bool = False
    in_envelope: bool = True
    knowledge_state: EpistemicState | None = None
    n_unresolved_contradictions: int = 0
    n_verified: int = 0
    description_length: float = 0.0
    log_p_gen: float | None = None
    is_residual: bool = False
    unexplained_mass: float = 0.0
    support: float = 0.0
    b: float = 0.0
    d: float = 0.0
    u: float = 1.0


def evaluate(h: HypothesisEval, cfg: BuildConfig) -> HypothesisEval:
    e, p = cfg.epistemics, cfg.prediction
    if h.is_residual:
        h.support = p.lambda_res * h.unexplained_mass - p.lambda_res0
        h.b, h.d, h.u = 0.0, 0.0, 1.0
        return h
    o = h.ledger.opinion(e.W)
    h.b, h.d, h.u = o.b, o.d, o.u
    s = logit(o.b, e.llr_eps) - e.lambda_c * h.n_unresolved_contradictions + e.lambda_v * h.n_verified
    s -= e.lambda_mdl * h.description_length
    if h.log_p_gen is not None:
        s += min(e.pi_cap, e.lambda_p * h.log_p_gen)
    h.support = s
    return h


def presentation_weights(hyps: list[HypothesisEval]) -> list[float]:
    """softmax(Support) for display only — never used as belief (06 §4)."""
    if not hyps:
        return []
    m = max(h.support for h in hyps)
    w = [math.exp(h.support - m) for h in hyps]
    z = sum(w)
    return [x / z for x in w]


def rank(hyps: list[HypothesisEval], cfg: BuildConfig) -> list[HypothesisEval]:
    for h in hyps:
        evaluate(h, cfg)
    return sorted(hyps, key=lambda h: (-h.support, h.hyp_id))
