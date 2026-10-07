"""Epistemic taint: information-flow typing of predictions (06 §2.2–2.3).

T1 claims from predictive producers are tainted.  T2 taint(c) = OR(taint(premises)).
T3 conjectures are tainted.  T4 taint is removed only by a promotion event.
T5 the Mouth renders tainted claims only as predictions (enforced in the Mouth).
"""

from __future__ import annotations

import enum

from srm.config.build_config import BuildConfig
from srm.core.workspace import Claim, Workspace
from srm.interface.messages import EpistemicState


class Promotion(str, enum.Enum):
    OBSERVATION_MATCH = "observation_match"
    TEST_PASS = "test_pass"
    UNTAINTED_DEDUCTION = "untainted_deduction"
    CORROBORATION = "corroboration"
    VERIFIER = "verifier"  # disallowed unless configured for low stakes (D-011)


PROMOTED_STATE = {
    Promotion.OBSERVATION_MATCH: EpistemicState.OBSERVED,
    Promotion.TEST_PASS: EpistemicState.TESTED,
    Promotion.UNTAINTED_DEDUCTION: EpistemicState.DERIVED,
    Promotion.CORROBORATION: EpistemicState.REMEMBERED,
    Promotion.VERIFIER: EpistemicState.DERIVED,
}


def derive_taint(ws: Workspace, premises: tuple[str, ...], producer_predictive: bool) -> bool:
    return producer_predictive or any(ws.claims[p].tainted for p in premises if p in ws.claims)


class TaintViolation(RuntimeError):
    """Raised when a promotion would violate the rules of 06 §2.3."""


def promote(ws: Workspace, claim_id: str, kind: Promotion, cfg: BuildConfig, stakes: float = 0.5,
            corroborating_belief: float | None = None) -> bool:
    """Apply a promotion event; returns True if taint was removed."""
    c = ws.claims[claim_id]
    if kind == Promotion.UNTAINTED_DEDUCTION:
        if not c.producer_exact or c.producer_predictive:
            raise TaintViolation("untainted deduction requires an exact, non-predictive producer")
        if any(ws.claims[p].tainted for p in c.premises if p in ws.claims):
            raise TaintViolation("untainted deduction requires untainted premises")
    elif kind == Promotion.CORROBORATION:
        if corroborating_belief is None or corroborating_belief < cfg.epistemics.theta_commit:
            raise TaintViolation("corroboration requires an untainted record with b ≥ θ_commit")
    elif kind == Promotion.VERIFIER:
        if not (cfg.epistemics.allow_verifier_promotion_low_stakes and stakes < 0.3):
            raise TaintViolation("verifier entailment does not remove taint (D-011)")
    was = c.tainted
    c.tainted = False
    c.state = PROMOTED_STATE[kind]
    c.promoted_by.append(kind.value)
    ws.log("promote", claim=claim_id, kind=kind.value)
    _cascade(ws, claim_id, cfg)
    return was


def _cascade(ws: Workspace, claim_id: str, cfg: BuildConfig) -> None:
    """Dependents produced by exact deduction become untainted once all their premises are (T2)."""
    for dep in sorted(ws.dependents.get(claim_id, ())):
        d = ws.claims.get(dep)
        if d is None or not d.tainted or not d.producer_exact or d.producer_predictive or d.is_conjecture:
            continue
        if all(not ws.claims[p].tainted for p in d.premises if p in ws.claims):
            promote(ws, dep, Promotion.UNTAINTED_DEDUCTION, cfg)


def check_invariant(claim: Claim, rendered_as_knowledge: bool) -> None:
    """T5 / 12 §4.6 taint-leak invariant: knowledge-class output must be untainted."""
    if rendered_as_knowledge and claim.tainted:
        raise TaintViolation(f"taint leak: tainted claim {claim.claim_id} rendered as knowledge")
