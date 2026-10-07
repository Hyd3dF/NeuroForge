"""Truth maintenance: derived belief, retraction cascades, contradictions, nogoods (05 §2.3, §3)."""

from __future__ import annotations

from dataclasses import dataclass

from srm.core.evidence import inherited_evidence, t_norm
from srm.core.workspace import Claim, Workspace
from srm.interface.messages import ClaimStatus, EpistemicState


def derived_belief(ws: Workspace, claim: Claim) -> float:
    """``b_inh = r_n · T(b(p_1), …, b(p_k))`` (05 §2.3)."""
    if not claim.premises:
        return 0.0
    beliefs = [ws.belief(p) if p in ws.claims else 0.0 for p in claim.premises]
    return claim.producer_reliability * t_norm(beliefs, ws.cfg.epistemics.t_norm)


def refresh(ws: Workspace, claim_id: str) -> None:
    """Recompute the inherited component of a derived claim; direct evidence is retained."""
    c = ws.claims[claim_id]
    if not c.premises:
        return
    e = ws.cfg.epistemics
    c.ledger.inherited_plus = inherited_evidence(derived_belief(ws, c), e.W, e.e_max)
    if c.status not in (ClaimStatus.RETRACTED,) and c.ledger.roots == {} and ws.belief(claim_id) < e.theta_dormant:
        if c.status != ClaimStatus.DORMANT:
            c.status = ClaimStatus.DORMANT
            ws.log("dormant", claim=claim_id)
    elif c.status == ClaimStatus.DORMANT and c.channel_mask != 0 and ws.belief(claim_id) >= e.theta_dormant:
        c.status = ClaimStatus.OPEN


def topo_dependents(ws: Workspace, claim_id: str) -> list[str]:
    """All transitive dependents in topological order."""
    order: list[str] = []
    seen: set[str] = set()

    def visit(cid: str) -> None:
        for dep in sorted(ws.dependents.get(cid, ())):
            if dep not in seen and dep in ws.claims:
                seen.add(dep)
                visit(dep)
                order.append(dep)

    visit(claim_id)
    return list(reversed(order))


def propagate(ws: Workspace, claim_id: str) -> list[str]:
    """Refresh every dependent of ``claim_id`` after its belief changed."""
    deps = topo_dependents(ws, claim_id)
    for d in deps:
        refresh(ws, d)
    return deps


def retract(ws: Workspace, claim_id: str, reason: str) -> list[str]:
    c = ws.claims[claim_id]
    if c.status == ClaimStatus.RETRACTED:
        return []
    c.status = ClaimStatus.RETRACTED
    ws.log("retract", claim=claim_id, reason=reason)
    return propagate(ws, claim_id)


def check_retraction(ws: Workspace, claim_id: str) -> bool:
    """Retract when ``d ≥ θ_retract`` (05 §3 step 1)."""
    c = ws.claims[claim_id]
    if c.status == ClaimStatus.RETRACTED:
        return False
    if c.opinion(ws.cfg.epistemics.W).d >= ws.cfg.epistemics.theta_retract:
        retract(ws, claim_id, "disbelief")
        return True
    return False


def conjectured_ancestors(ws: Workspace, claim_id: str) -> set[str]:
    out: set[str] = set()
    stack = [claim_id]
    seen: set[str] = set()
    while stack:
        cid = stack.pop()
        if cid in seen or cid not in ws.claims:
            continue
        seen.add(cid)
        c = ws.claims[cid]
        if c.is_conjecture:
            out.add(cid)
        stack.extend(c.premises)
    return out


@dataclass
class ContradictionOutcome:
    kind: str  # "different_channels" | "retracted" | "channels_killed"
    retracted: str | None = None
    killed_channels: tuple[int, ...] = ()
    nogood: frozenset[str] = frozenset()


def handle_contradiction(ws: Workspace, a: str, b: str) -> ContradictionOutcome:
    """05 §3 step 2: kill shared channels and record the assumption set as a nogood, or retract
    the weaker claim when no assumptions are involved.

    The nogood is the union of both claims' CONJECTURED ancestors (the ATMS environment in
    which the contradiction holds).
    """
    ca, cb = ws.claims[a], ws.claims[b]
    shared = ca.channel_mask & cb.channel_mask
    if shared == 0:
        return ContradictionOutcome("different_channels")
    assumptions = conjectured_ancestors(ws, a) | conjectured_ancestors(ws, b)
    if not assumptions:
        weaker = a if ws.belief(a) < ws.belief(b) or (ws.belief(a) == ws.belief(b) and a > b) else b
        retract(ws, weaker, f"contradiction with {b if weaker == a else a}")
        nogood = frozenset({ws.claims[weaker].content_key()})
        return ContradictionOutcome("retracted", retracted=weaker, nogood=nogood)
    nogood = frozenset(ws.claims[x].content_key() for x in assumptions)
    ws.add_nogood(nogood)
    killed = []
    for idx in range(ws.H):
        if shared & (1 << idx) and idx in ws.channels and ws.channels[idx].alive:
            ws.kill_channel(idx)
            killed.append(idx)
    return ContradictionOutcome("channels_killed", killed_channels=tuple(killed), nogood=nogood)


def mark_state(claim: Claim, state: EpistemicState) -> None:
    claim.state = state
