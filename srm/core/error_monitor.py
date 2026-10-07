"""Error Monitor (05 §8): memory-consistency checks, contradiction scans and the verification
dispatch.  The learned, information-bottlenecked verifier network is added in M9 and plugs
in through :meth:`ErrorMonitor.set_verifier`."""

from __future__ import annotations

from typing import Any, Callable

from srm.core import tms
from srm.core.workspace import Claim, Workspace
from srm.interface.messages import ClaimStatus
from srm.interface.values import Value

VerifierFn = Callable[[Claim, list[Claim]], tuple[float, float, float]]


class ErrorMonitor:
    def __init__(self, config: Any) -> None:
        self.cfg = config
        self.verifier: VerifierFn | None = None
        self.events: list[dict[str, Any]] = []

    def set_verifier(self, fn: VerifierFn) -> None:
        self.verifier = fn

    def memory_consistency(self, ctx: Any, claim: Claim) -> int:
        """Evidence from stored engrams with the same (subject, functional relation) (05 §8.2)."""
        if not isinstance(claim.object, Value) or claim.record_id is not None:
            return 0
        if not ctx.interface.relations.is_functional(claim.relation):
            return 0
        n = 0
        for rid in ctx.memory.retrieve_pattern(claim.subject, claim.relation, qualifiers=claim.qualifiers):
            p = ctx.memory.record(rid).payload
            b, _, _ = ctx.memory.opinion(rid)
            if p.object.matches(claim.object):
                claim.ledger.add(f"memory:{rid}", self.cfg.epistemics.kappa * b, 0.0)
            else:
                claim.ledger.add(f"memory:{rid}", 0.0, self.cfg.epistemics.kappa * b)
            n += 1
        if n:
            tms.check_retraction(ctx.ws, claim.claim_id)
        return n

    def contradiction_scan(self, ws: Workspace, functional: Callable[[str], bool]) -> list[tuple[str, str]]:
        """Derived/predicted claims sharing (subject, functional relation) with different objects."""
        found = []
        claims = [c for c in ws.active() if c.record_id is None and not c.relation.startswith("spec:")]
        for i, a in enumerate(claims):
            for b in claims[i + 1 :]:
                if a.subject == b.subject and a.relation == b.relation and functional(a.relation) \
                        and isinstance(a.object, Value) and isinstance(b.object, Value) \
                        and not a.object.matches(b.object) and a.channel_mask & b.channel_mask:
                    out = tms.handle_contradiction(ws, a.claim_id, b.claim_id)
                    found.append((a.claim_id, b.claim_id))
                    self.events.append({"contradiction": (a.claim_id, b.claim_id), "outcome": out.kind})
        return found

    def verify(self, ctx: Any, claim: Claim) -> None:
        """Learned verifier evidence (bottlenecked inputs); a no-op until M9 installs the network."""
        if self.verifier is None or claim.status == ClaimStatus.RETRACTED:
            return
        premises = [ctx.ws.claims[p] for p in claim.premises if p in ctx.ws.claims]
        p_entail, p_contra, _ = self.verifier(claim, premises)
        claim.ledger.add(f"verifier:{claim.claim_id}", self.cfg.epistemics.kappa_v * p_entail,
                         self.cfg.epistemics.kappa_v * p_contra)
        ctx.accounting.verifier_calls += 1
        tms.check_retraction(ctx.ws, claim.claim_id)
