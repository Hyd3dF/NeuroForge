"""Workspace (05 §1): fixed-capacity claims, referents, goals, hypothesis channels, nogoods, trace."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

from srm.config.build_config import BuildConfig
from srm.core.evidence import EvidenceLedger, Opinion
from srm.interface.messages import ClaimStatus, EpistemicState, Goal, Qualifiers, Referent
from srm.interface.values import Value


class WorkspaceFull(RuntimeError):
    pass


@dataclass
class Claim:
    """A workspace claim: message content plus epistemic bookkeeping (03 §5, 05 §1.1)."""

    claim_id: str
    subject: str
    relation: str
    object: Value | str
    qualifiers: Qualifiers = field(default_factory=Qualifiers)
    polarity: str = "positive"
    modality: str = "asserted"
    state: EpistemicState | None = None
    status: ClaimStatus = ClaimStatus.OPEN
    tainted: bool = False
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    producer_id: str = ""
    producer_reliability: float = 1.0
    producer_exact: bool = True
    producer_predictive: bool = False
    premises: tuple[str, ...] = ()
    channel_mask: int = -1  # set to all-ones by the workspace
    is_conjecture: bool = False
    relevance: float = 1.0
    record_id: int | None = None
    grounded_leaf: bool = False
    provenance: list[str] = field(default_factory=list)
    promoted_by: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def opinion(self, W: float) -> Opinion:
        return self.ledger.opinion(W)

    def content_key(self) -> str:
        obj = self.object.key() if isinstance(self.object, Value) else str(self.object)
        return f"{self.subject}|{self.relation}|{obj}|{self.polarity}"


@dataclass
class Channel:
    index: int
    assumption_claim_id: str
    alive: bool = True


class Workspace:
    def __init__(self, config: BuildConfig) -> None:
        w = config.workspace
        self.cfg = config
        self.K, self.R, self.H, self.A = w.K, w.R, w.H, w.A
        self.ALL = (1 << self.H) - 1
        self.claims: dict[str, Claim] = {}
        self.dependents: dict[str, set[str]] = {}
        self.referents: dict[str, Referent] = {}
        self.goals: dict[str, Goal] = {}
        self.channels: dict[int, Channel] = {}
        self.nogoods: set[frozenset[str]] = set()
        self.protected: set[str] = set()  # claims on a goal-support path (never evicted)
        self.trace: list[dict[str, Any]] = []
        self._ids = itertools.count()

    # --- ids ------------------------------------------------------------------------------------
    def new_id(self, prefix: str = "c") -> str:
        return f"{prefix}{next(self._ids)}"

    def log(self, event: str, **data: Any) -> None:
        self.trace.append({"event": event, **data})

    # --- claims ---------------------------------------------------------------------------------
    def add_claim(self, claim: Claim) -> Claim:
        if len(claim.premises) > self.A:
            raise ValueError(f"claim {claim.claim_id} has {len(claim.premises)} premises (> A={self.A})")
        if claim.claim_id in self.claims:
            raise ValueError(f"duplicate claim id {claim.claim_id}")
        if len(self.claims) >= self.K:
            self.evict()
        if claim.channel_mask == -1:
            mask = self.ALL
            for p in claim.premises:
                mask &= self.claims[p].channel_mask
            claim.channel_mask = mask
        self.claims[claim.claim_id] = claim
        self.dependents.setdefault(claim.claim_id, set())
        for p in claim.premises:
            self.dependents.setdefault(p, set()).add(claim.claim_id)
        self.log("add_claim", claim=claim.claim_id, producer=claim.producer_id, tainted=claim.tainted)
        return claim

    def get(self, claim_id: str) -> Claim:
        return self.claims[claim_id]

    def belief(self, claim_id: str) -> float:
        c = self.claims[claim_id]
        if c.status == ClaimStatus.RETRACTED:
            return 0.0
        return c.opinion(self.cfg.epistemics.W).b

    def active(self) -> list[Claim]:
        return [c for c in self.claims.values() if c.status not in (ClaimStatus.RETRACTED, ClaimStatus.DORMANT)]

    def evict(self) -> str:
        """Evict the lowest relevance×activity claim not on any goal-support path (05 §1.2)."""
        candidates = [
            c for c in self.claims.values()
            if c.claim_id not in self.protected and not self.dependents.get(c.claim_id)
        ]
        if not candidates:
            raise WorkspaceFull(f"workspace full (K={self.K}) and every claim is protected or has dependents")
        victim = min(candidates, key=lambda c: (c.relevance * (1.0 if c.status != ClaimStatus.DORMANT else 0.1), c.claim_id))
        self.log("evict", claim=victim.claim_id)
        del self.claims[victim.claim_id]
        for p in victim.premises:
            self.dependents.get(p, set()).discard(victim.claim_id)
        self.dependents.pop(victim.claim_id, None)
        return victim.claim_id

    # --- channels (05 §2.4) ---------------------------------------------------------------------
    def open_channel(self, assumption: Claim) -> int | None:
        used = {i for i, ch in self.channels.items() if ch.alive}
        free = [i for i in range(self.H) if i not in used and i not in self.channels]
        if not free:
            free = [i for i in range(self.H) if i not in used]
        if not free:
            return None
        idx = free[0]
        self.channels[idx] = Channel(idx, assumption.claim_id)
        assumption.channel_mask = 1 << idx
        assumption.is_conjecture = True
        assumption.state = EpistemicState.CONJECTURED
        assumption.tainted = True
        self.log("open_channel", channel=idx, assumption=assumption.claim_id)
        return idx

    def kill_channel(self, index: int) -> list[str]:
        ch = self.channels.get(index)
        if ch is None or not ch.alive:
            return []
        ch.alive = False
        bit = 1 << index
        dormant = []
        for c in self.claims.values():
            if c.channel_mask & bit:
                c.channel_mask &= ~bit
                if c.channel_mask == 0 and c.status not in (ClaimStatus.RETRACTED,):
                    c.status = ClaimStatus.DORMANT
                    dormant.append(c.claim_id)
        self.log("kill_channel", channel=index, dormant=dormant)
        return dormant

    # --- nogoods (05 §3) ------------------------------------------------------------------------
    def add_nogood(self, content_keys: frozenset[str]) -> None:
        if len(self.nogoods) < self.cfg.workspace.NG_max:
            self.nogoods.add(content_keys)
            self.log("nogood", keys=sorted(content_keys))

    def is_nogood(self, content_keys: frozenset[str]) -> bool:
        return any(ng <= content_keys for ng in self.nogoods)

    # --- referents and goals --------------------------------------------------------------------
    def add_referent(self, ref: Referent) -> Referent:
        if len(self.referents) >= self.R and ref.ref_id not in self.referents:
            raise WorkspaceFull(f"referent slots full (R={self.R})")
        self.referents[ref.ref_id] = ref
        return ref

    def add_goal(self, goal: Goal) -> Goal:
        if len(self.goals) >= self.cfg.workspace.G_max:
            raise WorkspaceFull(f"goal stack full (G_max={self.cfg.workspace.G_max})")
        self.goals[goal.goal_id] = goal
        return goal
