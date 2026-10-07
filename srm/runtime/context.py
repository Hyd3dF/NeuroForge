"""Per-query runtime context (11 §3) and claim helpers shared by circuit ops."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from srm.body.tools import Body
from srm.config.build_config import BuildConfig
from srm.core import tms
from srm.core.evidence import EvidenceLedger, initial_evidence
from srm.core.workspace import Claim, Workspace
from srm.interface.layer import InterfaceLayer
from srm.interface.messages import (
    LIFECYCLE_ORDER,
    ClaimStatus,
    EpistemicState,
    Goal,
    Lifecycle,
    Qualifiers,
)
from srm.interface.values import Value
from srm.memory.system import MemorySystem
from srm.prediction import taint as T
from srm.prediction.decision import Decision
from srm.prediction.support import HypothesisEval

TASK_INPUT_ROOT = "input:task"


@dataclass
class Accounting:
    beats: int = 0
    flops: float = 0.0
    bytes: float = 0.0
    records_touched: set[int] = field(default_factory=set)
    tool_calls: int = 0
    verifier_calls: int = 0
    funded_actions: int = 0
    unfunded_bids: int = 0

    def to_dict(self, total_records: int) -> dict[str, Any]:
        return {
            "beats": self.beats, "flops": self.flops, "bytes": self.bytes,
            "records_touched": len(self.records_touched), "total_records": total_records,
            "active_record_fraction": len(self.records_touched) / max(1, total_records),
            "tool_calls": self.tool_calls, "verifier_calls": self.verifier_calls,
            "funded_actions": self.funded_actions, "unfunded_bids": self.unfunded_bids,
        }


@dataclass
class QueryContext:
    cfg: BuildConfig
    interface: InterfaceLayer
    memory: MemorySystem
    body: Body
    ws: Workspace
    request: Any
    stakes: float
    goal: Goal | None = None
    graph: Any = None
    hypotheses: list[HypothesisEval] = field(default_factory=list)
    goal_keys: list[str] = field(default_factory=list)
    input_keys: set[str] = field(default_factory=set)
    open_question: bool = False
    pending_derivations: bool = False
    budget_exhausted: bool = False
    decision: Decision | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    accounting: Accounting = field(default_factory=Accounting)
    sources: list[dict[str, Any]] = field(default_factory=list)
    beat: int = 0
    notes: list[str] = field(default_factory=list)
    contested_fn: Any = None  # record_id → bool (from the ingestor's contradiction registry)

    # --- claim helpers -----------------------------------------------------------------------------
    def is_remembered(self, rid: int) -> bool:
        rec = self.memory.record(rid)
        b, _, _ = self.memory.opinion(rid)
        if b < self.cfg.epistemics.theta_commit:
            return False
        if LIFECYCLE_ORDER.get(rec.lifecycle, -1) >= LIFECYCLE_ORDER[Lifecycle.CORROBORATED]:
            return True
        return max((p.get("trust", 0.0) for p in rec.provenance), default=0.0) >= \
            self.cfg.epistemics.remembered_single_source_min_trust

    def touch(self, rid: int) -> None:
        self.memory.touch(rid)
        self.accounting.records_touched.add(rid)
        i = self.cfg.interface
        self.accounting.bytes += i.B * 3 + i.d * 2 + i.d_k * 2  # codes + dense + key

    def claim_from_record(self, rid: int, subject: str, relation: str, obj: Value | str,
                          qualifiers: Qualifiers | None = None, producer: str = "RETRIEVE") -> Claim:
        rec = self.memory.record(rid)
        self.touch(rid)
        c = Claim(
            claim_id=self.ws.new_id(), subject=subject, relation=relation, object=obj,
            qualifiers=qualifiers or Qualifiers(), producer_id=producer, producer_exact=True,
            record_id=rid, provenance=[p.get("source_id", "") for p in rec.provenance],
        )
        c.ledger = rec.ledger.merge(EvidenceLedger())
        c.grounded_leaf = self.is_remembered(rid)
        c.state = EpistemicState.REMEMBERED if c.grounded_leaf else None
        c.status = ClaimStatus.CHECKED
        self.ws.add_claim(c)
        self.ws.protected.add(c.claim_id)
        self.sources.extend({"source_id": p.get("source_id"), "trust": p.get("trust")} for p in rec.provenance)
        return c

    def observed_claim(self, subject: str, relation: str, obj: Value | str, root: str = TASK_INPUT_ROOT,
                       trust_class: str = "task_given", producer: str = "INTAKE") -> Claim:
        """A claim grounded in the current input (OBSERVED, 06 §5).

        Task-given premises carry ``task_given`` trust (D-026): they are the premises of the
        task frame, not assertions about the world.
        """
        e = self.cfg.epistemics
        trust = e.trust_priors.get(trust_class, 1.0)
        c = Claim(claim_id=self.ws.new_id(), subject=subject, relation=relation, object=obj, producer_id=producer)
        c.ledger.add(root, initial_evidence(e.kappa, trust, 1.0, 1.0), 0.0)
        c.state = EpistemicState.OBSERVED
        c.grounded_leaf = True
        c.status = ClaimStatus.CHECKED
        self.ws.add_claim(c)
        self.ws.protected.add(c.claim_id)
        return c

    def derived_claim(self, subject: str, relation: str, obj: Value | str, premises: tuple[str, ...], producer: str,
                      exact: bool = True, predictive: bool = False, reliability: float = 1.0,
                      state: EpistemicState | None = None) -> Claim:
        c = Claim(
            claim_id=self.ws.new_id(), subject=subject, relation=relation, object=obj, premises=premises,
            producer_id=producer, producer_exact=exact, producer_predictive=predictive,
            producer_reliability=reliability,
        )
        c.tainted = T.derive_taint(self.ws, premises, predictive)
        self.ws.add_claim(c)
        tms.refresh(self.ws, c.claim_id)
        if state is not None:
            c.state = state
        elif not c.tainted and all(self.ws.claims[p].grounded_leaf or self.ws.claims[p].state in
                                   (EpistemicState.DERIVED, EpistemicState.TESTED, EpistemicState.OBSERVED,
                                    EpistemicState.REMEMBERED) for p in premises):
            c.state = EpistemicState.DERIVED
        else:
            c.state = EpistemicState.PREDICTED if predictive else None
        self.ws.protected.add(c.claim_id)
        return c

    def add_test_evidence(self, claim_id: str, passed: bool, root: str) -> None:
        e = self.cfg.epistemics
        c = self.ws.claims[claim_id]
        c.ledger.add(root, e.kappa_t if passed else 0.0, 0.0 if passed else e.kappa_t)
        self.accounting.tool_calls += 1
        if passed:
            if c.tainted:
                T.promote(self.ws, claim_id, T.Promotion.TEST_PASS, self.cfg, self.stakes)
            c.state = EpistemicState.TESTED
            c.grounded_leaf = True
        tms.propagate(self.ws, claim_id)
        tms.check_retraction(self.ws, claim_id)

    def grounded(self, claim_id: str) -> bool:
        """Path to OBSERVED / REMEMBERED / TESTED leaves through untainted claims (06 §6)."""
        c = self.ws.claims[claim_id]
        if c.tainted or c.status == ClaimStatus.RETRACTED:
            return False
        if c.grounded_leaf:
            return True
        return bool(c.premises) and all(self.grounded(p) for p in c.premises)
