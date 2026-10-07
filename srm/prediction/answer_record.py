"""Answer Record: why the model answered (06 §7)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from srm.core.workspace import Workspace
from srm.prediction.decision import Decision
from srm.prediction.support import presentation_weights


@dataclass
class AnswerRecord:
    goal_id: str
    decision_class: str
    decision_state: str | None
    answer: Any
    answer_claims: list[str]
    justification_subgraph: dict[str, dict[str, Any]]
    sources: list[dict[str, Any]]
    circuit_trace: list[dict[str, Any]]
    rejected_hypotheses: list[dict[str, Any]]
    taint_map: dict[str, dict[str, Any]]
    envelope_flags: dict[str, bool]
    budget: dict[str, Any]
    reason: str
    missing_keys: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_answer_record(ws: Workspace, goal_id: str, decision: Decision, sources: list[dict[str, Any]],
                        budget: dict[str, Any] | None = None) -> AnswerRecord:
    best = decision.best
    answer_claims = list(best.claim_ids) if best is not None else []
    # justification subgraph: all ancestors of the answer claims
    sub: dict[str, dict[str, Any]] = {}
    stack = list(answer_claims)
    while stack:
        cid = stack.pop()
        if cid in sub or cid not in ws.claims:
            continue
        c = ws.claims[cid]
        sub[cid] = {
            "producer": c.producer_id, "premises": list(c.premises), "state": c.state.value if c.state else None,
            "status": c.status.value, "evidence": c.ledger.to_dict(), "record_id": c.record_id,
        }
        stack.extend(c.premises)
    weights = presentation_weights(decision.ranked)
    rejected = [
        {"hypothesis": h.hyp_id, "binding": h.binding.to_dict() if h.binding is not None else None,
         "support": h.support, "belief": h.b, "display_weight": w, "residual": h.is_residual}
        for h, w in zip(decision.ranked, weights) if h is not best
    ]
    taint = {cid: {"tainted": ws.claims[cid].tainted, "promoted_by": list(ws.claims[cid].promoted_by)}
             for cid in sub}
    envelope = {h.hyp_id: h.in_envelope for h in decision.ranked if not h.is_residual}
    return AnswerRecord(
        goal_id=goal_id, decision_class=decision.cls.value,
        decision_state=decision.state.value if decision.state else None,
        answer=decision.answer.to_dict() if decision.answer is not None else None,
        answer_claims=answer_claims, justification_subgraph=sub, sources=sources,
        circuit_trace=[t for t in ws.trace], rejected_hypotheses=rejected, taint_map=taint,
        envelope_flags=envelope, budget=budget or {}, reason=decision.reason, missing_keys=decision.missing_keys,
    )
