"""Factual answering path (11 §6.1–6.2): goal → Sketch → open questions → structured retrieval →
workspace claims → hypothesis set → Support → decision → Answer Record.

This is the epistemic vertical slice (14 §4 "M0–M3 + M5 factual path").  The full beat
loop (Heart, Selector, Composer) arrives in the next batch and reuses these pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from srm.config.build_config import BuildConfig
from srm.core.evidence import EvidenceLedger
from srm.core.workspace import Claim, Workspace
from srm.data.sef import QuestionRecord
from srm.interface.messages import (
    LIFECYCLE_ORDER,
    ClaimStatus,
    DecisionClass,
    EpistemicState,
    Goal,
    Lifecycle,
    PatternAtom,
    Qualifiers,
)
from srm.interface.codec import normalize_surface
from srm.interface.values import Value
from srm.memory import sketch as SK
from srm.memory.payloads import EngramPayload
from srm.memory.system import MemorySystem
from srm.prediction.answer_record import AnswerRecord, build_answer_record
from srm.prediction.decision import Decision, DecisionInputs, decide
from srm.prediction.support import HypothesisEval


@dataclass
class FactualAnswer:
    goal: Goal
    decision: Decision
    record: AnswerRecord
    workspace: Workspace

    @property
    def cls(self) -> DecisionClass:
        return self.decision.cls

    @property
    def answer(self) -> Value | None:
        return self.decision.answer


def goal_from_question(q: QuestionRecord, goal_id: str, stakes: float) -> Goal:
    if q.pattern is None or not q.pattern.atoms:
        raise ValueError("question needs a structured pattern (the text parser arrives in M5/S2)")
    atoms = []
    for a in q.pattern.atoms:
        obj: Any = a.object
        if not isinstance(obj, str):
            obj = Value(obj.type, obj.value, obj.unit, float(obj.tolerance or 0.0))
        atoms.append(PatternAtom(a.subject, a.relation, obj))
    answer_vars = [(v, "any") for a in atoms for v in a.variables()]
    quals = Qualifiers(time=q.context.time, version=q.context.version, world_id=q.context.world_id)
    surface = {a.subject: a.subject[1:] for a in atoms if a.subject.startswith("@")}
    return Goal(goal_id, atoms, answer_vars, quals, stakes, surface_forms=surface)


class FactualAnswerer:
    def __init__(self, memory: MemorySystem, config: BuildConfig, contested: Any = None) -> None:
        self.mem = memory
        self.cfg = config
        self._contested = contested  # callable(record_id) -> bool, from the ingestor
        self._n = 0

    def is_remembered(self, rid: int) -> bool:
        """REMEMBERED leaf (06 §5): b ≥ θ_commit and (lifecycle ≥ CORROBORATED or a high-trust single source)."""
        rec = self.mem.record(rid)
        b, _, _ = self.mem.opinion(rid)
        if b < self.cfg.epistemics.theta_commit:
            return False
        if LIFECYCLE_ORDER.get(rec.lifecycle, -1) >= LIFECYCLE_ORDER[Lifecycle.CORROBORATED]:
            return True
        trusts = [p.get("trust", 0.0) for p in rec.provenance]
        return max(trusts, default=0.0) >= self.cfg.epistemics.remembered_single_source_min_trust

    def answer(self, question: QuestionRecord, stakes: float | None = None) -> FactualAnswer:
        self._n += 1
        stakes = self.cfg.epistemics.default_stakes if stakes is None else stakes
        goal = goal_from_question(question, f"g{self._n}", stakes)
        ws = Workspace(self.cfg)
        ws.add_goal(goal)
        atom = goal.atoms[0]

        # entity linking (02 §7.1); unlinked surface forms become alias keys
        keys: list[str] = []
        subjects: list[str] = []
        if atom.subject.startswith("@"):
            linked = self.mem.link_surface(atom.subject[1:])
            if linked:
                subjects = linked
            else:
                keys.append(SK.key_alias(normalize_surface(atom.subject[1:])))
        else:
            subjects = [atom.subject]
        for s in subjects:
            keys += [SK.key_entity(s), SK.key_entity_relation(s, atom.relation)]
        ws.log("goal", goal=goal.goal_id, subjects=subjects, relation=atom.relation)

        open_q = any(self.mem.find_open_questions(s, atom.relation) for s in subjects)

        # structured retrieval → claims → hypotheses grouped by object value
        hyps: list[HypothesisEval] = []
        sources: list[dict[str, Any]] = []
        for s in subjects:
            for rid in self.mem.retrieve_pattern(s, atom.relation, qualifiers=goal.qualifiers):
                rec = self.mem.record(rid)
                p: EngramPayload = rec.payload
                if p.polarity != "positive":
                    continue
                claim = Claim(
                    claim_id=ws.new_id(), subject=p.subject, relation=p.relation, object=p.object,
                    qualifiers=p.qualifiers, polarity=p.polarity, modality=p.modality,
                    producer_id="RETRIEVE_PATTERN", producer_exact=True, producer_predictive=False,
                    record_id=rid, provenance=[pr.get("source_id", "") for pr in rec.provenance],
                )
                claim.ledger = rec.ledger.merge(EvidenceLedger())  # copy
                claim.grounded_leaf = self.is_remembered(rid)
                claim.state = EpistemicState.REMEMBERED if claim.grounded_leaf else None
                claim.status = ClaimStatus.CHECKED
                ws.add_claim(claim)
                ws.protected.add(claim.claim_id)
                self.mem.touch(rid)
                sources.extend({"source_id": pr.get("source_id"), "trust": pr.get("trust")} for pr in rec.provenance)
                contested = bool(self._contested(rid)) if self._contested else rec.lifecycle == Lifecycle.CONTESTED
                h = next((h for h in hyps if h.binding is not None and h.binding.matches(p.object)), None)
                if h is None:
                    h = HypothesisEval(hyp_id=f"h{len(hyps)}", binding=p.object)
                    hyps.append(h)
                h.claim_ids.append(claim.claim_id)
                h.ledger = h.ledger.merge(claim.ledger)
                h.grounded = h.grounded or claim.grounded_leaf
                h.n_unresolved_contradictions += int(contested)
                if claim.grounded_leaf:
                    h.knowledge_state = EpistemicState.REMEMBERED
        hyps.append(HypothesisEval(hyp_id="residual", binding=None, is_residual=True))

        decision = decide(DecisionInputs(
            goal_keys=keys, absent=self.mem.sketch.definitely_absent, open_question=open_q,
            hypotheses=hyps, stakes=stakes,
        ), self.cfg)
        if decision.cls == DecisionClass.KNOWN and decision.best is not None:
            for cid in decision.best.claim_ids:
                ws.claims[cid].status = ClaimStatus.COMMITTED
                rid = ws.claims[cid].record_id
                if rid is not None:
                    self.mem.update_lifecycle(rid)
        ws.log("decision", cls=decision.cls.value, reason=decision.reason)
        record = build_answer_record(ws, goal.goal_id, decision, sources, {"beats": 1})
        return FactualAnswer(goal, decision, record, ws)
