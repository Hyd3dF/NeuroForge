"""The SRM runtime: query lifecycle and the beat loop (11 §2–3).

Phases per beat (11 §2): 0 modulate · 1 perceive (intake beat) · 2 background (Subconscious)
· 3 diastole (bids) · 4 systole (Heart allocation, dispatch in fixed order) · 5 epistemic
update · 6 maintain · 7 learn (credit, TD, Hebbian) · 8 decide · 9 trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable

from srm.body.tools import Body
from srm.config.build_config import BuildConfig
from srm.config.profiles import f0
from srm.control import Gate, Heart, Modulators, Subconscious
from srm.core.circuit import CircuitGraph, NodeState, OpResult
from srm.core.composer import Composer
from srm.core.error_monitor import ErrorMonitor
from srm.core.evidence import EvidenceLedger
from srm.core.ops import OPS
from srm.core.selector import Selector
from srm.core.workspace import Claim, Workspace
from srm.data.io import Dataset
from srm.data.sef import RecordBase
from srm.ingest import Ingestor, IngestReport
from srm.interface.layer import InterfaceLayer
from srm.interface.messages import Bid, ClaimStatus, DecisionClass, EpistemicState, Goal, RecordKind
from srm.memory.payloads import EpisodePayload, ProcedurePayload
from srm.memory.system import HIPPOCAMPUS, MemorySystem
from srm.mouth import Mouth, Rendered, UtterancePlan
from srm.perception import Perception, Request, intake, relation_lexicon
from srm.perception.encoder import Percept
from srm.prediction import taint as T
from srm.prediction.answer_record import AnswerRecord, build_answer_record
from srm.prediction.decision import Decision, DecisionInputs, decide
from srm.prediction.support import HypothesisEval
from srm.runtime.context import QueryContext

DISPATCH_ORDER = {"retrieve": 0, "synthesis": 1, "transform": 2, "test": 3, "simulate": 3, "answer": 4}
SELF_SOURCE = "self:episodes"


@dataclass
class Response:
    text: str
    request: Request
    decision: Decision
    answer_record: AnswerRecord
    plan: UtterancePlan
    rendered: Rendered
    accounting: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)
    circuit: list[dict[str, Any]] = field(default_factory=list)

    @property
    def cls(self) -> DecisionClass:
        return self.decision.cls

    @property
    def answer(self) -> Any:
        return self.decision.answer


class SRM:
    """A runnable Final SRM instance (F0 profile by default)."""

    def __init__(self, config: BuildConfig | None = None) -> None:
        self.cfg = config or f0()
        self.interface = InterfaceLayer(self.cfg)
        self.memory = MemorySystem(self.cfg, self.interface)
        self.ingestor = Ingestor(self.memory, self.cfg)
        self.body = Body(self.cfg.body.tools, self.cfg.body.sandbox_timeout_s, self.cfg.body.sandbox_memory_mb)
        self.perception = Perception(self.cfg, self.interface, self.memory)
        self.composer = Composer(self.cfg)
        self.selector = Selector(self.cfg)
        self.error_monitor = ErrorMonitor(self.cfg)
        self.heart = Heart(self.cfg)
        self.gate = Gate()
        self.modulators = Modulators()
        self.subconscious = Subconscious(self.cfg)
        self.mouth = Mouth(self.name_of)
        self._lexicon: dict[str, str] = {}
        self._lexicon_size = -1
        self._queries = 0
        self.memory.register_source(SELF_SOURCE, "model_output", "model_output")

    # --- knowledge ---------------------------------------------------------------------------------
    def ingest(self, data: Dataset | Iterable[RecordBase]) -> IngestReport:
        records = data.records if isinstance(data, Dataset) else list(data)
        return self.ingestor.ingest(records)

    def name_of(self, symbol: str) -> str:
        rid = self.memory.symbols.get(symbol)
        if rid is not None:
            rec = self.memory.record(rid)
            if rec.kind == RecordKind.ENTITY:
                return rec.payload.name
        return symbol.rsplit(":", 1)[-1]

    def lexicon(self) -> dict[str, str]:
        if len(self.interface.relations) != self._lexicon_size:
            self._lexicon = relation_lexicon(self.interface.relations)
            self._lexicon_size = len(self.interface.relations)
        return self._lexicon

    # --- query lifecycle (11 §3) ---------------------------------------------------------------------
    def ask(self, request: Any, stakes: float | None = None, max_beats: int | None = None,
            learn: bool = True) -> Response:
        self._queries += 1
        stakes = self.cfg.epistemics.default_stakes if stakes is None else stakes
        req = intake(request, self.lexicon())
        ws = Workspace(self.cfg)
        ctx = QueryContext(self.cfg, self.interface, self.memory, self.body, ws, req, stakes,
                           contested_fn=self.ingestor.contested)
        variables = [v for a in req.atoms for v in a.variables()]
        final = "?x" if "?x" in variables else (variables[-1] if variables else "?x")
        answer_vars = [(final, "any")]  # the goal's answer variable; intermediate hops are internal
        ctx.goal = ws.add_goal(Goal(f"q{self._queries}", list(req.atoms), answer_vars, req.qualifiers, stakes))
        percept = self.perception.process(req.text) if req.text else Percept()
        graph = self.composer.build(ctx)
        ctx.graph = graph
        ws.log("intake", kind=req.kind, novelty=percept.novelty, nodes=len(graph.nodes))

        budget = self.heart.query_budget(stakes, self.modulators.ne)
        spent = 0.0
        t_max = max_beats or self.cfg.control.T_max
        surprise = 0
        for beat in range(1, t_max + 1):
            ctx.beat = beat
            ctx.accounting.beats = beat
            # 0 modulate
            u = self.selector.goal_uncertainty(ctx)
            td = self.selector.td_update(ctx, 1.0 - u)
            self.modulators.update(td, percept.mean_error if beat == 1 else 0.0, surprise, stakes,
                                   max(0.0, 1.0 - spent / budget))
            surprise = 0
            # 2 background (Subconscious) — never commits; outputs SUGGESTED
            bids: list[Bid] = []
            categories: dict[str, str] = {}
            buffer = self.subconscious.prime(ctx, set(ctx.accounting.records_touched))
            if beat == 1:
                buffer += self.subconscious.match_percepts(ctx, percept)
            if buffer:
                bids.append(Bid("subconscious", {"flops": self.cfg.control.op_flops["SUBCONSCIOUS"]}, 0.05))
                categories["subconscious"] = "subconscious"
            # 3 diastole: Selector proposes funded actions
            actions = self.selector.propose(ctx, graph)
            if self.selector.should_stop(ctx, graph, actions):
                break
            node_of = {}
            for a in actions:
                pid = f"node:{a.node.node_id}"
                node_of[pid] = a
                bids.append(Bid(pid, {"flops": a.flops}, a.voc))
                if OPS[a.node.op].kind == "test":
                    categories[pid] = "error_monitor"
            # 4 systole
            alloc = self.heart.allocate(bids, budget - spent)
            ctx.accounting.funded_actions += len(alloc.funded)
            ctx.accounting.unfunded_bids += len(alloc.rejected)
            funded_ids = {b.process_id for b in alloc.funded}
            if "subconscious" in funded_ids:
                self._admit(ctx, buffer)
                for store_id, page in self.subconscious.prefetch_pages(buffer):
                    self.heart.note_page(store_id, page)
            ordered = sorted((node_of[b.process_id] for b in alloc.funded if b.process_id in node_of),
                             key=lambda a: (DISPATCH_ORDER[OPS[a.node.op].kind], a.node.node_id))
            for a in ordered:
                res = self._execute(ctx, graph, a.node)
                self.heart.update_credit(f"node:{a.node.node_id}", a.voc, res.value_signal * max(a.voc, 1e-3))
                if not res.ok and a.node.state == NodeState.FAILED:
                    surprise += 1
                    if self.composer.repair(ctx, graph, a.node):
                        a.node.attempts += 1
            spent += alloc.flops
            # 5 epistemic update
            surprise += len(self.error_monitor.contradiction_scan(ws, self.interface.relations.is_functional))
            # 7 learn: Hebbian co-activation of the records this beat touched (bounded)
            touched = sorted(ctx.accounting.records_touched)[-8:]
            if len(touched) > 1:
                self.memory.assoc.hebbian({r: 1.0 for r in touched}, self.cfg.memory.eta_H,
                                          self.modulators.hebbian_modulator or 0.1, self.cfg.memory.eta_decay,
                                          self.cfg.memory.w_max)
            # 8 decide / budget
            if graph.finished():
                break
            if spent >= budget:
                ctx.budget_exhausted = True
                break
        else:
            ctx.budget_exhausted = True
        if not graph.finished():
            ctx.pending_derivations = True
            ctx.budget_exhausted = True
        ctx.accounting.flops = spent

        decision = decide(DecisionInputs(
            goal_keys=list(dict.fromkeys(ctx.goal_keys)), absent=self.memory.sketch.definitely_absent,
            input_keys=ctx.input_keys, open_question=ctx.open_question,
            hypotheses=ctx.hypotheses + [HypothesisEval(hyp_id="residual", binding=None, is_residual=True)],
            budget_exhausted=ctx.budget_exhausted, pending_derivations=ctx.pending_derivations, stakes=stakes,
        ), self.cfg)
        ctx.decision = decision
        self._commit(ctx, decision)
        record = build_answer_record(ws, ctx.goal.goal_id, decision, ctx.sources,
                                     {"beats": ctx.accounting.beats, "flops": spent, "budget": budget})
        record.circuit_trace = graph.trace()
        plan = self.mouth.plan(req, decision, record, ctx.artifacts, ws)
        rendered = self.mouth.render(plan)
        if learn:
            self._post(ctx, decision)
        return Response(rendered.text, req, decision, record, plan, rendered,
                        ctx.accounting.to_dict(self.memory.n_records()), ctx.artifacts, graph.trace())

    # --- helpers --------------------------------------------------------------------------------------
    def _execute(self, ctx: QueryContext, graph: CircuitGraph, node: Any) -> OpResult:
        spec = OPS[node.op]
        node.state = NodeState.RUNNING
        try:
            res = spec.fn(ctx, node, graph.resolve(node))
        except Exception as exc:  # noqa: BLE001 — a failing op is an observation, not a crash (11 §4)
            res = OpResult(ok=False, note=f"{type(exc).__name__}: {exc}")
        flops = res.flops or self.cfg.control.op_flops.get(spec.cost_key, 1e5)
        node.flops_spent += flops
        node.produced.extend(res.claims)
        node.note = res.note or node.note
        if not res.done:
            node.state = NodeState.RUNNING
        else:
            node.outputs.update({k: v for k, v in res.outputs.items()})
            node.state = NodeState.DONE if res.ok else NodeState.FAILED
        ctx.ws.log("op", node=node.node_id, op=node.op, ok=res.ok, done=res.done, note=res.note)
        return res

    def _admit(self, ctx: QueryContext, buffer: list[Any]) -> None:
        free = max(0, min(4, ctx.ws.K - len(ctx.ws.claims) - 8))
        for item in self.gate.admit(buffer, free, self.modulators.gain_ach, self.cfg.interface.B):
            rec = self.memory.record(item.record_id)
            p = rec.payload
            if rec.kind != RecordKind.ENGRAM:
                continue
            c = Claim(claim_id=ctx.ws.new_id(), subject=p.subject, relation=p.relation, object=p.object,
                      producer_id="SUBCONSCIOUS", producer_predictive=True, tainted=True,
                      state=EpistemicState.SUGGESTED, record_id=item.record_id, relevance=item.salience)
            c.ledger = rec.ledger.merge(EvidenceLedger())
            ctx.ws.add_claim(c)

    def _commit(self, ctx: QueryContext, decision: Decision) -> None:
        if decision.cls != DecisionClass.KNOWN or decision.best is None:
            return
        for cid in decision.best.claim_ids:
            c = ctx.ws.claims[cid]
            T.check_invariant(c, rendered_as_knowledge=True)  # taint-leak invariant (12 §4.6)
            c.status = ClaimStatus.COMMITTED
            for p in self._leaf_records(ctx, cid):
                self.memory.update_lifecycle(p)

    def _leaf_records(self, ctx: QueryContext, cid: str) -> list[int]:
        c = ctx.ws.claims[cid]
        out = [c.record_id] if c.record_id is not None else []
        for p in c.premises:
            if p in ctx.ws.claims:
                out += self._leaf_records(ctx, p)
        return out

    def _post(self, ctx: QueryContext, decision: Decision) -> None:
        """Episode write + online library learning (08 §4.1, §6.1 candidates)."""
        outcome = {"class": decision.cls.value, "state": decision.state.value if decision.state else None,
                   "answer": decision.answer.to_dict() if decision.answer is not None else None}
        ep = EpisodePayload(episode_id=f"episode:{ctx.goal.goal_id}:{self._queries}", goal=ctx.request.text or ctx.request.kind,
                            kind=ctx.request.kind, outcome=outcome, trace=ctx.graph.trace(),
                            record_ids=sorted(ctx.accounting.records_touched))
        dense = self.perception.encoder.encode(ctx.request.text or ctx.request.kind)
        self.memory.write(RecordKind.EPISODE, ep.episode_id, dense if dense.any() else dense + 1.0, ep, target=HIPPOCAMPUS)
        if ctx.request.kind in ("dsl_task", "code_task") and decision.cls in (
                DecisionClass.KNOWN, DecisionClass.AMBIGUOUS, DecisionClass.PREDICTION) and ctx.artifacts.get("programs"):
            prog = Mouth._program_for(decision, ctx.artifacts)
            if prog is None:
                return
            pid = f"proc:dsl:{json.dumps(prog['json'], sort_keys=True)}"
            if pid not in self.memory.symbols:
                payload = ProcedurePayload(procedure_id=pid, domain="dsl", signature=f"dsl:{ctx.request.ret_type}",
                                           goal="solved DSL task", program=prog["json"], uses=1, successes=1)
                ledger = EvidenceLedger()
                ledger.add(SELF_SOURCE, self.cfg.epistemics.trust_priors["model_output"], 0.0)
                self.memory.write(RecordKind.PROCEDURE, pid, self.perception.encoder.encode(prog["dsl"]), payload, ledger,
                                  [{"source_id": SELF_SOURCE, "root": SELF_SOURCE, "trust": 0.2, "episode": ep.episode_id}],
                                  target=HIPPOCAMPUS)
