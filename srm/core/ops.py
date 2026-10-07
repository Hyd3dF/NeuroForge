"""Circuit operations executed by the beat loop (05 §4–5, 11 §2 phase 4).

Each op reads resolved inputs, writes claims into the workspace through the query
context, and returns an :class:`OpResult`.  Ops that run for several beats
(synthesis) are ``incremental``: they stay RUNNING and are funded again each beat.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import sympy as sp

from srm.body import dsl
from srm.core.circuit import CircuitNode, OpResult
from srm.core.simulator import group_by_behaviour, probe_inputs
from srm.core.synthesis import Synthesizer
from srm.interface.codec import normalize_surface
from srm.interface.messages import EpistemicState, Lifecycle, PatternAtom, is_var
from srm.interface.values import Value
from srm.memory import sketch as SK
from srm.memory.payloads import EngramPayload, ProcedurePayload
from srm.prediction.support import HypothesisEval

OpFn = Callable[[Any, CircuitNode, dict[str, Any]], OpResult]


@dataclass(frozen=True)
class OpSpec:
    fn: OpFn
    kind: str  # retrieve | transform | test | synthesis | simulate | answer
    cost_key: str
    incremental: bool = False


# =================================================================================================
# factual: LINK → JOIN_HOP* → ANSWER_FACT
# =================================================================================================
def op_link(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    surface: str = node.params["surface"]
    if surface.startswith("@"):
        name = surface[1:]
        ids = ctx.memory.link_surface(name)
        if not ids:
            ctx.goal_keys.append(SK.key_alias(normalize_surface(name)))
            return OpResult(ok=False, outputs={"frontier": []}, note=f"no entity named {name!r}")
    else:
        ids = [surface]
    frontier = [({surface: Value("entity_ref", i)}, None) for i in ids]
    for i in ids:
        ctx.goal_keys.append(SK.key_entity(i))
    return OpResult(outputs={"frontier": frontier}, note=f"linked {len(ids)}")


def op_join_hop(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    atom: PatternAtom = node.params["atom"]
    frontier = inp.get("frontier") or []
    out_frontier: list[tuple[dict[str, Any], str | None]] = []
    produced: list[str] = []
    for bindings, last in frontier:
        subj = str(bindings[atom.subject].value) if (is_var(atom.subject) or atom.subject.startswith("@")) else atom.subject
        ctx.goal_keys.append(SK.key_entity_relation(subj, atom.relation))
        if ctx.memory.find_open_questions(subj, atom.relation):
            ctx.open_question = True
        for rid in ctx.memory.retrieve_pattern(subj, atom.relation, qualifiers=ctx.goal.qualifiers):
            rec = ctx.memory.record(rid)
            p: EngramPayload = rec.payload
            if p.polarity != "positive":
                continue
            leaf = ctx.claim_from_record(rid, p.subject, p.relation, p.object, p.qualifiers, producer="JOIN_HOP")
            produced.append(leaf.claim_id)
            if rec.lifecycle == Lifecycle.DEPRECATED:
                continue  # retracted knowledge is never chained through
            claim_id = leaf.claim_id
            if last is not None:
                d = ctx.derived_claim(ctx.goal.goal_id, f"chain:{node.node_id}", p.object, (last, leaf.claim_id), "JOIN_HOP")
                claim_id = d.claim_id
                produced.append(claim_id)
            nb = dict(bindings)
            if is_var(atom.object):
                nb[atom.object] = p.object
            out_frontier.append((nb, claim_id))
    return OpResult(outputs={"frontier": out_frontier}, claims=produced, value_signal=float(bool(out_frontier)))


def _leaf_records(ctx: Any, claim_id: str) -> list[int]:
    c = ctx.ws.claims[claim_id]
    if c.record_id is not None:
        return [c.record_id]
    out: list[int] = []
    for p in c.premises:
        out += _leaf_records(ctx, p)
    return out


def op_answer_fact(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    var: str = node.params["var"]
    hyps: list[HypothesisEval] = []
    for bindings, claim_id in inp.get("frontier") or []:
        if claim_id is None or var not in bindings:
            continue
        value: Value = bindings[var]
        c = ctx.ws.claims[claim_id]
        h = next((h for h in hyps if h.binding is not None and h.binding.matches(value)), None)
        if h is None:
            h = HypothesisEval(hyp_id=f"h{len(hyps)}", binding=value)
            hyps.append(h)
        h.claim_ids.append(claim_id)
        h.ledger = h.ledger.merge(c.ledger)
        h.grounded = h.grounded or ctx.grounded(claim_id)
        h.tainted = h.tainted or c.tainted
        contested = ctx.contested_fn is not None and any(ctx.contested_fn(r) for r in _leaf_records(ctx, claim_id))
        h.n_unresolved_contradictions += int(contested)
        if h.grounded:
            h.knowledge_state = c.state if c.state is not None else EpistemicState.DERIVED
    ctx.hypotheses = hyps
    return OpResult(outputs={"n_hypotheses": len(hyps)}, value_signal=1.0)


# =================================================================================================
# DSL programming: SYNTH_DSL → TEST_DSL → SIMULATE_DSL → [PY_CHECK] → ANSWER_DSL
# =================================================================================================
def op_synth_dsl(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    syn: Synthesizer | None = node.outputs.get("_synth")
    if syn is None:
        library = []
        for rid in ctx.memory.find_procedures("dsl", f"dsl:{node.params['ret_type']}"):
            prog = ctx.memory.record(rid).payload.program
            if prog is not None:
                library.append(dsl.Node.from_json(prog))
                ctx.touch(rid)
        syn = Synthesizer(node.params["examples"], node.params["ret_type"], library,
                          max_size=node.params.get("max_size", ctx.cfg.core.synth_max_size),
                          max_solutions=8)
        node.outputs["_synth"] = syn
        node.outputs["library_size"] = len(library)
    before = syn.evaluated
    finished = syn.step(ctx.cfg.core.synth_evals_per_beat)
    flops = (syn.evaluated - before) * ctx.cfg.control.op_flops["SYNTH_DSL_EVAL"]
    if not finished and syn.evaluated < ctx.cfg.core.synth_max_evals:
        return OpResult(done=False, flops=flops, note=f"evaluated {syn.evaluated}")
    node.outputs["candidates"] = list(syn.solutions)
    node.outputs["evaluated"] = syn.evaluated
    ok = bool(syn.solutions)
    return OpResult(ok=ok, flops=flops, outputs=node.outputs,
                    note=f"{len(syn.solutions)} candidates after {syn.evaluated} evaluations", value_signal=float(ok))


def op_test_dsl(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    examples = node.params["examples"]
    spec = ctx.observed_claim(ctx.goal.goal_id, "spec:examples", Value("text", json.dumps(examples)))
    passing, produced = [], [spec.claim_id]
    for prog in inp.get("candidates") or []:
        c = ctx.derived_claim(ctx.goal.goal_id, "solves", Value("code_ref", json.dumps(prog.to_json())),
                              (spec.claim_id,), "SYNTH_DSL", exact=False, predictive=True)
        produced.append(c.claim_id)
        res = ctx.body.call("tests", tests=examples, program=prog)
        ctx.add_test_evidence(c.claim_id, res.ok, root=f"test:examples:{c.claim_id}")
        if res.ok:
            passing.append((prog, c.claim_id))
    return OpResult(ok=bool(passing), outputs={"passing": passing}, claims=produced, value_signal=float(bool(passing)))


def op_simulate_dsl(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    passing = inp.get("passing") or []
    rng = np.random.default_rng(len(ctx.ws.trace))
    inputs = probe_inputs(rng, ctx.cfg.core.simulate_probes, node.params.get("examples"))
    groups = group_by_behaviour([p for p, _ in passing], inputs)
    claim_of = {id(p): cid for p, cid in passing}
    out = []
    for g in groups:
        rep = min(g, key=lambda pr: (len(pr.program.ops()), str(pr.program)))
        cids = [claim_of[id(pr.program)] for pr in g]
        if rep.errors:  # crashes on probe inputs: prediction-error evidence against robustness
            for cid in cids:
                ctx.ws.claims[cid].ledger.add(f"simulate:{cid}", 0.0, 0.2 * rep.errors / len(inputs) * ctx.cfg.epistemics.kappa)
        out.append({"program": rep.program, "claims": cids, "errors": rep.errors, "size": len(rep.program.ops())})
    return OpResult(outputs={"groups": out}, value_signal=1.0)


def op_py_check(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    """Deterministic articulator check: the Python rendering must pass the same examples (07 §5.2)."""
    checked = []
    for g in inp.get("groups") or []:
        source = dsl.to_python(g["program"])
        res = ctx.body.call("tests", tests=node.params["examples"], source=source)
        for cid in g["claims"]:
            ctx.add_test_evidence(cid, res.ok, root=f"test:python:{cid}")
        checked.append({**g, "python": source, "python_ok": res.ok})
    # the check contributes evidence (a failing rendering lowers belief); it never blocks the answer
    return OpResult(ok=True, outputs={"groups": checked}, value_signal=float(any(g["python_ok"] for g in checked)))


def op_answer_dsl(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    hyps = []
    for i, g in enumerate(inp.get("groups") or []):
        h = HypothesisEval(hyp_id=f"p{i}", binding=Value("code_ref", json.dumps(g["program"].to_json())))
        for cid in g["claims"]:
            h.claim_ids.append(cid)
            h.ledger = h.ledger.merge(ctx.ws.claims[cid].ledger)
            h.grounded = h.grounded or ctx.grounded(cid)
            h.tainted = all(ctx.ws.claims[c].tainted for c in g["claims"])
        h.description_length = float(g["size"])
        if h.grounded:
            h.knowledge_state = EpistemicState.TESTED
        hyps.append(h)
        ctx.artifacts.setdefault("programs", []).append(
            {"dsl": str(g["program"]), "json": g["program"].to_json(), "python": g.get("python") or dsl.to_python(g["program"])}
        )
    ctx.hypotheses = hyps
    return OpResult(outputs={"n_hypotheses": len(hyps)}, value_signal=1.0)


# =================================================================================================
# math word problems: MATCH_PROCEDURE → BIND_QUANTITIES → EVAL_FORMULA → CHECK → ANSWER_VALUE
# =================================================================================================
def op_match_procedure(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    text = normalize_surface(node.params["text"])
    best, best_score = None, 0
    for rid in ctx.memory.find_procedures(node.params["domain"]):
        payload: ProcedurePayload = ctx.memory.record(rid).payload
        score = sum(1 for cue in payload.cues if normalize_surface(cue) in text)
        if score > best_score:
            best, best_score = rid, score
    if best is None or best_score < ctx.cfg.core.procedure_min_cues:
        return OpResult(ok=False, note="no stored procedure applies")
    payload = ctx.memory.record(best).payload
    claim = ctx.claim_from_record(best, payload.procedure_id, "procedure", Value("text", payload.steps[0]["op"]["formula"]),
                                  producer="MATCH_PROCEDURE")
    return OpResult(outputs={"procedure": payload, "claim": claim.claim_id, "score": best_score}, claims=[claim.claim_id])


_NUM = r"(-?\d+(?:\.\d+)?)"


def op_bind_quantities(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    payload: ProcedurePayload = inp["procedure"]
    op = payload.steps[0]["op"]
    text = node.params["text"]
    bindings, claims = {}, []
    for name, pattern in op["bind"].items():
        m = re.search(pattern, text)
        if m is None:
            return OpResult(ok=False, note=f"could not bind quantity {name}")
        val = float(m.group(1))
        bindings[name] = val
        c = ctx.observed_claim(ctx.goal.goal_id, f"quantity:{name}", Value("float", val))
        claims.append(c.claim_id)
    # quantities are joined into one bindings claim so no claim exceeds A premises
    groups = [claims[i : i + ctx.cfg.workspace.A] for i in range(0, len(claims), ctx.cfg.workspace.A)]
    joined = []
    for g in groups:
        joined.append(ctx.derived_claim(ctx.goal.goal_id, "bindings", Value("text", json.dumps(bindings)), tuple(g), "BIND_QUANTITIES").claim_id)
    while len(joined) > 1:
        joined = [ctx.derived_claim(ctx.goal.goal_id, "bindings", Value("text", json.dumps(bindings)),
                                    tuple(joined[:ctx.cfg.workspace.A]), "BIND_QUANTITIES").claim_id] + joined[ctx.cfg.workspace.A:]
    return OpResult(outputs={"bindings": bindings, "claim": joined[0]}, claims=claims + joined)


def op_eval_formula(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    payload: ProcedurePayload = inp["procedure"]
    formula = payload.steps[0]["op"]["formula"]
    value = float(sp.sympify(formula).subs(inp["bindings"]))
    unit = payload.steps[0]["op"].get("unit") or None
    v = Value("int" if value == int(value) else "float", int(value) if value == int(value) else value, unit)
    c = ctx.derived_claim(ctx.goal.goal_id, "answer", v, (inp["procedure_claim"], inp["bindings_claim"]), "EVAL_FORMULA")
    return OpResult(outputs={"value": v, "claim": c.claim_id, "formula": formula}, claims=[c.claim_id])


def op_check_math(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    res = ctx.body.call("sympy", op="evaluate", expr=inp["formula"], subs=inp["bindings"])
    ok = res.ok and abs(float(res.value) - float(inp["value"].value)) <= 1e-9 * max(1.0, abs(float(res.value)))
    ctx.add_test_evidence(inp["claim"], ok, root=f"test:sympy:{inp['claim']}")
    return OpResult(ok=ok, outputs={"claim": inp["claim"], "value": inp["value"]})


def op_algebra(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    res = ctx.body.call("sympy", op=node.params["algebra_op"], expr=node.params["expr"])
    if not res.ok:
        return OpResult(ok=False, note=res.error or "sympy failed")
    c = ctx.observed_claim(ctx.goal.goal_id, "answer", Value("expr_ref", res.value), root=f"tool:sympy:{node.node_id}",
                           trust_class="tool_execution", producer="ALGEBRA")
    eq = ctx.body.call("sympy", op="equivalent", expr=node.params["expr"], other=res.value)
    ctx.add_test_evidence(c.claim_id, bool(eq.ok and eq.value), root=f"test:equiv:{c.claim_id}")
    return OpResult(outputs={"value": c.object, "claim": c.claim_id}, claims=[c.claim_id])


def op_answer_value(ctx: Any, node: CircuitNode, inp: dict[str, Any]) -> OpResult:
    cid = inp["claim"]
    c = ctx.ws.claims[cid]
    h = HypothesisEval(hyp_id="h0", binding=c.object, claim_ids=[cid], ledger=c.ledger.merge(type(c.ledger)()),
                       grounded=ctx.grounded(cid), tainted=c.tainted, knowledge_state=c.state)
    ctx.hypotheses = [h]
    return OpResult(outputs={"n_hypotheses": 1}, value_signal=1.0)


OPS: dict[str, OpSpec] = {
    "LINK": OpSpec(op_link, "retrieve", "LINK"),
    "JOIN_HOP": OpSpec(op_join_hop, "retrieve", "JOIN_HOP"),
    "ANSWER_FACT": OpSpec(op_answer_fact, "answer", "ANSWER"),
    "SYNTH_DSL": OpSpec(op_synth_dsl, "synthesis", "SYNTH_DSL_EVAL", incremental=True),
    "TEST_DSL": OpSpec(op_test_dsl, "test", "TEST"),
    "SIMULATE_DSL": OpSpec(op_simulate_dsl, "simulate", "SIMULATE"),
    "PY_CHECK": OpSpec(op_py_check, "test", "PY_CHECK"),
    "ANSWER_DSL": OpSpec(op_answer_dsl, "answer", "ANSWER"),
    "MATCH_PROCEDURE": OpSpec(op_match_procedure, "retrieve", "MATCH_PROCEDURE"),
    "BIND_QUANTITIES": OpSpec(op_bind_quantities, "transform", "BIND_QUANTITIES"),
    "EVAL_FORMULA": OpSpec(op_eval_formula, "transform", "EVAL_FORMULA"),
    "CHECK_MATH": OpSpec(op_check_math, "test", "CHECK"),
    "ALGEBRA": OpSpec(op_algebra, "transform", "ALGEBRA"),
    "ANSWER_VALUE": OpSpec(op_answer_value, "answer", "ANSWER"),
}
