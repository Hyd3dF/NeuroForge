"""Composer (05 §5): builds a temporary circuit graph per goal and repairs it locally.

Construction order (05 §5.2): stored procedure schema → analogical transfer → type-directed
synthesis.  In F0 the strategies per request kind are:

* factual — a QUERY_JOIN chain (LINK → JOIN_HOP per atom → ANSWER_FACT)
* dsl/code — retrieval of stored programs as library components, then synthesis
  (SYNTH_DSL → TEST_DSL → SIMULATE_DSL → [PY_CHECK] → ANSWER_DSL)
* math word — stored procedure schema (MATCH_PROCEDURE → BIND_QUANTITIES → EVAL_FORMULA → CHECK_MATH)
* algebra — a Body tool circuit (ALGEBRA → ANSWER_VALUE)

The learned policy ``π_θ`` (S4) will rank synthesis expansions; F0 uses uniform priors.
"""

from __future__ import annotations

from typing import Any

from srm.core.circuit import CircuitGraph, CircuitNode, NodeState


class Composer:
    def __init__(self, config: Any) -> None:
        self.cfg = config

    def build(self, ctx: Any) -> CircuitGraph:
        req = ctx.request
        g = CircuitGraph(ctx.goal.goal_id, "direct")
        if req.kind == "factual":
            atoms = ctx.goal.atoms
            g.origin = "schema"
            link = g.add("LINK", surface=atoms[0].subject)
            prev = f"{link}.frontier"
            for atom in atoms:
                hop = g.add("JOIN_HOP", {"frontier": prev}, atom=atom)
                prev = f"{hop}.frontier"
            g.add("ANSWER_FACT", {"frontier": prev}, var=ctx.goal.answer_vars[0][0])
        elif req.kind in ("dsl_task", "code_task"):
            g.origin = "synthesized"
            synth = g.add("SYNTH_DSL", examples=req.examples, ret_type=req.ret_type)
            test = g.add("TEST_DSL", {"candidates": f"{synth}.candidates"}, examples=req.examples)
            sim = g.add("SIMULATE_DSL", {"passing": f"{test}.passing"}, examples=req.examples)
            last = sim
            if req.output_format == "python":
                last = g.add("PY_CHECK", {"groups": f"{sim}.groups"}, examples=req.examples)
            g.add("ANSWER_DSL", {"groups": f"{last}.groups"})
        elif req.kind == "math_word":
            g.origin = "schema"
            m = g.add("MATCH_PROCEDURE", text=req.text, domain="math/word")
            b = g.add("BIND_QUANTITIES", {"procedure": f"{m}.procedure"}, text=req.text)
            e = g.add("EVAL_FORMULA", {"procedure": f"{m}.procedure", "bindings": f"{b}.bindings",
                                       "procedure_claim": f"{m}.claim", "bindings_claim": f"{b}.claim"})
            c = g.add("CHECK_MATH", {"formula": f"{e}.formula", "bindings": f"{b}.bindings", "value": f"{e}.value",
                                     "claim": f"{e}.claim"})
            g.add("ANSWER_VALUE", {"claim": f"{c}.claim"})
        elif req.kind == "algebra":
            a = g.add("ALGEBRA", algebra_op=req.algebra_op, expr=req.expr)
            g.add("ANSWER_VALUE", {"claim": f"{a}.claim"})
        return g

    def repair(self, ctx: Any, graph: CircuitGraph, failed: CircuitNode) -> bool:
        """Local repair (05 §5.2 step 5): re-synthesize the failed subgraph with a wider search."""
        if failed.op == "SYNTH_DSL" and failed.attempts < 2:
            size = failed.params.get("max_size", self.cfg.core.synth_max_size)
            if size < self.cfg.core.synth_max_size + 1:
                failed.params["max_size"] = size + 1
                failed.outputs.pop("_synth", None)
                failed.state = NodeState.PENDING
                for n in graph.nodes.values():  # downstream nodes become pending again
                    if n.state == NodeState.SKIPPED:
                        n.state = NodeState.PENDING
                ctx.ws.log("repair", node=failed.node_id, max_size=size + 1)
                return True
        return False
