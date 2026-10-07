"""Selector (05 §7): proposes actions with value-of-computation bids; decides when to stop.

Before S4 trains the actor-critic, the heuristic
``VOC_h(a) = u(target)·(1 + centrality)·(0.5 + stakes)·type_prior(a) − λ_cost·cost(a)``
is used (it remains the fallback outside the learned head's envelope).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from srm.core.circuit import CircuitGraph, CircuitNode
from srm.core.ops import OPS


@dataclass
class Action:
    node: CircuitNode
    voc: float
    flops: float


class Selector:
    def __init__(self, config: Any) -> None:
        self.cfg = config
        self.td_errors: list[float] = []
        self._last_value: float | None = None

    def goal_uncertainty(self, ctx: Any) -> float:
        """``U_goal``: 1 − best belief among hypotheses (1.0 before any hypothesis exists)."""
        hs = [h for h in ctx.hypotheses if not h.is_residual]
        if not hs:
            return 1.0
        from srm.core.evidence import belief

        return 1.0 - max(belief(h.ledger.e_plus, h.ledger.e_minus, self.cfg.epistemics.W) for h in hs)

    def propose(self, ctx: Any, graph: CircuitGraph) -> list[Action]:
        u = self.goal_uncertainty(ctx)
        actions = []
        n_nodes = max(1, len(graph.nodes))
        for node in graph.ready():
            spec = OPS[node.op]
            flops = self.cfg.control.op_flops.get(spec.cost_key, 1e5)
            if spec.incremental:
                flops *= self.cfg.core.synth_evals_per_beat
            downstream = sum(1 for n in graph.nodes.values() if any(r.startswith(node.node_id + ".") for r in n.inputs.values()))
            centrality = downstream / n_nodes
            prior = self.cfg.control.type_prior.get(spec.kind, 0.5)
            cost = flops * self.cfg.control.prices["flops"]
            voc = max(u, 0.05) * (1.0 + centrality) * (0.5 + ctx.stakes) * prior - self.cfg.core.lambda_cost * cost
            actions.append(Action(node, voc, flops))
        return actions

    def should_stop(self, ctx: Any, graph: CircuitGraph, actions: list[Action]) -> bool:
        if graph.finished():
            return True
        return bool(actions) and max(a.voc for a in actions) < 0.0

    def td_update(self, ctx: Any, success_estimate: float, reward: float = 0.0, gamma: float = 0.95) -> float:
        """TD error ``δ = r + γV(s') − V(s)`` with V = 1 − U_goal; this is the DA signal (07 §3)."""
        value = success_estimate
        prev = self._last_value if self._last_value is not None else value
        delta = reward + gamma * value - prev
        self._last_value = value
        self.td_errors.append(delta)
        return delta
