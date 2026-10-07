"""Circuit graphs: the temporary computation circuits built by the Composer (05 §5.1)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class NodeState(str, enum.Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class OpResult:
    done: bool = True
    ok: bool = True
    outputs: dict[str, Any] = field(default_factory=dict)
    claims: list[str] = field(default_factory=list)
    flops: float = 0.0
    note: str = ""
    value_signal: float = 0.0  # realized value reported to the Heart's credit ledger


@dataclass
class CircuitNode:
    node_id: str
    op: str
    inputs: dict[str, str] = field(default_factory=dict)  # port → "node_id.port"
    params: dict[str, Any] = field(default_factory=dict)
    state: NodeState = NodeState.PENDING
    outputs: dict[str, Any] = field(default_factory=dict)
    produced: list[str] = field(default_factory=list)
    attempts: int = 0
    flops_spent: float = 0.0
    note: str = ""


@dataclass
class CircuitGraph:
    goal_id: str
    origin: str  # schema | analogy | synthesized | direct
    nodes: dict[str, CircuitNode] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)

    def add(self, op: str, inputs: dict[str, str] | None = None, **params: Any) -> str:
        nid = f"n{len(self.order)}:{op}"
        self.nodes[nid] = CircuitNode(nid, op, dict(inputs or {}), params)
        self.order.append(nid)
        return nid

    def _source_state(self, ref: str) -> NodeState:
        return self.nodes[ref.split(".", 1)[0]].state

    def ready(self) -> list[CircuitNode]:
        out = []
        for nid in self.order:
            n = self.nodes[nid]
            if n.state == NodeState.RUNNING:
                out.append(n)
            elif n.state in (NodeState.PENDING, NodeState.READY):
                states = [self._source_state(r) for r in n.inputs.values()]
                if any(s in (NodeState.FAILED, NodeState.SKIPPED) for s in states):
                    n.state = NodeState.SKIPPED
                    n.note = "upstream failed"
                elif all(s == NodeState.DONE for s in states):
                    n.state = NodeState.READY
                    out.append(n)
        return out

    def resolve(self, node: CircuitNode) -> dict[str, Any]:
        out = {}
        for port, ref in node.inputs.items():
            src_id, src_port = ref.split(".", 1)
            out[port] = self.nodes[src_id].outputs.get(src_port)
        return out

    def finished(self) -> bool:
        return all(n.state in (NodeState.DONE, NodeState.FAILED, NodeState.SKIPPED) for n in self.nodes.values())

    def failed(self) -> list[CircuitNode]:
        return [n for n in self.nodes.values() if n.state in (NodeState.FAILED, NodeState.SKIPPED)]

    def trace(self) -> list[dict[str, Any]]:
        return [
            {"node": n.node_id, "op": n.op, "state": n.state.value, "attempts": n.attempts,
             "flops": n.flops_spent, "produced": list(n.produced), "note": n.note}
            for n in (self.nodes[i] for i in self.order)
        ]
