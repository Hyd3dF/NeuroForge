"""Internal SceneGraph (03 §5 ``SceneGraph``): referent nodes with typed property values and
relations.  Converts from/to the SEF ``Scene`` structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from srm.data.sef import Scene, SceneNode, SceneRelation, TypedValue
from srm.interface.values import Value


@dataclass
class SNode:
    ref: str
    type: str | None = None
    props: dict[str, Value] = field(default_factory=dict)
    dense: np.ndarray | None = None


@dataclass
class SceneGraph:
    nodes: dict[str, SNode] = field(default_factory=dict)
    relations: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    root: str | None = None

    # --- conversion ---------------------------------------------------------------------------
    @staticmethod
    def from_sef(scene: Scene | dict[str, Any]) -> "SceneGraph":
        if isinstance(scene, dict):
            scene = Scene.model_validate(scene)
        g = SceneGraph(root=scene.root)
        for n in scene.nodes:
            g.nodes[n.ref] = SNode(n.ref, n.type, {pid: _value(v) for pid, v in n.properties.items()})
        g.relations = [(r.relation, tuple(r.args)) for r in scene.relations]
        return g

    def to_sef(self) -> Scene:
        return Scene(
            nodes=[SceneNode(ref=n.ref, type=n.type, properties={p: TypedValue(**v.to_dict()) for p, v in n.props.items()})
                   for n in self.nodes.values()],
            relations=[SceneRelation(relation=r, args=list(a)) for r, a in self.relations],
            root=self.root,
        )

    # --- views ---------------------------------------------------------------------------------
    @property
    def root_node(self) -> SNode:
        if self.root is None:
            raise ValueError("scene has no root")
        return self.nodes[self.root]

    def neighbors(self, ref: str) -> list[tuple[str, str, int]]:
        """(relation, other ref, position of ``ref`` in the relation)."""
        out = []
        for rel, args in self.relations:
            if ref in args:
                pos = args.index(ref)
                for other in args:
                    if other != ref:
                        out.append((rel, other, pos))
        return out

    def parts(self, ref: str | None = None) -> list[SNode]:
        ref = ref or self.root
        return [self.nodes[a[0]] for r, a in self.relations if r == "rel:part_of" and len(a) == 2 and a[1] == ref]

    def context_types(self) -> list[str]:
        return [n.type for n in self.nodes.values() if n.type and n.type.startswith("ctx:")]

    def signature(self) -> list[tuple[str, str, str]]:
        """Content-free relational pattern: (relation, type of arg0, type of arg1) (03 §2.7)."""
        def t(ref: str) -> str:
            node = self.nodes.get(ref)
            return (node.type or "?") if node is not None else "?"
        return sorted((r, t(a[0]), t(a[1]) if len(a) > 1 else "") for r, a in self.relations)


def _value(v: TypedValue) -> Value:
    val = v.value
    if isinstance(val, list):
        val = tuple(val)
    return Value(v.type, val, v.unit, float(v.tolerance or 0.0))
