"""Append-only registries: roles, properties, relations, concept types, value types (03 §3–4).

After the interface freeze (03 §1.4) existing entries are immutable; new entries may
be appended (a MINOR interface version) and are marked ``provisional``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from srm.interface import codes as C
from srm.interface.codespace import CodeSpace, seeded_dense
from srm.util.canonical import canonical_hash


@dataclass
class RegistryEntry:
    id: str
    name: str
    identity_code: np.ndarray
    dense: np.ndarray
    content_code: np.ndarray
    aliases: tuple[str, ...] = ()
    provisional: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """Content used for registry digests (codes are derived, so only symbols and meta)."""
        return {"id": self.id, "name": self.name, "aliases": list(self.aliases), "meta": self.meta}


class Registry:
    """A typed, append-only symbol table with identity and content codes."""

    kind = "generic"

    def __init__(self, codespace: CodeSpace) -> None:
        self.cs = codespace
        self._entries: dict[str, RegistryEntry] = {}
        self._order: list[str] = []
        self._by_name: dict[str, str] = {}
        self.frozen_count: int | None = None

    # --- mutation -------------------------------------------------------------------------
    def add(
        self,
        id: str,
        name: str | None = None,
        meta: dict[str, Any] | None = None,
        aliases: Iterable[str] = (),
        dense: np.ndarray | None = None,
    ) -> RegistryEntry:
        meta = dict(meta or {})
        name = name or id
        if id in self._entries:
            existing = self._entries[id]
            if existing.meta != meta or existing.name != name:
                raise ValueError(f"{self.kind} registry: entry {id!r} is immutable and differs from the new definition")
            return existing
        vec = seeded_dense(id, self.cs.d, salt=self.kind) if dense is None else np.asarray(dense, dtype=np.float32)
        entry = RegistryEntry(
            id=id,
            name=name,
            identity_code=C.code_from_key(id, self.cs.B, self.cs.L, salt=self.kind),
            dense=vec,
            content_code=self.cs.project(vec),
            aliases=tuple(aliases),
            provisional=self.frozen_count is not None,
            meta=meta,
        )
        self._entries[id] = entry
        self._order.append(id)
        for key in (name, *entry.aliases):
            self._by_name.setdefault(_norm(key), id)
        return entry

    def freeze(self) -> None:
        self.frozen_count = len(self._order)

    # --- lookup ---------------------------------------------------------------------------
    def __contains__(self, id: str) -> bool:
        return id in self._entries

    def __getitem__(self, id: str) -> RegistryEntry:
        return self._entries[id]

    def get(self, id: str) -> RegistryEntry | None:
        return self._entries.get(id)

    def by_name(self, name: str) -> RegistryEntry | None:
        rid = self._by_name.get(_norm(name))
        return self._entries[rid] if rid else None

    def __len__(self) -> int:
        return len(self._order)

    def ids(self) -> list[str]:
        return list(self._order)

    def entries(self) -> list[RegistryEntry]:
        return [self._entries[i] for i in self._order]

    # --- digests (03 §1.3) ----------------------------------------------------------------
    def digest(self, baseline_only: bool = False) -> str:
        ids = self._order[: self.frozen_count] if (baseline_only and self.frozen_count is not None) else self._order
        return canonical_hash([self._entries[i].describe() for i in ids])


def _norm(s: str) -> str:
    return " ".join(s.casefold().split())


class RoleRegistry(Registry):
    kind = "role"

    BASE_ROLES = (
        "subject", "object", "agent", "patient", "instrument", "part", "whole", "property", "value",
        "function", "cause", "effect", "condition", "before", "after", "input", "output", "state",
        "goal", "container", "element", "index", "relation", "type", "parent", "child", "source",
        "target", "time", "location", "quantity", "unit", "operator", "argument", "result", "context",
    )

    def populate(self, n_roles: int) -> None:
        for name in self.BASE_ROLES[:n_roles]:
            self.add(f"role:{name}", name)
        for k in range(len(self.BASE_ROLES), n_roles):
            self.add(f"role:slot{k}", f"slot{k}")


class PropertyRegistry(Registry):
    kind = "property"
    KINDS = ("geometric", "dynamic", "functional", "relational", "computational", "contextual", "other")
    VALUE_TYPES = ("bool", "int", "float", "range", "enum", "entity_ref", "text")

    def add_property(
        self, id: str, name: str, kind: str, value_type: str, unit: str | None = None,
        enum_values: Iterable[str] | None = None, range_: tuple[float, float] | None = None,
        log_scale: bool = False, description: str = "",
    ) -> RegistryEntry:
        if kind not in self.KINDS:
            raise ValueError(f"unknown property kind {kind!r}")
        if value_type not in self.VALUE_TYPES:
            raise ValueError(f"unknown value type {value_type!r}")
        meta = {
            "kind": kind, "value_type": value_type, "unit": unit,
            "enum_values": list(enum_values) if enum_values is not None else None,
            "range": list(range_) if range_ is not None else None,
            "log_scale": log_scale, "description": description,
        }
        return self.add(id, name, meta)


class RelationRegistry(Registry):
    kind = "relation"

    def add_relation(
        self, id: str, name: str, arity: int = 2, arg_types: Iterable[str] = (), cardinality: str = "multi",
        temporal: bool = False, symmetric: bool = False, inverse_of: str | None = None, description: str = "",
    ) -> RegistryEntry:
        if cardinality not in ("functional", "multi"):
            raise ValueError("cardinality must be 'functional' or 'multi'")
        meta = {
            "arity": arity, "arg_types": list(arg_types), "cardinality": cardinality,
            "temporal": temporal, "symmetric": symmetric, "inverse_of": inverse_of,
            "description": description,
        }
        return self.add(id, name, meta)

    def is_functional(self, id: str) -> bool:
        e = self.get(id)
        return bool(e and e.meta.get("cardinality") == "functional")

    def is_temporal(self, id: str) -> bool:
        e = self.get(id)
        return bool(e and e.meta.get("temporal"))


class ConceptTypeRegistry(Registry):
    kind = "concept_type"
    TOP_TYPES = (
        "physical_object", "artifact", "agent", "event", "process", "quantity", "code_construct",
        "math_object", "place", "time", "abstract",
    )

    def populate(self) -> None:
        for t in self.TOP_TYPES:
            self.add(f"type:{t}", t)


class ValueTypeRegistry(Registry):
    kind = "value_type"
    VALUE_TYPES = (
        "int", "float", "bool", "enum", "date", "range", "text", "entity_ref", "code_ref", "expr_ref",
    )

    def populate(self) -> None:
        for t in self.VALUE_TYPES:
            self.add(f"vtype:{t}", t)
