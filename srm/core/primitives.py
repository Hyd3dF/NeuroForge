"""Primitive Basis implementations (05 §4).

Signatures are fixed in :mod:`srm.interface.primitives`.  A-class primitives are exact
algorithms; H-class primitives (COMPARE, ALIGN, EXPLAIN) use scoring functions whose
parameters are learned in bootstrap stage S3 (defaults here are the pre-training
heuristics).  ``IMPLEMENTATIONS`` maps every primitive name to its callable.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np

from srm.core.scene import SceneGraph, SNode
from srm.core.schema import PropertyProfile, Schema
from srm.interface import codes as C
from srm.interface.messages import PatternAtom, Qualifiers, is_var
from srm.interface.primitives import PRIMITIVE_SIGNATURES
from srm.interface.values import Value


# =================================================================================================
# COMPARE (H): structured difference of an instance against a schema (or another instance)
# =================================================================================================
@dataclass
class DiffItem:
    property_id: str
    status: str  # match | mismatch | missing_a | missing_b
    magnitude: float = 0.0
    weight: float = 1.0


@dataclass
class DiffRecord:
    items: list[DiffItem]
    score: float

    def by_status(self, status: str) -> list[DiffItem]:
        return [i for i in self.items if i.status == status]


def _phi(z: float, z_match: float) -> float:
    """Monotone mismatch penalty (calibrated in S3; this is the pre-training default)."""
    return min(3.0, 0.5 + (z - z_match) / z_match)


def compare(instance: dict[str, Value], schema: Schema, z_match: float = 2.5, lambda_miss: float = 0.5,
            global_std: dict[str, float] | None = None, defining_only: bool = False) -> DiffRecord:
    items: list[DiffItem] = []
    score = 0.0
    for pid, prof in schema.profiles.items():
        w = schema.w_def(pid, (global_std or {}).get(pid, 1.0))
        if defining_only and w < 0.5:
            continue
        v = instance.get(pid)
        if v is None:
            items.append(DiffItem(pid, "missing_a", 0.0, w))
            score -= lambda_miss * w
            continue
        if prof.vtype == "scalar":
            z = prof.z(v)
            ok = z <= z_match
            mag = 0.0 if ok else _phi(z, z_match)
        else:
            p = prof.prob(v)
            ok = p >= (0.2 if prof.vtype == "bool" else 0.1)
            mag = 0.0 if ok else 1.0
        items.append(DiffItem(pid, "match" if ok else "mismatch", mag, w))
        score += w if ok else -w * mag
    for pid in instance:
        if pid not in schema.profiles:
            items.append(DiffItem(pid, "missing_b", 0.0, 0.0))
    return DiffRecord(items, score)


def compare_values(a: dict[str, Value], b: dict[str, Value]) -> DiffRecord:
    """Instance-vs-instance comparison (exact/tolerance matching)."""
    items, score = [], 0.0
    for pid in sorted(set(a) | set(b)):
        if pid not in a:
            items.append(DiffItem(pid, "missing_a"))
        elif pid not in b:
            items.append(DiffItem(pid, "missing_b"))
        elif a[pid].matches(b[pid]):
            items.append(DiffItem(pid, "match"))
            score += 1.0
        else:
            items.append(DiffItem(pid, "mismatch", 1.0))
            score -= 1.0
    return DiffRecord(items, score)


# =================================================================================================
# ALIGN (H): structure mapping between two scene graphs (05 §6.2)
# =================================================================================================
@dataclass
class Alignment:
    pairs: dict[str, str]  # source ref → target ref
    score: float
    affinity: np.ndarray
    candidate_inferences: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)  # relations in target terms
    property_inferences: list[tuple[str, str, Value]] = field(default_factory=list)  # (target ref, pid, value)


def node_affinity(a: SNode, b: SNode) -> float:
    """Pre-training affinity: type agreement, shared-property agreement and dense cosine."""
    type_term = 1.0 if (a.type is not None and a.type == b.type) else (0.5 if a.type is None or b.type is None else 0.1)
    shared = set(a.props) & set(b.props)
    union = set(a.props) | set(b.props)
    prop_term = (sum(1.0 for p in shared if a.props[p].matches(b.props[p])) + 0.5 * len(shared)) / max(1, 1.5 * len(union))
    dense_term = 0.0
    if a.dense is not None and b.dense is not None:
        dense_term = float(np.dot(a.dense, b.dense) / (np.linalg.norm(a.dense) * np.linalg.norm(b.dense) + 1e-9))
    return 0.4 * type_term + 0.5 * prop_term + 0.1 * dense_term


def _sinkhorn(A: np.ndarray, iters: int = 10) -> np.ndarray:
    M = np.maximum(A, 1e-9)
    for _ in range(iters):
        M = M / M.sum(axis=1, keepdims=True)
        M = M / M.sum(axis=0, keepdims=True)
    return M


def align(src: SceneGraph, tgt: SceneGraph, t_align: int = 5, lambda_s: float = 0.5, theta_pair: float = 0.3,
          lambda_rel: float = 0.5, lambda_unmatched: float = 0.25,
          affinity: Callable[[SNode, SNode], float] = node_affinity) -> Alignment:
    s_refs, t_refs = list(src.nodes), list(tgt.nodes)
    if not s_refs or not t_refs:
        return Alignment({}, 0.0, np.zeros((len(s_refs), len(t_refs))))
    si = {r: i for i, r in enumerate(s_refs)}
    ti = {r: j for j, r in enumerate(t_refs)}
    A0 = np.array([[affinity(src.nodes[a], tgt.nodes[b]) for b in t_refs] for a in s_refs])
    if src.root in si and tgt.root in ti:
        A0[si[src.root], ti[tgt.root]] += 0.5  # roots correspond by default
    A = A0.copy()
    for _ in range(t_align):  # structural refinement: same relation, mapped arguments
        bonus = np.zeros_like(A)
        for rs, sa in src.relations:
            for rt, ta in tgt.relations:
                if rs != rt or len(sa) != len(ta):
                    continue
                for k, (x, y) in enumerate(zip(sa, ta)):
                    others = [A[si[sa[m]], ti[ta[m]]] for m in range(len(sa)) if m != k and sa[m] in si and ta[m] in ti]
                    if x in si and y in ti and others:
                        bonus[si[x], ti[y]] += float(np.mean(others))
        A = A0 + lambda_s * bonus
    P = _sinkhorn(A)
    # greedy one-to-one assignment on the normalized matrix (D-025: greedy instead of Hungarian in F0)
    pairs: dict[str, str] = {}
    used_t: set[int] = set()
    for flat in np.argsort(-P, axis=None):
        i, j = divmod(int(flat), len(t_refs))
        if s_refs[i] in pairs or j in used_t:
            continue
        if A[i, j] < theta_pair:
            continue
        pairs[s_refs[i]] = t_refs[j]
        used_t.add(j)
    mapped = set(pairs)
    consistent = 0
    t_rel = {(r, a) for r, a in tgt.relations}
    inferences: list[tuple[str, tuple[str, ...]]] = []
    for r, a in src.relations:
        if all(x in mapped for x in a):
            image = tuple(pairs[x] for x in a)
            if (r, image) in t_rel:
                consistent += 1
            else:
                inferences.append((r, image))
    prop_inf = []
    for s_ref, t_ref in pairs.items():
        for pid, v in src.nodes[s_ref].props.items():
            if pid not in tgt.nodes[t_ref].props:
                prop_inf.append((t_ref, pid, v))
    unmatched = len(s_refs) - len(pairs)
    score = sum(float(A[si[s], ti[t]]) for s, t in pairs.items()) + lambda_rel * consistent - lambda_unmatched * unmatched
    return Alignment(pairs, score, A, inferences, prop_inf)


# =================================================================================================
# ABSTRACT (A): anti-unification across aligned instances (05 §6.3); SPECIALIZE (A)
# =================================================================================================
def abstract(instances: list[SceneGraph], schema_id: str = "schema:new", name: str = "") -> Schema:
    """Constants where instances agree, variables (profiles with spread) where they differ."""
    schema = Schema(schema_id, name)
    for g in instances:
        root = g.root_node
        parts = [p.type for p in g.parts() if p.type]
        schema.observe(root.props, parts, g.context_types())
    return schema


def specialize(schema: Schema, bindings: dict[str, Value]) -> tuple[dict[str, Value], set[str]]:
    """Return properties for an instance: given bindings plus typical values marked INHERITED."""
    props = dict(bindings)
    inherited = set()
    for pid, prof in schema.profiles.items():
        if pid not in props:
            props[pid] = prof.typical()
            inherited.add(pid)
    return props, inherited


# =================================================================================================
# UNIFY / QUERY_JOIN (A): patterns over engrams (multi-hop facts)
# =================================================================================================
def unify(atom: PatternAtom, subject: str, relation: str, obj: Value, bindings: dict[str, Any]) -> dict[str, Any] | None:
    if atom.relation != relation:
        return None
    out = dict(bindings)
    for term, val in ((atom.subject, Value("entity_ref", subject)), (atom.object, obj)):
        if is_var(term):
            if term in out:
                if not out[term].matches(val):
                    return None
            else:
                out[term] = val
        else:
            const = term if isinstance(term, Value) else Value("entity_ref", term)
            if not const.matches(val):
                return None
    return out


@dataclass
class JoinResult:
    bindings: list[dict[str, Any]]
    support: list[list[int]]  # record ids used per binding (justification)
    missing: list[tuple[str, str]]  # (subject, relation) hops with no records


def query_join(atoms: list[PatternAtom], lookup: Callable[[str | None, str], list[tuple[int, str, str, Value]]],
               initial: dict[str, Any] | None = None) -> JoinResult:
    """Conjunctive query: atoms are joined left to right on shared variables.

    ``lookup(subject, relation)`` returns ``(record_id, subject, relation, object)`` tuples.
    """
    frontier: list[tuple[dict[str, Any], list[int]]] = [(dict(initial or {}), [])]
    missing: list[tuple[str, str]] = []
    for atom in atoms:
        nxt: list[tuple[dict[str, Any], list[int]]] = []
        for b, used in frontier:
            subj = atom.subject
            if is_var(subj):
                if subj not in b:
                    raise ValueError(f"unbound subject variable {subj}; order atoms so subjects are bound")
                subj = str(b[subj].value)
            rows = lookup(subj, atom.relation)
            if not rows:
                missing.append((subj, atom.relation))
            for rid, s, r, o in rows:
                u = unify(atom, s, r, o, b)
                if u is not None:
                    nxt.append((u, used + [rid]))
        frontier = nxt
    return JoinResult([b for b, _ in frontier], [u for _, u in frontier], missing)


# =================================================================================================
# ORDER / COUNT / AGGREGATE (A)
# =================================================================================================
def order(values: list[Value], descending: bool = False) -> list[Value]:
    return sorted(values, key=lambda v: (float(v.value) if v.is_numeric() else str(v.value)), reverse=descending)


def count(values: list[Value]) -> Value:
    return Value("int", len(values))


def aggregate(values: list[Value], op: str) -> Value:
    nums = [float(v.value) for v in values]
    if not nums:
        raise ValueError("aggregate of empty list")
    fn = {"sum": sum, "min": min, "max": max, "mean": lambda xs: sum(xs) / len(xs)}[op]
    out = fn(nums)
    return Value("int" if out == int(out) and all(v.type == "int" for v in values) else "float", out)


# =================================================================================================
# SEARCH (A): budgeted best-first with nogood masking
# =================================================================================================
def search(start: Iterable[Any], expand: Callable[[Any], Iterable[Any]], priority: Callable[[Any], float],
           is_goal: Callable[[Any], bool], budget: int, nogood: Callable[[Any], bool] = lambda s: False,
           max_solutions: int = 1) -> tuple[list[Any], int]:
    counter = itertools.count()
    heap = [(-priority(s), next(counter), s) for s in start]
    heapq.heapify(heap)
    solutions, expansions = [], 0
    while heap and expansions < budget and len(solutions) < max_solutions:
        _, _, s = heapq.heappop(heap)
        if nogood(s):
            continue
        if is_goal(s):
            solutions.append(s)
            continue
        expansions += 1
        for child in expand(s):
            if not nogood(child):
                heapq.heappush(heap, (-priority(child), next(counter), child))
    return solutions, expansions


# =================================================================================================
# Registry
# =================================================================================================
IMPLEMENTATIONS: dict[str, Callable[..., Any]] = {
    "BIND": C.bind, "UNBIND": C.unbind, "BUNDLE": C.bundle, "PERMUTE": C.permute, "SPARSIFY": C.sparsify,
    "SIM": C.overlap, "COMPARE": compare, "ALIGN": align, "ABSTRACT": abstract, "SPECIALIZE": specialize,
    "ORDER": order, "COUNT": count, "AGGREGATE": aggregate, "SEARCH": search, "UNIFY": unify,
    "QUERY_JOIN": query_join,
}
# Implemented elsewhere (memory / composer / simulator / body):
EXTERNAL = {
    "RETRIEVE": "srm.memory.system.MemorySystem.retrieve",
    "CLEANUP": "srm.memory.system.MemorySystem.cleanup",
    "TRANSFORM": "srm.core.composer (DSL/SymPy operators)",
    "PREDICT_STEP": "srm.core.simulator",
    "REGRESS": "srm.core.composer",
    "DECOMPOSE": "srm.core.composer",
    "EXPLAIN": "srm.prediction.support",
    "TEST": "srm.core.error_monitor / srm.body",
    "ITERATE": "srm.core.composer (control node)",
    "BRANCH": "srm.core.composer (control node)",
}


def check_complete() -> list[str]:
    """Names in the interface signature set without an implementation binding."""
    return [p.name for p in PRIMITIVE_SIGNATURES if p.name not in IMPLEMENTATIONS and p.name not in EXTERNAL]


_ = PropertyProfile  # re-exported for schema construction convenience
