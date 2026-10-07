"""ObjectWorld generator (02 §14.1).

A concept hierarchy over a ground-truth Property Basis.  Concepts have defining
properties (narrow variability), typical properties (wide variability), parts,
functions and invariances.  Instances are sampled from leaf concepts and placed in
context scenes.  Held-out splits: whole leaf concepts (few-shot), held-out property
compositions, and wheel-style out-of-range instances.  Definitions and instances
are rendered into controlled English for parser training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from srm.data.generators.common import GenContext
from srm.data.io import Dataset
from srm.data.sef import (
    ConceptDefinitionRecord,
    ConceptExampleRecord,
    Invariance,
    PartSpec,
    PropertyConstraint,
    PropertyDefRecord,
    Scene,
    SceneNode,
    SceneRelation,
    SegmentRecord,
    Span,
    TaskRecord,
    TypedValue,
    Verifier,
)

VERSION = "0.1"
CATEGORY = "synthetic/objectworld"
KINDS = ("geometric", "dynamic", "functional", "relational", "computational", "contextual")
UNITS = {"geometric": "m", "dynamic": "m/s", "relational": None, "computational": None, "contextual": None, "functional": None}
TRANSFORMS = ("scale", "color", "material", "rotation")


@dataclass
class Prop:
    pid: str
    name: str
    kind: str
    vtype: str  # float | bool | enum
    unit: str | None = None
    lo: float = 0.0  # log10 bounds for float
    hi: float = 1.0
    enum_values: tuple[str, ...] = ()


@dataclass
class Constraint:
    """Generator-internal distribution of a property within a concept."""

    center: float = 0.0  # log10 for float
    sigma: float = 0.1
    p_true: float = 0.5
    probs: dict[str, float] = field(default_factory=dict)


@dataclass
class Concept:
    cid: str
    name: str
    parent: str | None
    depth: int
    is_component: bool = False
    defining: dict[str, Constraint] = field(default_factory=dict)
    typical: dict[str, Constraint] = field(default_factory=dict)
    parts: list[tuple[str, str, tuple[int, int]]] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    invariances: dict[str, bool] = field(default_factory=dict)
    contexts: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    heldout: bool = False
    heldout_combo: tuple[str, str] | None = None  # (enum property, excluded value)


class ObjectWorld:
    def __init__(self, ctx: GenContext, n_props: int, n_concepts: int, n_heldout: int, max_depth: int = 4,
                 n_contexts: int = 12) -> None:
        self.ctx = ctx
        rng = ctx.rng
        self.props: dict[str, Prop] = {}
        for i in range(n_props):
            kind = KINDS[i % len(KINDS)]
            name = f"{kind[:3]}_{ctx.names.make((1, 2)).lower()}"
            pid = f"prop:{name}"
            if kind == "functional":
                vtype = "bool"
            else:
                vtype = ("float", "float", "bool", "enum")[int(rng.integers(4))]
            if vtype == "float":
                lo = float(rng.uniform(-2, 2))
                p = Prop(pid, name, kind, "float", UNITS[kind], lo, lo + float(rng.uniform(1.5, 3.0)))
            elif vtype == "enum":
                vals = tuple(ctx.names.make((1, 1)).lower() for _ in range(int(rng.integers(3, 7))))
                p = Prop(pid, name, kind, "enum", enum_values=vals)
            else:
                p = Prop(pid, name, kind, "bool")
            self.props[pid] = p
        self.contexts = [f"ctx:{ctx.names.make((2, 2)).lower()}" for _ in range(n_contexts)]
        self.concepts: dict[str, Concept] = {}
        self._build_tree(n_concepts, max_depth)
        leaves = [c for c in self.concepts.values() if not c.children and not c.is_component]
        order = rng.permutation(len(leaves))
        for idx in order[:n_heldout]:
            leaves[idx].heldout = True
        for idx in order[n_heldout : n_heldout + max(1, len(leaves) // 6)]:
            c = leaves[idx]
            enums = [pid for pid in c.typical if self.props[pid].vtype == "enum"]
            if enums:
                pid = enums[0]
                c.heldout_combo = (pid, self.props[pid].enum_values[-1])

    # --- tree -------------------------------------------------------------------------------
    def _constraint(self, prop: Prop, defining: bool) -> Constraint:
        rng = self.ctx.rng
        if prop.vtype == "float":
            return Constraint(
                center=float(rng.uniform(prop.lo + 0.3, prop.hi - 0.3)),
                sigma=float(rng.uniform(0.04, 0.10)) if defining else float(rng.uniform(0.25, 0.45)),
            )
        if prop.vtype == "bool":
            p = 0.97 if rng.random() < 0.6 else 0.03
            return Constraint(p_true=p if defining else float(rng.uniform(0.55, 0.8)))
        vals = list(prop.enum_values)
        if defining:
            main = vals[int(rng.integers(len(vals)))]
            probs = {v: (0.92 if v == main else 0.08 / (len(vals) - 1)) for v in vals}
        else:
            w = rng.dirichlet(np.ones(len(vals)) * 2.0)
            probs = {v: float(x) for v, x in zip(vals, w)}
        return Constraint(probs=probs)

    def _ancestry_props(self, c: Concept) -> set[str]:
        out: set[str] = set()
        cur: Concept | None = c
        while cur is not None:
            out |= set(cur.defining) | set(cur.typical)
            cur = self.concepts.get(cur.parent) if cur.parent else None
        return out

    def _new_concept(self, parent: Concept | None, is_component: bool) -> Concept:
        rng = self.ctx.rng
        name = self.ctx.names.make((2, 3)).lower()
        c = Concept(f"concept:{name}", name, parent.cid if parent else None,
                    (parent.depth + 1) if parent else 0, is_component=is_component)
        used = self._ancestry_props(parent) if parent else set()
        free = [pid for pid in self.props if pid not in used]
        rng.shuffle(free)
        n_def = int(rng.integers(2, 4))
        n_typ = int(rng.integers(2, 5)) if parent is not None else 1
        for pid in free[:n_def]:
            c.defining[pid] = self._constraint(self.props[pid], True)
            if self.props[pid].kind == "functional" and c.defining[pid].p_true > 0.5:
                c.functions.append(pid)
        for pid in free[n_def : n_def + n_typ]:
            c.typical[pid] = self._constraint(self.props[pid], False)
        if parent is None:
            c.invariances = {t: bool(rng.random() < 0.7) for t in TRANSFORMS}
            c.invariances["scale"] = True
        else:
            c.invariances = dict(parent.invariances)
        if not is_component:
            c.contexts = [self.ctx.choice(self.contexts) for _ in range(int(rng.integers(1, 3)))]
        self.concepts[c.cid] = c
        if parent:
            parent.children.append(c.cid)
        return c

    def _build_tree(self, n_concepts: int, max_depth: int) -> None:
        rng = self.ctx.rng
        n_component = max(4, n_concepts // 8)
        comp_root = self._new_concept(None, True)
        frontier = [comp_root]
        while len([c for c in self.concepts.values() if c.is_component]) < n_component and frontier:
            p = frontier.pop(0)
            for _ in range(int(rng.integers(2, 4))):
                frontier.append(self._new_concept(p, True))
        component_leaves = [c.cid for c in self.concepts.values() if c.is_component and not c.children]
        roots = [self._new_concept(None, False) for _ in range(3)]
        frontier = list(roots)
        while frontier and len(self.concepts) < n_concepts:
            p = frontier.pop(0)
            if p.depth >= max_depth - 1:
                continue
            for _ in range(int(rng.integers(3, 7))):
                if len(self.concepts) >= n_concepts:
                    break
                child = self._new_concept(p, False)
                frontier.append(child)
        for c in self.concepts.values():
            if c.is_component:
                continue
            inherited = list(self.concepts[c.parent].parts) if c.parent else []
            c.parts = inherited
            if rng.random() < 0.6 and component_leaves:
                lo = int(rng.integers(1, 4))
                c.parts = inherited + [("role:part", self.ctx.choice(component_leaves), (lo, lo + int(rng.integers(0, 3))))]

    # --- distributions -----------------------------------------------------------------------
    def all_constraints(self, cid: str) -> dict[str, tuple[Constraint, bool]]:
        """Property → (constraint, is_defining), child constraints override ancestors."""
        chain = []
        cur: Concept | None = self.concepts[cid]
        while cur is not None:
            chain.append(cur)
            cur = self.concepts.get(cur.parent) if cur.parent else None
        out: dict[str, tuple[Constraint, bool]] = {}
        for c in reversed(chain):
            for pid, con in c.typical.items():
                out[pid] = (con, False)
            for pid, con in c.defining.items():
                out[pid] = (con, True)
        return out

    def sample_value(self, prop: Prop, con: Constraint, shift_sigmas: float = 0.0,
                     exclude: str | None = None, force: str | None = None) -> TypedValue:
        rng = self.ctx.rng
        if prop.vtype == "float":
            x = con.center + con.sigma * (float(rng.standard_normal()) + shift_sigmas)
            return TypedValue(type="float", value=round(10.0 ** x, 4), unit=prop.unit)
        if prop.vtype == "bool":
            return TypedValue(type="bool", value=bool(rng.random() < con.p_true))
        if force is not None:
            return TypedValue(type="enum", value=force)
        vals = [v for v in con.probs if v != exclude]
        p = np.array([con.probs[v] for v in vals])
        return TypedValue(type="enum", value=vals[int(rng.choice(len(vals), p=p / p.sum()))])

    def sample_instance(self, cid: str, ref: str = "x", combo_mode: str = "exclude",
                        out_of_range: str | None = None, context: str | None = None) -> Scene:
        c = self.concepts[cid]
        nodes: list[SceneNode] = []
        rels: list[SceneRelation] = []
        props: dict[str, TypedValue] = {}
        for pid, (con, _def) in self.all_constraints(cid).items():
            prop = self.props[pid]
            exclude = force = None
            if c.heldout_combo and pid == c.heldout_combo[0]:
                if combo_mode == "exclude":
                    exclude = c.heldout_combo[1]
                else:
                    force = c.heldout_combo[1]
            shift = 6.0 if out_of_range == pid else 0.0
            props[pid] = self.sample_value(prop, con, shift, exclude, force)
        nodes.append(SceneNode(ref=ref, type=None, properties=props))
        k = 0
        for role, part_cid, (lo, hi) in c.parts:
            for _ in range(int(self.ctx.rng.integers(lo, hi + 1))):
                pref = f"{ref}.p{k}"
                k += 1
                pprops = {
                    pid: self.sample_value(self.props[pid], con)
                    for pid, (con, d) in self.all_constraints(part_cid).items() if d
                }
                nodes.append(SceneNode(ref=pref, type=part_cid, properties=pprops))
                rels.append(SceneRelation(relation="rel:part_of", args=[pref, ref]))
        ctx_id = context or (self.ctx.choice(c.contexts) if c.contexts else None)
        if ctx_id:
            nodes.append(SceneNode(ref=f"{ref}.ctx", type=ctx_id))
            rels.append(SceneRelation(relation="rel:in_context", args=[ref, f"{ref}.ctx"]))
        return Scene(nodes=nodes, relations=rels, root=ref)

    # --- SEF ---------------------------------------------------------------------------------
    def to_constraint(self, prop: Prop, con: Constraint, defining: bool) -> TypedValue:
        if prop.vtype == "float":
            k = 2.0 if defining else 1.5
            lo, hi = 10 ** (con.center - k * con.sigma), 10 ** (con.center + k * con.sigma)
            return TypedValue(type="range", value=[round(lo, 4), round(hi, 4)], unit=prop.unit)
        if prop.vtype == "bool":
            return TypedValue(type="bool", value=con.p_true > 0.5)
        keep = sorted(v for v, p in con.probs.items() if p >= (0.5 if defining else 0.15))
        return TypedValue(type="enum", value=keep or sorted(con.probs))

    def definition(self, c: Concept, source_id: str) -> ConceptDefinitionRecord:
        siblings = [s for s in (self.concepts[c.parent].children if c.parent else []) if s != c.cid]
        return ConceptDefinitionRecord(
            record_id=self.ctx.rid("concept_definition"), source_id=source_id, data_category=CATEGORY,
            extraction=self.ctx.extraction(), concept_id=c.cid, name=c.name,
            parent_ids=[c.parent] if c.parent else [],
            defining_properties=[
                PropertyConstraint(property_id=pid, constraint=self.to_constraint(self.props[pid], con, True))
                for pid, con in c.defining.items()
            ],
            typical_properties=[
                PropertyConstraint(property_id=pid, constraint=self.to_constraint(self.props[pid], con, False))
                for pid, con in c.typical.items()
            ],
            parts=[PartSpec(role=r, concept_id=p, count=cnt) for r, p, cnt in c.parts],
            functions=list(c.functions),
            invariances=[Invariance(transform=t, preserves_identity=v) for t, v in sorted(c.invariances.items())],
            contrast_with=siblings,
        )

    # --- rendering (controlled English) ------------------------------------------------------
    def render_value(self, prop: Prop, v: TypedValue) -> str:
        if v.type == "range":
            return f"between {v.value[0]:g} and {v.value[1]:g}{' ' + prop.unit if prop.unit else ''}"
        if v.type == "float":
            return f"{v.value:g}{' ' + prop.unit if prop.unit else ''}"
        if v.type == "enum" and isinstance(v.value, list):
            return " or ".join(v.value)
        return str(v.value).lower()

    def render_definition(self, rec: ConceptDefinitionRecord) -> str:
        rng = self.ctx.rng
        parent = self.concepts[rec.parent_ids[0]].name if rec.parent_ids else "thing"
        opening = (
            f"A {rec.name} is a kind of {parent}.",
            f"Every {rec.name} is a {parent}.",
            f"The {rec.name} is a type of {parent}.",
        )[int(rng.integers(3))]
        parts = [opening]
        for pc in rec.defining_properties:
            prop = self.props[pc.property_id]
            if prop.vtype == "bool":
                parts.append(f"It {'always' if pc.constraint.value else 'never'} has {prop.name}.")
            else:
                parts.append(f"Its {prop.name} is {self.render_value(prop, pc.constraint)}.")
        for pc in rec.typical_properties:
            prop = self.props[pc.property_id]
            if prop.vtype != "bool":
                parts.append(f"Its {prop.name} is usually {self.render_value(prop, pc.constraint)}.")
        for ps in rec.parts:
            parts.append(f"It has {ps.count[0]} to {ps.count[1]} {self.concepts[ps.concept_id].name} parts.")
        return " ".join(parts)

    def render_scene(self, scene: Scene, label_name: str | None) -> str:
        root = next(n for n in scene.nodes if n.ref == scene.root)
        head = f"This is a {label_name}." if label_name else "Here is an object."
        bits = [head]
        for pid, v in root.properties.items():
            prop = self.props[pid]
            if prop.vtype == "bool":
                bits.append(f"It {'has' if v.value else 'lacks'} {prop.name}.")
            else:
                bits.append(f"Its {prop.name} is {self.render_value(prop, v)}.")
        n_parts = sum(1 for r in scene.relations if r.relation == "rel:part_of")
        if n_parts:
            bits.append(f"It has {n_parts} parts.")
        ctxn = next((n for n in scene.nodes if n.type and n.type.startswith("ctx:")), None)
        if ctxn:
            bits.append(f"It is found in the {ctxn.type.split(':', 1)[1]}.")
        return " ".join(bits)


def generate(dataset_id: str = "ow", seed: int = 0, n_props: int = 200, n_concepts: int = 300,
             n_heldout: int = 50, n_instances: int = 20_000, n_texts: int = 2_000,
             fewshot_shots: tuple[int, ...] = (1, 2, 4), queries_per_episode: int = 6) -> Dataset:
    ctx = GenContext(dataset_id, seed, "objectworld", VERSION)
    world = ObjectWorld(ctx, n_props, n_concepts, n_heldout)
    src = ctx.source("gen", "generator_truth", category=CATEGORY)
    rng = ctx.rng

    for p in world.props.values():
        ctx.add(PropertyDefRecord(
            record_id=ctx.rid("property_def"), source_id=src, data_category=CATEGORY,
            extraction=ctx.extraction(), property_id=p.pid, name=p.name, property_kind=p.kind,
            value_type=p.vtype, unit=p.unit, enum_values=list(p.enum_values) or None,
            range=(round(10 ** p.lo, 6), round(10 ** p.hi, 6)) if p.vtype == "float" else None,
            log_scale=p.vtype == "float",
        ))

    texts_left = n_texts
    for c in world.concepts.values():
        if c.heldout:
            continue
        rec = world.definition(c, src)
        if texts_left > 0:
            seg_id = ctx.rid("segment")
            text = world.render_definition(rec)
            ctx.add(SegmentRecord(record_id=seg_id, source_id=src, data_category=CATEGORY,
                                  extraction=ctx.extraction(), segment_id=seg_id, text=text))
            rec.span = Span(segment_id=seg_id, start=0, end=len(text))
            texts_left -= 1
        ctx.add(rec)

    leaves = [c for c in world.concepts.values() if not c.children and not c.is_component]
    train_leaves = [c for c in leaves if not c.heldout]
    per_leaf = max(1, n_instances // max(1, len(train_leaves)))
    for c in train_leaves:
        for i in range(per_leaf):
            scene = world.sample_instance(c.cid)
            split = "dev" if i % 10 == 9 else "train"
            ex = ConceptExampleRecord(
                record_id=ctx.rid("concept_example"), source_id=src, data_category=CATEGORY,
                extraction=ctx.extraction(), example_id=ctx.rid("example"), label=c.cid, scene=scene,
            )
            if texts_left > 0 and rng.random() < 0.3:
                seg_id = ctx.rid("segment")
                text = world.render_scene(scene, c.name)
                ctx.add(SegmentRecord(record_id=seg_id, source_id=src, data_category=CATEGORY,
                                      extraction=ctx.extraction(), segment_id=seg_id, text=text))
                ex.span = Span(segment_id=seg_id, start=0, end=len(text))
                texts_left -= 1
            ctx.add(ex)
            ctx.add(TaskRecord(
                record_id=ctx.rid("task"), source_id=src, data_category=CATEGORY, extraction=ctx.extraction(),
                task_id=ctx.rid("task_id"), family="objectworld/classify",
                goal={"predict": "label", "type": "concept"}, inputs={"scene": scene.model_dump(mode="json")},
                verifier=Verifier(type="exact", spec={"answer": c.cid}), split=split,
            ))

    # held-out compositions (02 §14.1 a)
    for c in train_leaves:
        if c.heldout_combo is None:
            continue
        for _ in range(3):
            scene = world.sample_instance(c.cid, combo_mode="force")
            ctx.add(TaskRecord(
                record_id=ctx.rid("task"), source_id=src, data_category=CATEGORY, extraction=ctx.extraction(),
                task_id=ctx.rid("task_id"), family="objectworld/classify",
                goal={"predict": "label", "type": "concept"}, inputs={"scene": scene.model_dump(mode="json")},
                verifier=Verifier(type="exact", spec={"answer": c.cid}), split="heldout_composition",
            ))

    # held-out concepts: few-shot episodes (02 §14.1 b)
    for c in leaves:
        if not c.heldout:
            continue
        siblings = [s for s in world.concepts[c.parent].children if s != c.cid] if c.parent else []
        for k in fewshot_shots:
            support = [world.sample_instance(c.cid).model_dump(mode="json") for _ in range(k)]
            queries, answers = [], []
            for q in range(queries_per_episode):
                target = c.cid if (q % 2 == 0 or not siblings) else ctx.choice(siblings)
                queries.append(world.sample_instance(target).model_dump(mode="json"))
                answers.append(target == c.cid)
            ctx.add(TaskRecord(
                record_id=ctx.rid("task"), source_id=src, data_category=CATEGORY, extraction=ctx.extraction(),
                task_id=ctx.rid("task_id"), family="objectworld/fewshot",
                goal={"predict": "membership", "concept": c.cid, "shots": k},
                inputs={"support": support, "queries": queries, "parent": c.parent},
                verifier=Verifier(type="exact", spec={"answers": answers}), split="heldout_concept",
            ))

    # wheel-style out-of-range instances (02 §14.1 c)
    for c in train_leaves[: max(1, len(train_leaves) // 4)]:
        typical_floats = [pid for pid, (con, d) in world.all_constraints(c.cid).items()
                          if not d and world.props[pid].vtype == "float"]
        if not typical_floats or c.parent is None:
            continue
        pid = ctx.choice(typical_floats)
        foreign = [x for x in world.contexts if x not in c.contexts]
        scene = world.sample_instance(c.cid, out_of_range=pid, context=ctx.choice(foreign) if foreign else None)
        ctx.add(TaskRecord(
            record_id=ctx.rid("task"), source_id=src, data_category=CATEGORY, extraction=ctx.extraction(),
            task_id=ctx.rid("task_id"), family="objectworld/wheelstyle",
            goal={"predict": "parent_class", "type": "concept"}, inputs={"scene": scene.model_dump(mode="json")},
            verifier=Verifier(type="exact", spec={"answer": c.parent, "leaf": c.cid, "out_of_range": pid}),
            split="wheelstyle",
        ))

    ctx.dataset.meta = {
        "n_props": len(world.props),
        "n_concepts": len(world.concepts),
        "heldout_concepts": sorted(c.cid for c in leaves if c.heldout),
        "heldout_combos": {c.cid: list(c.heldout_combo) for c in train_leaves if c.heldout_combo},
        "max_depth": max(c.depth for c in world.concepts.values()),
    }
    ctx.dataset.world = world  # type: ignore[attr-defined]  # in-memory handle for tests and later stages
    return ctx.dataset


def log10(x: float) -> float:
    return math.log10(max(x, 1e-12))
