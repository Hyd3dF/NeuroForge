"""M4: Body tools (07 §6), primitives (05 §4), synthesis (05 §5.2), simulator (05 §10)."""

from __future__ import annotations

import numpy as np
import pytest

from srm.body import dsl
from srm.body.tools import Body, SubsetViolation, validate_python_subset
from srm.core import primitives as P
from srm.core.scene import SceneGraph, SNode
from srm.core.schema import PropertyProfile, Schema
from srm.core.simulator import group_by_behaviour, identity_counterexample, probe_inputs
from srm.core.synthesis import Synthesizer
from srm.interface.messages import PatternAtom
from srm.interface.values import Value


# --- Body ----------------------------------------------------------------------------------------
@pytest.mark.parametrize("src", [
    "import os\ndef solve(xs, k):\n    return xs",
    "def solve(xs, k):\n    return open('/etc/passwd').read()",
    "def solve(xs, k):\n    return xs.__class__",
    "class A:\n    pass",
    "def solve(xs, k):\n    raise RuntimeError('x')",
    "def solve(xs, k):\n    return eval('1')",
    "lambda x: x",
])
def test_subset_validator_rejects_unsafe_code(src: str) -> None:
    with pytest.raises(SubsetViolation):
        validate_python_subset(src)


def test_sandbox_runs_tests_and_enforces_timeout() -> None:
    body = Body(timeout_s=1.0)
    good = "def solve(xs: list[int], k: int) -> int:\n    return sum(xs) + k\n"
    res = body.call("tests", tests=[{"xs": [1, 2], "k": 3, "out": 6}, {"xs": [], "k": 1, "out": 2}], source=good)
    assert res.ok is False and res.value["passed"] == 1
    loop = "def solve(xs: list[int], k: int) -> int:\n    while True:\n        k = k + 1\n    return k\n"
    assert body.call("python_sandbox", source=loop, entry="solve", cases=[[[1], 0]]).error == "Timeout"
    assert body.call("python_sandbox", source="import os", entry="solve", cases=[]).error.startswith("SubsetViolation")


def test_dsl_sympy_units_tools() -> None:
    body = Body()
    prog = dsl.Node("sum", (dsl.Node("xs"),))
    assert body.call("dsl", program=prog, xs=[1, 2, 3], k=0).value == 6
    assert not body.call("dsl", program=dsl.Node("head", (dsl.Node("xs"),)), xs=[], k=0).ok
    assert body.call("sympy", op="expand", expr="(x+1)*(x-1)").value == "x**2 - 1"
    assert body.call("sympy", op="equivalent", expr="(x+1)**2", other="x**2+2*x+1").value is True
    assert body.call("units", value=36.0, from_unit="km/h", to_unit="m/s").value == pytest.approx(10.0)
    assert not body.call("units", value=1.0, from_unit="km", to_unit="kg").ok


# --- COMPARE / ALIGN / ABSTRACT / SPECIALIZE -------------------------------------------------------
def scene(props: dict[str, Value], parts: list[tuple[str, dict[str, Value]]] = (), ctx: str | None = None,
          prefix: str = "x") -> SceneGraph:
    g = SceneGraph(root=prefix)
    g.nodes[prefix] = SNode(prefix, None, dict(props))
    for i, (ptype, pprops) in enumerate(parts):
        ref = f"{prefix}.p{i}"
        g.nodes[ref] = SNode(ref, ptype, dict(pprops))
        g.relations.append(("rel:part_of", (ref, prefix)))
    if ctx:
        g.nodes[f"{prefix}.ctx"] = SNode(f"{prefix}.ctx", ctx)
        g.relations.append(("rel:in_context", (prefix, f"{prefix}.ctx")))
    return g


def wheel(diameter: float, rotates: bool = True) -> dict[str, Value]:
    return {"geo_round": Value("bool", True), "dyn_rotates": Value("bool", rotates),
            "geo_diameter": Value("float", diameter, "m"), "rel_color": Value("enum", "black")}


def test_abstract_and_compare_tolerate_variable_properties() -> None:
    wheels = [scene(wheel(d)) for d in (0.6, 0.65, 0.7, 0.62)]
    schema = P.abstract(wheels, "schema:wheel", "wheel")
    assert schema.n_instances == 4 and schema.profiles["geo_diameter"].vtype == "scalar"
    ok = P.compare(wheel(0.66), schema)
    assert not ok.by_status("mismatch") and ok.score > 0
    huge = P.compare(wheel(6.0), schema)  # far out of range → mismatch on diameter only
    assert [i.property_id for i in huge.by_status("mismatch")] == ["geo_diameter"]
    assert huge.score < ok.score
    broken = P.compare(wheel(0.65, rotates=False), schema)
    assert any(i.property_id == "dyn_rotates" for i in broken.by_status("mismatch"))


def test_profile_from_definition_and_specialize() -> None:
    prof = PropertyProfile.from_constraint(Value("range", (0.5, 0.9), "m"), defining=True)
    assert prof.z(Value("float", 0.7, "m")) < 1.0 and prof.z(Value("float", 9.0, "m")) > 5.0
    s = Schema("schema:x")
    s.profiles["geo_diameter"] = prof
    s.profiles["dyn_rotates"] = PropertyProfile.from_constraint(Value("bool", True), True)
    props, inherited = P.specialize(s, {"geo_diameter": Value("float", 0.8, "m")})
    assert inherited == {"dyn_rotates"} and props["dyn_rotates"].value is True


def test_align_recovers_correspondence_and_candidate_inferences() -> None:
    src = scene(wheel(0.6), [("concept:hub", {"geo_round": Value("bool", True)}), ("concept:tire", {"rel_color": Value("enum", "black")})],
                ctx="ctx:road", prefix="a")
    tgt_props = dict(wheel(1.2))
    tgt = scene(tgt_props, [("concept:tire", {"rel_color": Value("enum", "black")}), ("concept:hub", {"geo_round": Value("bool", True)})],
                ctx="ctx:road", prefix="b")
    tgt.relations = [r for r in tgt.relations if r[0] != "rel:in_context"]  # target lacks the context relation
    al = P.align(src, tgt)
    assert al.pairs["a"] == "b"
    assert al.pairs["a.p0"] == "b.p1" and al.pairs["a.p1"] == "b.p0"  # types win over order
    assert ("rel:in_context", ("b", "b.ctx")) in al.candidate_inferences


def test_query_join_multi_hop_and_missing() -> None:
    facts = {("acme", "rel:ceo"): [(1, "acme", "rel:ceo", Value("entity_ref", "ann"))],
             ("ann", "rel:born_in"): [(2, "ann", "rel:born_in", Value("entity_ref", "paris"))]}
    lookup = lambda s, r: facts.get((s, r), [])  # noqa: E731
    atoms = [PatternAtom("acme", "rel:ceo", "?p"), PatternAtom("?p", "rel:born_in", "?c")]
    res = P.query_join(atoms, lookup)
    assert res.bindings[0]["?c"].value == "paris" and res.support[0] == [1, 2]
    res2 = P.query_join(atoms + [PatternAtom("?c", "rel:located_in", "?x")], lookup)
    assert res2.bindings == [] and res2.missing == [("paris", "rel:located_in")]


def test_order_aggregate_search_and_registry() -> None:
    vals = [Value("int", 3), Value("int", 1), Value("int", 2)]
    assert [v.value for v in P.order(vals)] == [1, 2, 3]
    assert P.aggregate(vals, "sum") == Value("int", 6)
    sols, n = P.search([0], lambda s: [s + 1, s + 2], lambda s: -abs(7 - s), lambda s: s == 7, budget=50)
    assert sols == [7] and n < 50
    assert P.check_complete() == []


# --- synthesis and simulator ------------------------------------------------------------------------
def examples_for(prog: dsl.Node, seed: int = 0, n: int = 5) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        xs, k = dsl.random_input(rng)
        try:
            out.append({"xs": xs, "k": k, "out": dsl.run(prog, xs, k)})
        except dsl.DSLError:
            pass
    return out


@pytest.mark.parametrize("source", ["(sort xs)", "(map double (filter is_even xs))", "(sum (take k xs))", "(length (unique xs))"])
def test_synthesis_finds_programs_that_generalize(source: str) -> None:
    from srm.perception.parsers import parse_dsl

    target = parse_dsl(source)
    ex = examples_for(target, seed=1, n=6)
    syn = Synthesizer(ex, dsl.type_of(target), max_size=6)
    sols = syn.run(300_000)
    assert sols, source
    held = examples_for(target, seed=99, n=20)
    assert any(all(dsl.run(s, e["xs"], e["k"]) == e["out"] for e in held) for s in sols)


def test_library_components_reduce_search() -> None:
    from srm.perception.parsers import parse_dsl

    target = parse_dsl("(map inc (sort_desc (filter is_pos xs)))")
    ex = examples_for(target, seed=3, n=6)
    cold = Synthesizer(ex, "list", max_size=6)
    cold.run(400_000)
    warm = Synthesizer(ex, "list", library=[parse_dsl("(sort_desc (filter is_pos xs))")], max_size=6)
    warm.run(400_000)
    assert warm.solutions and warm.evaluated < max(1, cold.evaluated) / 5


def test_incremental_steps_respect_budget() -> None:
    syn = Synthesizer(examples_for(dsl.Node("sort", (dsl.Node("xs"),))), "list", max_size=5)
    syn.step(10)
    assert syn.evaluated <= 10 + 1


def test_simulator_groups_and_distinguishes() -> None:
    from srm.perception.parsers import parse_dsl

    a, b, c = parse_dsl("(sort xs)"), parse_dsl("(reverse (sort_desc xs))"), parse_dsl("(unique (sort xs))")
    inputs = probe_inputs(np.random.default_rng(0), 24)
    groups = group_by_behaviour([a, b, c], inputs)
    assert sorted(len(g) for g in groups) == [1, 2]  # sort ≡ reverse∘sort_desc; unique differs on duplicates
    assert identity_counterexample("(x+1)**2", "x**2+1") is not None
    assert identity_counterexample("(x+1)**2", "x**2+2*x+1") is None
