"""M1: SEF models (02 §3–4), IO and dataset manifests (02 §3), synthetic generators (02 §14)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from pydantic import ValidationError

from srm.body import dsl
from srm.data.generators import dslworld, factstream, mathworld, objectworld
from srm.data.io import read_dataset, write_dataset
from srm.data.sef import FactRecord, parse_record, record_to_dict


def _fact(**over):
    base = {
        "record_id": "t:fact:0", "kind": "fact", "source_id": "t:source:a",
        "subject": "e1", "relation": "rel:x", "object": {"type": "int", "value": 3},
    }
    base.update(over)
    return base


def test_parse_valid_fact_and_content_hash_ignores_record_id() -> None:
    a = parse_record(_fact())
    b = parse_record(_fact(record_id="t:fact:99"))
    assert isinstance(a, FactRecord)
    assert a.compute_content_hash() == b.compute_content_hash()
    c = parse_record(_fact(object={"type": "int", "value": 4}))
    assert c.compute_content_hash() != a.compute_content_hash()


@pytest.mark.parametrize(
    "bad",
    [
        _fact(kind="nonsense"),
        _fact(source_id=None),
        _fact(sef_version="2.0"),
        _fact(epistemic={"modality": "reported"}),               # reported needs attribution
        _fact(epistemic={"modality": "hedged", "hedge": 0.0}),   # hedged needs hedge > 0
        _fact(context={"time": [2010, 2000]}),                   # interval order
        _fact(extraction={"method": "generator", "confidence": 1.5}),
        _fact(unexpected_field=1),
    ],
)
def test_invalid_records_are_rejected(bad: dict) -> None:
    with pytest.raises(ValidationError):
        parse_record(bad)


def test_dataset_roundtrip_and_tamper_detection(tmp_path) -> None:
    ds = factstream.generate("fs", 3, n_countries=5)
    manifest = write_dataset(ds, tmp_path, shard_records=200)
    assert manifest["record_counts"] == ds.counts()
    back = read_dataset(tmp_path)
    assert sorted(r.content_hash for r in back.records) == sorted(r.content_hash for r in ds.records)
    shard = tmp_path / manifest["shards"][0]["path"]
    lines = shard.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["record_id"] = rec["record_id"] + "x"
    lines[0] = json.dumps(rec, sort_keys=True)
    shard.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="shard hash mismatch"):
        read_dataset(tmp_path)


def test_generators_are_deterministic() -> None:
    for gen, kwargs in (
        (factstream.generate, {"n_countries": 6}),
        (objectworld.generate, {"n_props": 40, "n_concepts": 30, "n_heldout": 4, "n_instances": 200, "n_texts": 20}),
        (dslworld.generate, {"n_tasks": 60}),
        (mathworld.generate, {"n_problems": 60}),
    ):
        a, b = gen("x", 11, **kwargs), gen("x", 11, **kwargs)
        assert a.content_digest() == b.content_digest()
        assert gen("x", 12, **kwargs).content_digest() != a.content_digest()


def test_every_generated_record_reparses() -> None:
    ds = objectworld.generate("ow", 2, n_props=40, n_concepts=30, n_heldout=4, n_instances=100, n_texts=20)
    for r in ds.records:
        again = parse_record(record_to_dict(r))
        assert again.compute_content_hash() == r.content_hash


def test_factstream_gold_states_cover_the_lattice() -> None:
    ds = factstream.generate("fs", 0, n_countries=25)
    golds = {q.gold_state for q in ds.by_kind("question")}
    assert golds == {"KNOWN", "INSUFFICIENT", "CONTESTED", "UNKNOWN-DECLARED", "UNKNOWN-ABSENT"}
    entity_names = {e.canonical_name for e in ds.by_kind("entity")}
    withheld = [q for q in ds.by_kind("question") if q.pattern.atoms[0].subject[1:] not in entity_names]
    assert withheld and all(q.gold_state == "UNKNOWN-ABSENT" for q in withheld)


def test_objectworld_heldout_splits_do_not_leak() -> None:
    ds = objectworld.generate("ow", 5, n_props=60, n_concepts=60, n_heldout=6, n_instances=600, n_texts=50)
    held = set(ds.meta["heldout_concepts"])
    defined = {r.concept_id for r in ds.by_kind("concept_definition")}
    labels = {r.label for r in ds.by_kind("concept_example")}
    assert held and not (held & defined) and not (held & labels)
    fewshot = [t for t in ds.by_kind("task") if t.family == "objectworld/fewshot"]
    assert {t.goal["concept"] for t in fewshot} == held
    combos = ds.meta["heldout_combos"]
    for ex in ds.by_kind("concept_example"):  # excluded value never appears in training instances
        if ex.label in combos:
            pid, val = combos[ex.label]
            root = next(n for n in ex.scene.nodes if n.ref == ex.scene.root)
            assert root.properties[pid].value != val
    forced = [t for t in ds.by_kind("task") if t.split == "heldout_composition"]
    assert forced
    wheel = [t for t in ds.by_kind("task") if t.family == "objectworld/wheelstyle"]
    assert wheel and all(t.verifier.spec["answer"] for t in wheel)


def test_dsl_reference_programs_match_python_rendering() -> None:
    ds = dslworld.generate("dsl", 4, n_tasks=150)
    builtins = {name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
                for name in ("len", "range", "sorted", "min", "max", "sum", "abs", "list", "dict", "set", "zip")}
    for task in ds.by_kind("task"):
        prog = dsl.Node.from_json(task.reference_solution["program"])
        ns: dict = {"__builtins__": builtins}
        exec(task.reference_solution["python"], ns)  # noqa: S102 (trusted generator output)
        for case in task.inputs["examples"] + task.verifier.spec["tests"]:
            expected = case["out"]
            assert dsl.run(prog, case["xs"], case["k"]) == expected
            assert ns["solve"](case["xs"], case["k"]) == expected


def test_dsl_heldout_compositions_are_absent_from_train() -> None:
    ds = dslworld.generate("dsl", 9, n_tasks=300)
    held = {tuple(p) for p in ds.meta["heldout_pairs"]}
    for task in ds.by_kind("task"):
        edges = set(dsl.Node.from_json(task.reference_solution["program"]).edges())
        if task.split == "heldout_composition":
            assert edges & held
        else:
            assert not (edges & held)


def test_dsl_type_checking_and_errors() -> None:
    prog = dsl.Node("head", (dsl.Node("xs"),))
    assert dsl.type_of(prog) == "int"
    with pytest.raises(dsl.DSLError):
        dsl.run(prog, [], 0)
    with pytest.raises(dsl.DSLError):
        dsl.type_of(dsl.Node("map", (dsl.Node("xs"), dsl.Node("xs"))))
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = dsl.sample_program(rng, "list", 4)
        assert dsl.type_of(p) == "list"


def test_mathworld_answers_verify() -> None:
    import sympy as sp

    ds = mathworld.generate("m", 1, n_problems=200)
    x = sp.Symbol("x")
    for t in ds.by_kind("task"):
        if t.family.startswith("math/word/"):
            q = t.reference_solution["quantities"]
            assert abs(float(sp.sympify(t.reference_solution["formula"]).subs(q)) - t.verifier.spec["answer"]) < 1e-9
        else:
            text = t.inputs["text"]
            expr = text.split(" ", 1)[1].rstrip(".")
            ans = sp.sympify(t.verifier.spec["answer"])
            assert sp.simplify(sp.sympify(expr.replace("^", "**")) - ans) == 0
    assert any(t.split == "test_perturbed" for t in ds.by_kind("task"))
