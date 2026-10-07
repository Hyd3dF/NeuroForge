"""Batch 1 integration: generators → SEF → ingestion/triage → memory → workspace → Support →
decision → Answer Record (the epistemic vertical slice, 11 §6.1–6.2)."""

from __future__ import annotations

from collections import Counter

import pytest

from srm.data.generators import factstream
from srm.data.io import read_dataset, write_dataset
from srm.ingest import Ingestor
from srm.interface import InterfaceLayer
from srm.interface.messages import DecisionClass, Lifecycle
from srm.interface.values import Value
from srm.memory import MemorySystem
from srm.runtime.factual import FactualAnswerer


def build(cfg, ds):
    il = InterfaceLayer(cfg)
    mem = MemorySystem(cfg, il)
    ing = Ingestor(mem, cfg)
    report = ing.ingest(ds.records)
    return mem, ing, report


@pytest.fixture(scope="module")
def world(cfg):
    ds = factstream.generate("fs", 21, n_countries=30)
    mem, ing, report = build(cfg, ds)
    return ds, mem, ing, report


def test_ingestion_report(world) -> None:
    ds, mem, ing, report = world
    o = report.outcomes
    assert o["entity"] == len(ds.by_kind("entity"))
    assert o["novel"] + o["known_same"] == len(ds.by_kind("fact"))
    assert o["conflict"] == o["conflict_resolved"] + sum(
        1 for pair in ing.contested_pairs
    )
    assert report.deferred["question"] == len(ds.by_kind("question"))  # eval stream, not memory


def test_every_question_gets_its_gold_state(world, cfg) -> None:
    ds, mem, ing, _ = world
    answerer = FactualAnswerer(mem, cfg, ing.contested)
    confusion = Counter()
    for q in ds.by_kind("question"):
        a = answerer.answer(q)
        confusion[(q.gold_state, a.cls.value)] += 1
        assert a.cls.value == q.gold_state, (q.query, q.gold_state, a.cls.value, a.decision.reason)
        if q.gold_state == "KNOWN":
            g = q.gold_answer
            assert a.answer.matches(Value(g.type, g.value, g.unit, float(g.tolerance or 0.0)))
            assert a.record.answer_claims and all(not a.workspace.claims[c].tainted for c in a.record.answer_claims)
        if q.gold_state == "UNKNOWN-ABSENT":
            assert a.record.missing_keys
    assert len({k[0] for k in confusion}) == 5


def test_answer_record_explains_rejections(world, cfg) -> None:
    ds, mem, ing, _ = world
    answerer = FactualAnswerer(mem, cfg, ing.contested)
    resolved = [rid for (a, b, nature) in ing.report.contradictions for rid in (a, b)
                if mem.record(rid).lifecycle == Lifecycle.DEPRECATED]
    assert resolved, "the stream contains resolved conflicts"
    payload = mem.record(resolved[0]).payload
    from srm.data.sef import FactPattern, PatternAtom, QuestionRecord

    name = mem.record(mem.symbols[payload.subject]).payload.name
    q = QuestionRecord(record_id="t:q", source_id="t:s", query="?", gold_state="KNOWN",
                       pattern=FactPattern(atoms=[PatternAtom(subject=f"@{name}", relation=payload.relation, object="?x")]))
    a = answerer.answer(q)
    assert a.cls == DecisionClass.KNOWN
    assert any(not r["residual"] and r["support"] < a.decision.best.support for r in a.record.rejected_hypotheses)
    assert a.record.sources and a.record.justification_subgraph


def test_persisted_memory_gives_identical_answers(world, cfg, tmp_path) -> None:
    ds, mem, ing, _ = world
    mem.save(tmp_path / "m")
    back = MemorySystem.load(tmp_path / "m", cfg, mem.interface)
    a1 = FactualAnswerer(mem, cfg, ing.contested)
    a2 = FactualAnswerer(back, cfg, ing.contested)
    for q in ds.by_kind("question")[:150]:
        r1, r2 = a1.answer(q), a2.answer(q)
        assert r1.cls == r2.cls
        assert (r1.answer is None) == (r2.answer is None)
        if r1.answer is not None:
            assert r1.answer.matches(r2.answer)


def test_sef_on_disk_roundtrip_feeds_the_same_pipeline(cfg, tmp_path) -> None:
    ds = factstream.generate("fs", 33, n_countries=8)
    write_dataset(ds, tmp_path / "ds")
    back = read_dataset(tmp_path / "ds")  # shards are grouped by category; ingestion re-orders by kind
    mem, ing, report = build(cfg, back)
    answerer = FactualAnswerer(mem, cfg, ing.contested)
    for q in back.by_kind("question"):
        assert answerer.answer(q).cls.value == q.gold_state


def test_hedged_and_reported_claims_never_become_knowledge(world, cfg) -> None:
    ds, mem, ing, _ = world
    answerer = FactualAnswerer(mem, cfg, ing.contested)
    weak_ids = {f.record_id for f in ds.by_kind("fact") if f.epistemic.modality in ("hedged", "reported")}
    for q in ds.by_kind("question"):
        a = answerer.answer(q)
        if a.cls == DecisionClass.KNOWN:
            sef_ids = {s for c in a.record.answer_claims
                       for s in mem.record(a.workspace.claims[c].record_id).payload.sef_record_ids}
            assert not sef_ids <= weak_ids
