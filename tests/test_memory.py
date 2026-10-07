"""M2: memory — record store, banded index recall vs formula (04 §4.1), sketch (04 §7),
association field (04 §6), lifecycle (04 §9), persistence (04 §11)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from srm.core.evidence import EvidenceLedger
from srm.interface import InterfaceLayer
from srm.interface import codes as C
from srm.interface.messages import Lifecycle, Qualifiers, RecordKind
from srm.interface.values import Value
from srm.memory import HIPPOCAMPUS, MemorySystem
from srm.memory.association import AssociationField
from srm.memory.index import AliasIndex, BandedIndex
from srm.memory.lifecycle import LifecycleInputs, next_lifecycle
from srm.memory.payloads import EngramPayload, EntityPayload
from srm.memory.sketch import KnowledgeSketch, sketch_size


def test_banded_recall_matches_formula() -> None:
    """Empirical recall of a code whose blocks agree with probability p equals 1 − (1 − p^r)^{n_b}."""
    B, L, r, n_b = 64, 64, 3, 21
    layout = tuple(tuple(range(j * r, j * r + r)) for j in range(n_b))
    rng = np.random.default_rng(0)
    for p in (0.4, 0.5, 0.6):
        idx = BandedIndex(layout, L)
        base = C.random_codes(rng, 3000, B, L)
        idx.add_many(np.arange(3000), base)
        hits = 0
        for i in range(3000):
            q = base[i].copy()
            flip = rng.random(B) > p
            q[flip] = (q[flip] + rng.integers(1, L, flip.sum())) % L
            locs, _ = idx.candidates(q, 10_000)
            hits += int(i in set(locs.tolist()))
        expected = 1 - (1 - p**r) ** n_b
        assert abs(hits / 3000 - expected) < 0.03, (p, hits / 3000, expected)


def test_banded_false_candidates_are_rare() -> None:
    B, L, r, n_b = 64, 64, 3, 21
    layout = tuple(tuple(range(j * r, j * r + r)) for j in range(n_b))
    rng = np.random.default_rng(1)
    idx = BandedIndex(layout, L)
    idx.add_many(np.arange(20_000), C.random_codes(rng, 20_000, B, L))
    counts = [len(idx.candidates(q, 10_000)[0]) for q in C.random_codes(rng, 200, B, L)]
    expected = 20_000 * n_b / L**r  # ≈ 1.6 per query
    assert np.mean(counts) < 3 * expected + 1


@settings(max_examples=30)
@given(st.lists(st.text(min_size=1, max_size=12), min_size=1, max_size=200, unique=True))
def test_sketch_has_no_false_negatives(keys: list[str]) -> None:
    sk = KnowledgeSketch(expected_keys=50, fpr=0.01)  # deliberately undersized → saturation
    for k in keys:
        sk.add(k)
    assert all(sk.maybe_contains(k) for k in keys)
    for k in keys[: len(keys) // 2]:
        sk.remove(k)
    assert all(sk.maybe_contains(k) for k in keys[len(keys) // 2 :])


def test_sketch_false_positive_rate_near_target() -> None:
    m, k = sketch_size(10_000, 1e-2)
    sk = KnowledgeSketch(10_000, 1e-2)
    assert (sk.m, sk.k) == (m, k)
    for i in range(10_000):
        sk.add(f"in:{i}")
    fp = sum(sk.maybe_contains(f"out:{i}") for i in range(20_000)) / 20_000
    assert fp < 0.02
    with pytest.raises(KeyError):
        KnowledgeSketch(10, 0.01).remove("never-added")


def test_alias_index_exact_and_fuzzy() -> None:
    a = AliasIndex()
    a.add("Freedonia", "e:1")
    a.add("Sylvania", "e:2")
    assert a.lookup("  FREEDONIA ") == {"e:1"}
    assert a.fuzzy("Fredonia", 0.5)[0][0] == "e:1"


def test_association_spread_and_hebbian() -> None:
    f = AssociationField()
    f.link(1, 2, "is_a", 1.0)
    f.link(2, 3, "part_of", 1.0)
    f.link(1, 4, "contradicts", 1.0)
    act = f.spread({1: 1.0}, {"is_a": 0.8, "part_of": 0.7, "contradicts": 0.0}, gamma=0.5, hops=2, k=10)
    assert act[2] > act.get(3, 0) > 0 and 4 not in act
    n = f.hebbian({1: 1.0, 2: 0.5}, eta=0.1, modulator=1.0, eta_decay=0.0, w_max=1.0)
    assert n == 2 and abs(f.weight(1, 2, "co_activated") - 0.05) < 1e-9


def test_lifecycle_transitions(cfg) -> None:
    def nx(**kw):
        base = dict(lifecycle=Lifecycle.NEW, n_independent_roots=1, belief=0.8, disbelief=0.0, e_plus=0.9,
                    e_minus=0.0, use_count=0)
        base.update(kw)
        return next_lifecycle(LifecycleInputs(**base), cfg)

    assert nx() == Lifecycle.NEW
    assert nx(n_independent_roots=2) == Lifecycle.CORROBORATED
    assert nx(n_independent_roots=2, use_count=cfg.memory.n_use) == Lifecycle.USED
    assert nx(unresolved_contradiction=True) == Lifecycle.CONTESTED
    assert nx(lifecycle=Lifecycle.CONTESTED) == Lifecycle.CONTESTED
    assert nx(lifecycle=Lifecycle.CONTESTED, contradiction_resolved=True) == Lifecycle.NEW
    assert nx(belief=0.2, disbelief=0.7, e_plus=0.4, e_minus=1.0) == Lifecycle.DEPRECATED


def _memory(cfg) -> MemorySystem:
    il = InterfaceLayer(cfg)
    il.relations.add_relation("rel:r", "r", cardinality="functional")
    return MemorySystem(cfg, il)


def test_write_retrieve_and_structured_lookup(cfg) -> None:
    mem = _memory(cfg)
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((300, cfg.interface.d)).astype(np.float32)
    ids = []
    for i, v in enumerate(vecs):
        payload = EngramPayload(f"e{i % 30}", "rel:r", Value("int", i), Qualifiers(time=(2000.0, 2010.0)))
        ids.append(mem.write(RecordKind.ENGRAM, f"f{i}", v, payload))
    hits = mem.retrieve(vecs[17], kind=RecordKind.ENGRAM, k=5)
    assert hits[0][0] == ids[17]
    assert len(mem.retrieve_pattern("e7", "rel:r")) == 10
    assert mem.retrieve_pattern("e7", "rel:r", qualifiers=Qualifiers(time=(2015.0, 2016.0))) == []
    assert mem.sketch.maybe_contains("er:e7|rel:r") and mem.sketch.definitely_absent("er:e7|rel:other")
    by_code = mem.retrieve(mem.record(ids[5]).content_code, k=1)
    assert by_code[0][0] == ids[5]


def test_cleanup_completes_a_noisy_cue(cfg) -> None:
    mem = _memory(cfg)
    rng = np.random.default_rng(4)
    vecs = rng.standard_normal((20, cfg.interface.d)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    ids = [mem.write(RecordKind.ENGRAM, f"x{i}", v, EngramPayload("s", "rel:r", Value("int", i))) for i, v in enumerate(vecs)]
    noisy = vecs[3] + 0.6 * rng.standard_normal(cfg.interface.d).astype(np.float32) / np.sqrt(cfg.interface.d) * 3
    y = mem.cleanup(noisy, ids)
    assert float(y @ vecs[3]) > float(noisy @ vecs[3]) / np.linalg.norm(noisy)


def test_evidence_independence_rule(cfg) -> None:
    mem = _memory(cfg)
    rid = mem.write(RecordKind.ENGRAM, "f", np.ones(cfg.interface.d), EngramPayload("s", "rel:r", Value("int", 1)))
    mem.add_evidence(rid, "rootA", 0.5, 0.0)
    mem.add_evidence(rid, "rootA", 0.5, 0.0)  # same root: max, not sum
    assert mem.record(rid).ledger.e_plus == pytest.approx(0.5)
    mem.add_evidence(rid, "rootB", 0.5, 0.0)
    assert mem.record(rid).ledger.e_plus == pytest.approx(1.0)


def test_persistence_roundtrip(cfg, tmp_path) -> None:
    mem = _memory(cfg)
    mem.register_source("s1", "curated_kb", "curated_kb")
    eid = mem.write(RecordKind.ENTITY, "ent", np.ones(cfg.interface.d), EntityPayload("ent:1", "Zorbia", "concept:country", ["ZB"]))
    led = EvidenceLedger()
    led.add("s1", 0.9, 0.0)
    fid = mem.write(RecordKind.ENGRAM, "fact", np.arange(cfg.interface.d, dtype=np.float32),
                    EngramPayload("ent:1", "rel:r", Value("int", 7)), led, [{"source_id": "s1"}])
    mem.link(eid, fid, "has_property", 0.7)
    mem.set_lifecycle(fid, Lifecycle.CORROBORATED)
    mem.save(tmp_path / "mem")
    back = MemorySystem.load(tmp_path / "mem", cfg, mem.interface)
    assert back.n_records() == mem.n_records()
    assert back.record(fid).payload == mem.record(fid).payload
    assert back.record(fid).ledger.e_plus == pytest.approx(0.9)
    assert back.record(fid).lifecycle == Lifecycle.CORROBORATED
    assert back.link_surface("zb") == ["ent:1"]
    assert back.retrieve_pattern("ent:1", "rel:r") == [fid]
    assert back.assoc.weight(eid, fid, "has_property") == pytest.approx(0.7)
    assert back.sketch.maybe_contains("er:ent:1|rel:r")
    assert back.sources["s1"].trust == pytest.approx(0.9)


def test_regions_and_page_assignment(cfg) -> None:
    mem = _memory(cfg)
    region = mem.new_region("objects")
    rid = mem.write(RecordKind.ENGRAM, "q", np.ones(cfg.interface.d), EngramPayload("s", "rel:r", Value("int", 1)), target=region)
    st, local = mem._loc(rid)
    assert st.name == "objects" and st.page_of(local) == 0
    assert mem.retrieve_pattern("s", "rel:r") == [rid]
    assert HIPPOCAMPUS in mem.stores
