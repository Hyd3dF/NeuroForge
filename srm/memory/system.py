"""MemorySystem: Hippocampal Index + Cortical Library regions + indices + sketch + associations (04).

Stores: id 0 is the Hippocampal Index (fast one-shot writes, 04 §8); ids ≥ 1 are
Library regions (04 §3).  Indices are per store; the alias index, Knowledge
Sketch, Association Field and source registry are global.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from srm.config.build_config import BuildConfig
from srm.core.evidence import EvidenceLedger
from srm.interface import codes as C
from srm.interface.codec import normalize_surface
from srm.interface.codespace import normalize
from srm.interface.layer import InterfaceLayer
from srm.interface.messages import EpistemicState, Lifecycle, Qualifiers, RecordKind
from srm.memory import sketch as SK
from srm.memory.association import AssociationField
from srm.memory.index import AliasIndex, BandedIndex, DenseIndex, TripleIndex
from srm.memory.lifecycle import LifecycleInputs, next_lifecycle, plasticity_for
from srm.memory.payloads import EngramPayload, EntityPayload, OpenQuestionPayload
from srm.memory.store import RecordStore, RecordView, make_record_id, split_record_id

HIPPOCAMPUS = 0
DEFAULT_REGION = 1


@dataclass
class SourceInfo:
    source_id: str
    source_type: str
    trust_class: str
    trust: float
    root_source_id: str
    privacy_scope: str = "public"
    title: str | None = None


@dataclass
class StoreIndices:
    content: BandedIndex
    signature: BandedIndex
    triples: TripleIndex = field(default_factory=TripleIndex)


class MemorySystem:
    def __init__(self, config: BuildConfig, interface: InterfaceLayer) -> None:
        self.cfg = config
        self.interface = interface
        self.cs = interface.codespace
        m, i = config.memory, config.interface
        self.stores: dict[int, RecordStore] = {}
        self.indices: dict[int, StoreIndices] = {}
        self.region_names: dict[int, str] = {}
        self._new_store(HIPPOCAMPUS, "hippocampus")
        self._new_store(DEFAULT_REGION, "default")
        self.aliases = AliasIndex()
        self.sketch = SK.KnowledgeSketch(m.sketch_expected_keys, m.sketch_fpr)
        self.assoc = AssociationField()
        self.sources: dict[str, SourceInfo] = {}
        self.symbols: dict[str, int] = {}  # symbol id (entity id, concept id) → record id
        self.open_questions: dict[tuple[str, str], list[int]] = {}
        self.wal: list[dict[str, Any]] = []
        self._dense_index = DenseIndex(i.d_k)

    # --- stores -----------------------------------------------------------------------------
    def _new_store(self, store_id: int, name: str) -> RecordStore:
        i, m = self.cfg.interface, self.cfg.memory
        st = RecordStore(store_id, name, i.B, i.d, i.d_k, m.S_context, m.page_records)
        self.stores[store_id] = st
        self.indices[store_id] = StoreIndices(BandedIndex(self.cs.band_layout, i.L), BandedIndex(self.cs.band_layout, i.L))
        self.region_names[store_id] = name
        return st

    def new_region(self, name: str) -> int:
        rid = max(self.stores) + 1
        self._new_store(rid, name)
        return rid

    def _loc(self, record_id: int) -> tuple[RecordStore, int]:
        sid, local = split_record_id(record_id)
        return self.stores[sid], local

    def record(self, record_id: int) -> RecordView:
        st, local = self._loc(record_id)
        return st.view(local)

    def n_records(self) -> int:
        return sum(st.n for st in self.stores.values())

    # --- sources (02 §4.1, §10.1) -----------------------------------------------------------------
    def register_source(self, source_id: str, source_type: str, trust_class: str, root_source_id: str | None = None,
                        privacy_scope: str = "public", title: str | None = None) -> SourceInfo:
        priors = self.cfg.epistemics.trust_priors
        trust = priors.get(trust_class, priors.get(source_type, 0.5))
        root = root_source_id or source_id
        while root in self.sources and self.sources[root].root_source_id != root:
            root = self.sources[root].root_source_id
        info = SourceInfo(source_id, source_type, trust_class, trust, root, privacy_scope, title)
        self.sources[source_id] = info
        return info

    def source(self, source_id: str) -> SourceInfo:
        if source_id not in self.sources:
            raise KeyError(f"unknown source {source_id}; source records must be ingested first (02 §3)")
        return self.sources[source_id]

    # --- writes (04 §10) ----------------------------------------------------------------------------
    def write(
        self, kind: RecordKind, identity_key: str, dense: np.ndarray, payload: Any,
        ledger: EvidenceLedger | None = None, provenance: Iterable[dict[str, Any]] = (),
        target: int = HIPPOCAMPUS, lifecycle: Lifecycle = Lifecycle.NEW, signature: np.ndarray | None = None,
        context_codes: list[np.ndarray] | None = None, meta: dict[str, Any] | None = None,
    ) -> int:
        dense = normalize(dense)
        st = self.stores[target]
        content = self.cs.project(dense)
        identity = C.code_from_key(identity_key, self.cs.B, self.cs.L, salt="record")
        rid = st.append(
            kind, identity, content, dense, self.cs.key(dense), payload, ledger or EvidenceLedger(),
            list(provenance), lifecycle, signature, context_codes, meta, plasticity_for(lifecycle, self.cfg),
        )
        local = split_record_id(rid)[1]
        idx = self.indices[target]
        idx.content.add(local, content)
        if signature is not None:
            idx.signature.add(local, signature)
        self._index_payload(rid, kind, payload, idx)
        self.wal.append({"op": "write", "record_id": rid, "kind": kind.value, "t": time.time()})
        return rid

    def _index_payload(self, rid: int, kind: RecordKind, payload: Any, idx: StoreIndices) -> None:
        if kind == RecordKind.ENTITY:
            p: EntityPayload = payload
            self.symbols[p.entity_id] = rid
            self.sketch.add(SK.key_entity(p.entity_id))
            for surface in (p.name, *p.aliases):
                self.aliases.add(surface, p.entity_id)
                self.sketch.add(SK.key_alias(normalize_surface(surface)))
        elif kind == RecordKind.ENGRAM:
            e: EngramPayload = payload
            idx.triples.add_fact(rid, e.subject, e.relation, e.object.key())
            self.sketch.add(SK.key_entity(e.subject))
            self.sketch.add(SK.key_entity_relation(e.subject, e.relation))
        elif kind == RecordKind.OPEN_QUESTION:
            q: OpenQuestionPayload = payload
            self.open_questions.setdefault((q.subject, q.relation), []).append(rid)
            # D-022: a declared unknown is stored knowledge about the key, so the key is not absent.
            self.sketch.add(SK.key_entity(q.subject))
            self.sketch.add(SK.key_entity_relation(q.subject, q.relation))

    def add_evidence(self, record_id: int, root: str, delta_plus: float, delta_minus: float) -> bool:
        st, local = self._loc(record_id)
        changed = st.ledgers[local].add(root, delta_plus, delta_minus)
        if changed:
            self.wal.append({"op": "evidence", "record_id": record_id, "root": root,
                             "dp": delta_plus, "dm": delta_minus})
        return changed

    def opinion(self, record_id: int) -> tuple[float, float, float]:
        st, local = self._loc(record_id)
        o = st.ledgers[local].opinion(self.cfg.epistemics.W)
        return o.b, o.d, o.u

    def set_lifecycle(self, record_id: int, lifecycle: Lifecycle) -> None:
        st, local = self._loc(record_id)
        st.set_lifecycle(local, lifecycle, plasticity_for(lifecycle, self.cfg))
        self.wal.append({"op": "lifecycle", "record_id": record_id, "lifecycle": lifecycle.value})

    def update_lifecycle(self, record_id: int, unresolved_contradiction: bool = False,
                         contradiction_resolved: bool = False, verified: bool = False) -> Lifecycle:
        st, local = self._loc(record_id)
        led = st.ledgers[local]
        o = led.opinion(self.cfg.epistemics.W)
        new = next_lifecycle(LifecycleInputs(
            st.lifecycle_of(local), led.n_roots_supporting(), o.b, o.d, led.e_plus, led.e_minus,
            int(st.col("usage")[local]), verified, unresolved_contradiction, contradiction_resolved,
        ), self.cfg)
        if new != st.lifecycle_of(local):
            self.set_lifecycle(record_id, new)
        return new

    def set_state(self, record_id: int, state: EpistemicState | None) -> None:
        st, local = self._loc(record_id)
        st.set_state(local, state)

    def touch(self, record_id: int, value: float = 0.0, cost: float = 0.0) -> None:
        st, local = self._loc(record_id)
        st.touch(local, time.time(), value, cost)

    def tombstone(self, record_id: int, reason: str) -> None:
        st, local = self._loc(record_id)
        self.set_lifecycle(record_id, Lifecycle.DEPRECATED)
        st.meta[local]["tombstone_reason"] = reason
        self.wal.append({"op": "tombstone", "record_id": record_id, "reason": reason})

    def link(self, src: int, dst: int, etype: str, weight: float = 1.0) -> None:
        self.assoc.link(src, dst, etype, weight)

    # --- retrieval (04 §5) ----------------------------------------------------------------------------
    def retrieve(self, query: np.ndarray, kind: RecordKind | None = None, context: list[np.ndarray] | None = None,
                 k: int | None = None, stores: Iterable[int] | None = None, use_signature: bool = False) -> list[tuple[int, float]]:
        """Stage-1 retrieval: banded candidates → context filter → ``α·ov/B + (1−α)·cos`` (04 §5.1).

        ``query`` may be a dense vector (projected) or a code.
        """
        m = self.cfg.memory
        k = k or m.k_ret
        q = np.asarray(query)
        if q.dtype == np.uint8 and q.shape == (self.cs.B,):
            code, qkey = q, self.cs.key(self.cs.embed(q))
        else:
            dense = normalize(q)
            code, qkey = self.cs.project(dense), self.cs.key(dense)
        out: list[tuple[int, float]] = []
        for sid in (stores if stores is not None else self.stores):
            st, idx = self.stores[sid], self.indices[sid]
            if m.index_backend == "dense_ann" and not use_signature:
                locs, _ = self._dense_index.search(qkey, st.col("key"), st.col("alive"), m.cand_max)
            else:
                locs, _ = (idx.signature if use_signature else idx.content).candidates(code, m.cand_max)
            if len(locs) == 0:
                continue
            alive = st.col("alive")[locs]
            locs = locs[alive]
            if kind is not None:
                locs = locs[st.col("kind")[locs] == list(RecordKind).index(kind)]
            if context:
                locs = np.array([l for l in locs if self._context_ok(st, int(l), context)], dtype=np.int64)
            if len(locs) == 0:
                continue
            codes = st.col("signature" if use_signature else "content")[locs]
            ov = (codes == code[None, :]).sum(axis=1) / self.cs.B
            cos = st.col("key")[locs].astype(np.float32) @ qkey
            score = m.alpha_score * ov + (1 - m.alpha_score) * cos
            out.extend((make_record_id(sid, int(l)), float(s)) for l, s in zip(locs, score))
        out.sort(key=lambda t: -t[1])
        return out[:k]

    def _context_ok(self, st: RecordStore, local: int, context: list[np.ndarray]) -> bool:
        n = int(st.col("n_context")[local])
        if n == 0:
            return True
        rec_ctx = st.col("context")[local, :n]
        return any(int((rc == qc).sum()) >= self.cs.B // 2 for rc in rec_ctx for qc in context)

    def retrieve_pattern(self, subject: str | None = None, relation: str | None = None, object_key: str | None = None,
                         qualifiers: Qualifiers | None = None, include_dead: bool = False) -> list[int]:
        """Exact structured lookup (04 §5.2) filtered by qualifier compatibility."""
        out: list[int] = []
        for sid, idx in self.indices.items():
            st = self.stores[sid]
            for rid in idx.triples.facts(subject, relation, object_key):
                local = split_record_id(rid)[1]
                if not include_dead and not st.col("alive")[local]:
                    continue
                payload: EngramPayload = st.payloads[local]
                if qualifiers is not None and not payload.qualifiers.compatible(qualifiers):
                    continue
                out.append(rid)
        return out

    def find_open_questions(self, subject: str, relation: str) -> list[int]:
        return list(self.open_questions.get((subject, relation), []))

    def link_surface(self, surface: str) -> list[str]:
        """Alias lookup: exact first, then fuzzy above ``θ_link`` (02 §7.1)."""
        exact = self.aliases.lookup(surface)
        if exact:
            return sorted(exact)
        return [sid for sid, s in self.aliases.fuzzy(surface, self.cfg.memory.theta_link)]

    def cleanup(self, x: np.ndarray, candidates: list[int]) -> np.ndarray:
        """Modern-Hopfield completion restricted to a candidate set (04 §5.4)."""
        if not candidates:
            return normalize(x)
        V = np.stack([self.record(r).dense for r in candidates]).astype(np.float32)
        K = np.stack([self.cs.key(v) for v in V])
        beta, steps = self.cfg.memory.hopfield_beta, self.cfg.memory.t_clean
        y = normalize(np.asarray(x, dtype=np.float32))
        for _ in range(steps):
            a = K @ self.cs.key(y) * beta
            a = np.exp(a - a.max())
            y = normalize((a / a.sum()) @ V)
        return y

    def spread(self, seeds: dict[int, float]) -> dict[int, float]:
        m = self.cfg.memory
        return self.assoc.spread(seeds, m.edge_gains, m.spread_gamma, m.h_spread, m.k_spread)

    # --- persistence ---------------------------------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for sid, st in self.stores.items():
            st.save(directory / f"store_{sid}")
        state = {
            "region_names": {str(k): v for k, v in self.region_names.items()},
            "sources": {k: asdict(v) for k, v in self.sources.items()},
            "links": self.assoc.to_rows(),
            "sketch": {"m": self.sketch.m, "k": self.sketch.k, "n_added": self.sketch.n_added},
        }
        (directory / "memory.json").write_text(json.dumps(state), encoding="utf-8")
        np.save(directory / "sketch.npy", self.sketch.counters)

    @staticmethod
    def load(directory: str | Path, config: BuildConfig, interface: InterfaceLayer) -> "MemorySystem":
        directory = Path(directory)
        ms = MemorySystem(config, interface)
        state = json.loads((directory / "memory.json").read_text(encoding="utf-8"))
        ms.stores, ms.indices = {}, {}
        for sid_text, name in state["region_names"].items():
            sid = int(sid_text)
            st = RecordStore.load(directory / f"store_{sid}")
            ms.stores[sid] = st
            ms.region_names[sid] = name
            ms.indices[sid] = StoreIndices(BandedIndex(ms.cs.band_layout, ms.cs.L), BandedIndex(ms.cs.band_layout, ms.cs.L))
            idx = ms.indices[sid]
            if st.n:
                idx.content.add_many(np.arange(st.n), st.col("content"))
                sig = np.nonzero(st.col("has_signature"))[0]
                if len(sig):
                    idx.signature.add_many(sig, st.col("signature")[sig])
            for local in range(st.n):
                rid = make_record_id(sid, local)
                kind, payload = st.kind_of(local), st.payloads[local]
                if kind == RecordKind.ENTITY:
                    ms.symbols[payload.entity_id] = rid
                    for surface in (payload.name, *payload.aliases):
                        ms.aliases.add(surface, payload.entity_id)
                elif kind == RecordKind.ENGRAM:
                    idx.triples.add_fact(rid, payload.subject, payload.relation, payload.object.key())
                elif kind == RecordKind.OPEN_QUESTION:
                    ms.open_questions.setdefault((payload.subject, payload.relation), []).append(rid)
        ms.sources = {k: SourceInfo(**v) for k, v in state["sources"].items()}
        for s, d, t, w in state["links"]:
            ms.assoc.link(int(s), int(d), t, float(w))
        ms.sketch.counters = np.load(directory / "sketch.npy")
        ms.sketch.n_added = state["sketch"]["n_added"]
        return ms
