"""Model-side ingestion and triage (02 §9–11).

Each SEF record is encoded, triaged (KNOWN-SAME / CONFLICT / NOVEL / NOISE) and
turned into memory operations with evidence initialized by 02 §10.3.  Record kinds
whose consumers are implemented in later milestones (concepts, procedures, causal
schemas, skills, tasks) are counted as ``deferred``, never silently dropped.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from srm.config.build_config import BuildConfig
from srm.core.evidence import EvidenceLedger, initial_evidence, modality_factor
from srm.data.sef import (
    ContradictionRecord,
    EntityRecord,
    FactRecord,
    KnownUnknownRecord,
    PropertyDefRecord,
    RecordBase,
    RelationDefRecord,
    SourceRecord,
)
from srm.ingest.encoder import SymbolEncoder
from srm.interface.messages import Lifecycle, Qualifiers, RecordKind
from srm.interface.values import Value
from srm.memory.payloads import EngramPayload, EntityPayload, OpenQuestionPayload
from srm.memory.system import HIPPOCAMPUS, MemorySystem

KIND_PRIORITY = {
    "source": 0, "property_def": 1, "relation_def": 1, "entity": 2, "segment": 3,
    "fact": 4, "known_unknown": 4, "contradiction": 5,
}
NON_ASSERTIONS = ("hypothetical", "fictional", "unknown_declared")


@dataclass
class IngestReport:
    outcomes: Counter = field(default_factory=Counter)
    deferred: Counter = field(default_factory=Counter)
    contradictions: list[tuple[int, int, str]] = field(default_factory=list)

    def __str__(self) -> str:
        return f"outcomes={dict(self.outcomes)} deferred={dict(self.deferred)}"


class Ingestor:
    def __init__(self, memory: MemorySystem, config: BuildConfig) -> None:
        self.mem = memory
        self.cfg = config
        self.enc = SymbolEncoder(memory.interface)
        self.sef_map: dict[str, int] = {}
        self.contested_pairs: set[frozenset[int]] = set()
        self.report = IngestReport()

    # --- driver -------------------------------------------------------------------------------
    def ingest(self, records: Iterable[RecordBase]) -> IngestReport:
        ordered = sorted(enumerate(records), key=lambda t: (KIND_PRIORITY.get(t[1].kind, 6), t[0]))
        for _, rec in ordered:
            handler = getattr(self, f"_on_{rec.kind}", None)
            if handler is None:
                self.report.deferred[rec.kind] += 1
                continue
            handler(rec)
        return self.report

    # --- registries and symbols ---------------------------------------------------------------
    def _on_source(self, r: SourceRecord) -> None:
        self.mem.register_source(r.record_id, r.source_type, r.trust_class, r.root_source_id, r.privacy_scope, r.title)
        self.report.outcomes["source"] += 1

    def _on_relation_def(self, r: RelationDefRecord) -> None:
        self.mem.interface.relations.add_relation(
            r.relation_id, r.name, r.arity, r.arg_types, r.cardinality, r.temporal, r.symmetric, r.inverse_of,
            r.description,
        )
        self.report.outcomes["relation_def"] += 1

    def _on_property_def(self, r: PropertyDefRecord) -> None:
        self.mem.interface.properties.add_property(
            r.property_id, r.name, r.property_kind, r.value_type, r.unit, r.enum_values,
            tuple(r.range) if r.range else None, r.log_scale, r.description,
        )
        self.report.outcomes["property_def"] += 1

    def _on_segment(self, r: RecordBase) -> None:
        self.report.deferred["segment"] += 1  # parser training input (S2), not memory content

    def _on_entity(self, r: EntityRecord) -> None:
        if r.entity_id in self.mem.symbols:
            self.report.outcomes["entity_known"] += 1
            return
        src = self.mem.source(r.source_id or "")
        ledger = EvidenceLedger()
        ledger.add(src.root_source_id, initial_evidence(self.cfg.epistemics.kappa, src.trust, r.extraction.confidence, 1.0), 0.0)
        rid = self.mem.write(
            RecordKind.ENTITY, r.entity_id, self.enc.entity(r.entity_id),
            EntityPayload(r.entity_id, r.canonical_name, r.entity_type, list(r.aliases), r.provisional),
            ledger, [{"source_id": r.source_id, "root": src.root_source_id, "sef": r.record_id}],
            target=HIPPOCAMPUS,
        )
        self.sef_map[r.record_id] = rid
        self.report.outcomes["entity"] += 1

    # --- facts (02 §9.1, §10, §11) ------------------------------------------------------------------
    def _on_fact(self, r: FactRecord) -> None:
        e = self.cfg.epistemics
        ep = r.epistemic
        if r.extraction.confidence < e.theta_noise:
            self.report.outcomes["noise"] += 1
            return
        if ep.modality in NON_ASSERTIONS:
            self.report.outcomes[f"non_assertion:{ep.modality}"] += 1
            return
        src = self.mem.source(r.source_id or "")
        mu = modality_factor(ep.modality, ep.hedge, e.modality_factors, e.hedge_slope)
        strength = initial_evidence(e.kappa, src.trust, r.extraction.confidence, mu)
        attributed = ep.modality == "reported"
        value = Value(r.object.type, tuple(r.object.value) if isinstance(r.object.value, list) else r.object.value,
                      r.object.unit, float(r.object.tolerance or 0.0))
        q = dict(r.qualifiers)
        if r.context.time is not None and "time" not in q:
            q["time"] = list(r.context.time)
        for name in ("version", "world_id"):
            if getattr(r.context, name) is not None and name not in q:
                q[name] = getattr(r.context, name)
        quals = Qualifiers.from_dict(q)
        polarity = "negative" if (ep.polarity == "negative" or ep.modality == "negated") else "positive"
        prov = {"source_id": r.source_id, "root": src.root_source_id, "trust": src.trust,
                "strength": strength, "sef": r.record_id, "modality": ep.modality}

        existing = self.mem.retrieve_pattern(r.subject, r.relation, qualifiers=quals)
        same, conflicts = [], []
        for rid in existing:
            p: EngramPayload = self.mem.record(rid).payload
            if (p.attribution is not None) != attributed or (attributed and p.attribution != ep.attribution):
                continue
            if p.object.matches(value) and p.polarity == polarity:
                same.append(rid)
            elif attributed:
                continue  # different reports about what a source said do not contradict each other
            elif p.object.matches(value) and p.polarity != polarity:
                conflicts.append((rid, "polarity_conflict"))
            elif self.mem.interface.relations.is_functional(r.relation) and p.polarity == polarity == "positive":
                conflicts.append((rid, "value_conflict"))

        if same:  # KNOWN-SAME: corroboration (new root) or no-op (same root)
            rid = same[0]
            if attributed:
                p = self.mem.record(rid).payload
                p.attributed_evidence = max(
                    p.attributed_evidence, initial_evidence(e.kappa, src.trust, r.extraction.confidence, 1.0))
            else:
                self.mem.add_evidence(rid, src.root_source_id, strength, 0.0)
            self.mem.record(rid).provenance.append(prov)
            self.sef_map[r.record_id] = rid
            self.report.outcomes["known_same"] += 1
            self._reassess_conflicts(rid)
            self.mem.update_lifecycle(rid, unresolved_contradiction=self._is_contested(rid))
            return

        ledger = EvidenceLedger()
        if not attributed:
            ledger.add(src.root_source_id, strength, 0.0)
        payload = EngramPayload(
            subject=r.subject, relation=r.relation, object=value, qualifiers=quals, polarity=polarity,
            modality=ep.modality, attribution=ep.attribution if attributed else None,
            attributed_evidence=initial_evidence(e.kappa, src.trust, r.extraction.confidence, 1.0) if attributed else 0.0,
            hedge=ep.hedge, sef_record_ids=[r.record_id],
        )
        rid = self.mem.write(RecordKind.ENGRAM, r.record_id, self.enc.fact(r.subject, r.relation, value), payload,
                             ledger, [prov], target=HIPPOCAMPUS)
        self.sef_map[r.record_id] = rid
        self.report.outcomes["novel"] += 1
        for other, nature in conflicts:
            self._register_conflict(rid, other, nature)
        self.mem.update_lifecycle(rid, unresolved_contradiction=self._is_contested(rid))

    def _register_conflict(self, a: int, b: int, nature: str) -> None:
        self.mem.link(a, b, "contradicts")
        self.mem.link(b, a, "contradicts")
        self.report.contradictions.append((a, b, nature))
        self.report.outcomes["conflict"] += 1
        self.contested_pairs.add(frozenset((a, b)))
        self._reassess_conflicts(a)

    def _is_contested(self, rid: int) -> bool:
        return any(rid in pair for pair in self.contested_pairs)

    def _reassess_conflicts(self, rid: int) -> None:
        """Resolve a contradiction when the evidence gap reaches θ_trust_gap (02 §11)."""
        e = self.cfg.epistemics
        for pair in [p for p in self.contested_pairs if rid in p]:
            a, b = tuple(pair)
            la, lb = self.mem.record(a).ledger, self.mem.record(b).ledger
            gap = la.e_plus - lb.e_plus
            if abs(gap) >= e.theta_trust_gap * e.kappa:
                strong, weak = (a, b) if gap > 0 else (b, a)
                strong_plus = self.mem.record(strong).ledger.e_plus
                self.mem.add_evidence(weak, f"contradiction:{strong}", 0.0, strong_plus)
                self.contested_pairs.discard(pair)
                for x in (a, b):
                    self.mem.update_lifecycle(x, unresolved_contradiction=self._is_contested(x),
                                              contradiction_resolved=not self._is_contested(x))
                self.report.outcomes["conflict_resolved"] += 1
            else:
                for x in (a, b):
                    self.mem.update_lifecycle(x, unresolved_contradiction=True)

    # --- declared unknowns and explicit contradictions ----------------------------------------------
    def _on_known_unknown(self, r: KnownUnknownRecord) -> None:
        for atom in r.pattern.atoms:
            if atom.subject.startswith("?"):
                continue
            src = self.mem.source(r.source_id or "")
            rid = self.mem.write(
                RecordKind.OPEN_QUESTION, r.record_id + atom.relation, self.enc.entity(atom.subject),
                OpenQuestionPayload(atom.subject, atom.relation, r.scope, r.as_of, r.note, r.source_id or ""),
                provenance=[{"source_id": r.source_id, "root": src.root_source_id, "sef": r.record_id}],
                target=HIPPOCAMPUS,
            )
            self.sef_map[r.record_id] = rid
            self.report.outcomes["open_question"] += 1

    def _on_contradiction(self, r: ContradictionRecord) -> None:
        ids = [self.sef_map[c] for c in r.claims if isinstance(c, str) and c in self.sef_map]
        if len(ids) < 2:
            self.report.deferred["contradiction_unresolved_refs"] += 1
            return
        if r.resolution and r.resolution.preferred in self.sef_map:
            keep = self.sef_map[r.resolution.preferred]
            for other in ids:
                if other != keep:
                    self.mem.add_evidence(other, f"resolution:{r.record_id}", 0.0,
                                          self.mem.record(keep).ledger.e_plus)
                    self.mem.update_lifecycle(other)
        else:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    self.contested_pairs.add(frozenset((ids[i], ids[j])))
                    self.mem.update_lifecycle(ids[i], unresolved_contradiction=True)
                    self.mem.update_lifecycle(ids[j], unresolved_contradiction=True)
        self.report.outcomes["contradiction_record"] += 1

    def contested(self, rid: int) -> bool:
        return self._is_contested(rid) or self.mem.record(rid).lifecycle == Lifecycle.CONTESTED
