"""FactStream generator (02 §14.3).

Entities with attributes and relations (some temporal) are emitted as a
time-ordered stream of SEF facts from sources of varied trust, including
corroboration, copies sharing a root source, hedged and reported statements,
injected contradictions, declared unknowns, withheld entities and omitted
attributes.  Every question carries the gold decision state implied by the
plan's evidence rules (06 §6), so the stream doubles as a calibration suite.
"""

from __future__ import annotations

from dataclasses import dataclass

from srm.data.generators.common import GenContext
from srm.data.io import Dataset
from srm.data.sef import (
    Context,
    EntityRecord,
    Epistemic,
    FactPattern,
    FactRecord,
    KnownUnknownRecord,
    PatternAtom,
    QuestionRecord,
    RelationDefRecord,
    TypedValue,
)

VERSION = "0.1"
CATEGORY = "synthetic/factstream"

# relation id → (subject type, object kind, functional, temporal, unit)
RELATIONS: dict[str, tuple[str, str, bool, bool, str | None]] = {
    "rel:capital": ("country", "city", True, False, None),
    "rel:located_in": ("city", "country", True, False, None),
    "rel:population": ("country", "int", True, True, None),
    "rel:area_km2": ("country", "float", True, False, "km2"),
    "rel:founded_year": ("company", "int", True, False, None),
    "rel:headquarters": ("company", "city", True, False, None),
    "rel:ceo": ("company", "person", True, False, None),
    "rel:born_in": ("person", "city", True, False, None),
    "rel:birth_year": ("person", "int", True, False, None),
}

# Emission scenario → gold decision state for questions about that (entity, relation).
SCENARIOS: dict[str, str] = {
    "known_strong": "KNOWN",        # one curated/reference source
    "known_web2": "KNOWN",          # two independent web sources
    "resolved_conflict": "KNOWN",   # curated truth + low-trust contradiction (trust gap)
    "web_copy": "INSUFFICIENT",     # web source + a copy sharing its root
    "weak": "INSUFFICIENT",         # one hedged forum statement
    "reported": "INSUFFICIENT",     # only an attributed (reported) claim
    "contested": "CONTESTED",       # two equally trusted sources disagree
    "declared_unknown": "UNKNOWN-DECLARED",
    "omitted": "UNKNOWN-ABSENT",
}
SCENARIO_WEIGHTS: dict[str, float] = {
    "known_strong": 0.40, "known_web2": 0.08, "resolved_conflict": 0.07, "web_copy": 0.06, "weak": 0.06,
    "reported": 0.05, "contested": 0.08, "declared_unknown": 0.05, "omitted": 0.15,
}


@dataclass
class _Truth:
    subject: str
    relation: str
    value: TypedValue
    interval: tuple[float, float] | None = None


def generate(dataset_id: str = "fs", seed: int = 0, n_countries: int = 20, withheld_fraction: float = 0.1,
             questions_per_fact: float = 1.0) -> Dataset:
    ctx = GenContext(dataset_id, seed, "factstream", VERSION)
    rng = ctx.rng
    ds = ctx.dataset

    # sources (02 §4.1); web_copy_* share the root of web_0
    src = {
        "kb": ctx.source("kb", "curated_kb", category=CATEGORY),
        "ref1": ctx.source("ref1", "reference_doc", category=CATEGORY),
        "ref2": ctx.source("ref2", "reference_doc", category=CATEGORY),
        "web_a": ctx.source("web_a", "web_document", category=CATEGORY),
        "web_b": ctx.source("web_b", "web_document", category=CATEGORY),
        "forum": ctx.source("forum", "forum", category=CATEGORY),
    }
    src["web_a_copy"] = ctx.source("web_a_copy", "web_document", root=src["web_a"], category=CATEGORY)

    for rel, (stype, okind, functional, temporal, unit) in RELATIONS.items():
        ctx.add(RelationDefRecord(
            record_id=ctx.rid("relation_def"), source_id=src["kb"], data_category=CATEGORY,
            extraction=ctx.extraction(), relation_id=rel, name=rel.split(":", 1)[1],
            arg_types=[f"concept:{stype}", okind if okind in ("int", "float") else f"concept:{okind}"],
            cardinality="functional" if functional else "multi", temporal=temporal,
        ))

    # --- world ---------------------------------------------------------------------------------
    entities: dict[str, list[tuple[str, str]]] = {"country": [], "city": [], "company": [], "person": []}
    withheld: list[tuple[str, str]] = []

    def new_entity(etype: str) -> tuple[str, str]:
        name = ctx.names.make()
        eid = f"{dataset_id}:ent:{etype}_{name.lower()}"
        return eid, name

    for _ in range(n_countries):
        entities["country"].append(new_entity("country"))
        for _c in range(3):
            entities["city"].append(new_entity("city"))
        entities["company"].append(new_entity("company"))
        entities["person"].append(new_entity("person"))
    n_withheld = max(1, int(round(withheld_fraction * n_countries)))
    for etype in ("country", "company", "person"):
        for _ in range(n_withheld):
            withheld.append((etype, new_entity(etype)[1]))

    truths: list[_Truth] = []
    cities = entities["city"]
    for i, (cid, _cname) in enumerate(entities["country"]):
        my_cities = cities[3 * i : 3 * i + 3]
        truths.append(_Truth(cid, "rel:capital", TypedValue(type="entity_ref", value=my_cities[0][0])))
        for city_id, _ in my_cities:
            truths.append(_Truth(city_id, "rel:located_in", TypedValue(type="entity_ref", value=cid)))
        pop1 = int(rng.integers(100_000, 50_000_000))
        pop2 = int(pop1 * (1.0 + rng.uniform(0.05, 0.4)))
        truths.append(_Truth(cid, "rel:population", TypedValue(type="int", value=pop1), (1990.0, 2010.0)))
        truths.append(_Truth(cid, "rel:population", TypedValue(type="int", value=pop2), (2010.0, 2030.0)))
        truths.append(_Truth(cid, "rel:area_km2", TypedValue(type="float", value=round(float(rng.uniform(1e3, 2e6)), 1),
                                                                unit="km2", tolerance=0.5)))
    for j, (coid, _n) in enumerate(entities["company"]):
        truths.append(_Truth(coid, "rel:founded_year", TypedValue(type="int", value=int(rng.integers(1850, 2020)))))
        truths.append(_Truth(coid, "rel:headquarters", TypedValue(type="entity_ref", value=ctx.choice(cities)[0])))
        truths.append(_Truth(coid, "rel:ceo", TypedValue(type="entity_ref", value=entities["person"][j][0])))
    for pid, _n in entities["person"]:
        truths.append(_Truth(pid, "rel:born_in", TypedValue(type="entity_ref", value=ctx.choice(cities)[0])))
        truths.append(_Truth(pid, "rel:birth_year", TypedValue(type="int", value=int(rng.integers(1940, 2000)))))

    # --- entity records --------------------------------------------------------------------------
    name_of: dict[str, str] = {}
    for etype, items in entities.items():
        for eid, name in items:
            name_of[eid] = name
            ctx.add(EntityRecord(
                record_id=ctx.rid("entity"), source_id=src["kb"], data_category=CATEGORY,
                extraction=ctx.extraction(), entity_id=eid, canonical_name=name,
                entity_type=f"concept:{etype}", aliases=[name.upper()],
            ))

    # --- scenarios -------------------------------------------------------------------------------
    names = list(SCENARIO_WEIGHTS)
    probs = [SCENARIO_WEIGHTS[n] for n in names]
    stream: list[tuple[float, FactRecord | KnownUnknownRecord]] = []
    questions: list[tuple[_Truth, str]] = []

    def wrong_value(t: _Truth) -> TypedValue:
        v = t.value
        if v.type == "entity_ref":
            pool = [e for e, _ in entities["city"] + entities["country"] + entities["person"] if e != v.value]
            return TypedValue(type="entity_ref", value=ctx.choice(pool))
        if v.type == "int":
            return TypedValue(type="int", value=int(v.value) + int(rng.integers(1, 50)) * (1 if rng.random() < 0.5 else -1))
        return TypedValue(type="float", value=round(float(v.value) * float(rng.uniform(1.2, 2.0)), 1),
                          unit=v.unit, tolerance=v.tolerance)

    def fact(t: _Truth, value: TypedValue, source: str, ep: Epistemic | None = None) -> FactRecord:
        q = {"time": list(t.interval)} if t.interval else {}
        return FactRecord(
            record_id=ctx.rid("fact"), source_id=src[source], data_category=CATEGORY,
            extraction=ctx.extraction(), epistemic=ep or Epistemic(), subject=t.subject,
            relation=t.relation, object=value, qualifiers=q,
        )

    hop_state: dict[tuple[str, str], tuple[str, _Truth]] = {}
    for t in truths:
        sc = names[int(rng.choice(len(names), p=probs))]
        # temporal facts are always strongly known (they test interval handling, 02 §11)
        if t.interval is not None:
            sc = "known_strong"
        emitted: list[FactRecord | KnownUnknownRecord] = []
        if sc == "known_strong":
            emitted.append(fact(t, t.value, ctx.choice(["kb", "ref1"])))
        elif sc == "known_web2":
            emitted += [fact(t, t.value, "web_a"), fact(t, t.value, "web_b")]
        elif sc == "resolved_conflict":
            emitted += [fact(t, t.value, "kb"), fact(t, wrong_value(t), "forum")]
        elif sc == "web_copy":
            emitted += [fact(t, t.value, "web_a"), fact(t, t.value, "web_a_copy")]
        elif sc == "weak":
            emitted.append(fact(t, t.value, "forum", Epistemic(modality="hedged", hedge=0.6)))
        elif sc == "reported":
            emitted.append(fact(t, t.value, "web_b", Epistemic(modality="reported", attribution=f"{dataset_id}:ent:analyst")))
        elif sc == "contested":
            emitted += [fact(t, t.value, "ref1"), fact(t, wrong_value(t), "ref2")]
        elif sc == "declared_unknown":
            emitted.append(KnownUnknownRecord(
                record_id=ctx.rid("known_unknown"), source_id=src["ref1"], data_category=CATEGORY,
                extraction=ctx.extraction(),
                pattern=FactPattern(atoms=[PatternAtom(subject=t.subject, relation=t.relation, object="?x")]),
                note="not recorded",
            ))
        for rec in emitted:
            stream.append((float(rng.random()), rec))
        if t.interval is None:
            hop_state[(t.subject, t.relation)] = (sc, t)
        if rng.random() < questions_per_fact:
            questions.append((t, sc))

    stream.sort(key=lambda p: p[0])
    for _ts, rec in stream:
        ctx.add(rec)

    # --- questions (02 §4.14) --------------------------------------------------------------------
    def question(subject_name: str, relation: str, gold: str, answer: TypedValue | None,
                 interval: tuple[float, float] | None) -> None:
        when = None
        if interval is not None:
            mid = (interval[0] + interval[1]) / 2.0
            when = (mid, mid + 1.0)
        ctx.add(QuestionRecord(
            record_id=ctx.rid("question"), source_id=src["kb"], data_category="eval/factstream",
            extraction=ctx.extraction(), context=Context(time=when),
            query=f"What is the {relation.split(':', 1)[1].replace('_', ' ')} of {subject_name}?",
            pattern=FactPattern(atoms=[PatternAtom(subject=f"@{subject_name}", relation=relation, object="?x")]),
            gold_state=gold, gold_answer=answer, split="dev",
        ))

    for t, sc in questions:
        gold = SCENARIOS[sc]
        answer = t.value if gold == "KNOWN" else None
        question(name_of[t.subject], t.relation, gold, answer, t.interval)
    for etype, name in withheld:
        rel = next(r for r, spec in RELATIONS.items() if spec[0] == etype)
        question(name, rel, "UNKNOWN-ABSENT", None, None)

    # multi-hop chains (11 §6.1): rendered as nested "the R of the R2 of X"
    chains = [("company", ("rel:ceo", "rel:born_in", "rel:located_in")),
              ("person", ("rel:born_in", "rel:located_in")),
              ("company", ("rel:headquarters", "rel:located_in"))]
    known = {"known_strong", "known_web2", "resolved_conflict"}
    for etype, rels in chains:
        for eid, ename in entities[etype]:
            subject, gold, answer = eid, "KNOWN", None
            for rel in rels:
                sc_t = hop_state.get((subject, rel))
                if sc_t is None:
                    gold = "SKIP"
                    break
                sc, t = sc_t
                if sc == "omitted":
                    gold = "UNKNOWN-ABSENT"
                    break
                if sc not in known:
                    gold = "SKIP"
                    break
                answer = t.value
                subject = str(t.value.value)
            if gold == "SKIP":
                continue
            phrase = " of the ".join(r.split(":", 1)[1].replace("_", " ") for r in reversed(rels))
            atoms, var = [], f"@{ename}"
            for i, rel in enumerate(rels):
                nxt = "?x" if i == len(rels) - 1 else f"?h{i}"
                atoms.append(PatternAtom(subject=var, relation=rel, object=nxt))
                var = nxt
            ctx.add(QuestionRecord(
                record_id=ctx.rid("question"), source_id=src["kb"], data_category="eval/factstream",
                extraction=ctx.extraction(), query=f"What is the {phrase} of {ename}?",
                pattern=FactPattern(atoms=atoms), gold_state=gold,
                gold_answer=answer if gold == "KNOWN" else None, split="dev_multihop",
            ))

    ds.meta = {
        "n_entities": sum(len(v) for v in entities.values()),
        "n_withheld": len(withheld),
        "n_truths": len(truths),
        "scenario_gold": SCENARIOS,
    }
    return ds
