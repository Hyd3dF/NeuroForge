# 14 — Implementation Roadmap (Python)

## 1. Technology stack (D-001 … D-006 in `16_decision_log.md`)

| Concern | Choice |
|---|---|
| Language | Python ≥ 3.11, fully type-annotated (mypy strict on core packages) |
| Tensors / training | PyTorch 2.x |
| Arrays / tables | NumPy; Apache Arrow / Parquet (pyarrow) |
| Weight files | safetensors |
| Schemas and validation | pydantic v2 for configs, SEF records and Manifest models |
| Canonical JSON | RFC 8785 JCS implementation (small internal module) + hashlib SHA-256 |
| Math tools | SymPy; pint (units) |
| Code parsing | Python `ast`; custom DSL parser |
| Testing | pytest, hypothesis (property-based), coverage |
| Optional | FAISS (dense ANN backend), zstandard |

**Dependency rule:** core packages (`interface`, `memory`, `core`, `prediction`, `components`) must not depend on optional packages.

## 2. Planned code layout (to be created in M0; documented here only)

```
srm/
  config/        build-config models, profiles (F0), validation rules (09 §3)
  data/          SEF models, readers/writers, converters (text, code, math, table), canonicalization,
                 synthetic generators (objectworld, dslworld, factstream, mathworld), dataset manifests
  interface/     code space + algebra, projection/embedding, registries, property basis,
                 message types, signature encoder, codec, ABI hash
  perception/    chunker, encoder, L0 predictor, recognition, parsers (text, code, math, table), goal intake
  memory/        record store (columnar), region/page/tier, hippocampus, indices (banded, signature,
                 triple, alias, dense_ann), sketch, association field, lifecycle, WAL
  core/          workspace, evidence algebra, tms, channels, primitives/ (one module per primitive),
                 composer (graph, synthesis, policy), selector, error_monitor (verifier, checks),
                 simulator, concept_formation, skills (shared core + low-rank modulation)
  prediction/    prediction records, taint, promotion, envelopes, support, decision procedure, answer record
  control/       heart, gate, modulators, subconscious processes
  mouth/         utterance planner, template renderer, unparsers, neural renderer, articulators, attribution
  body/          tool interface, sandbox, dsl interpreter, test runner, sympy tool, units tool
  learning/      bootstrap stages S0–S7, online rules, local buffers, sleep engine, compilation
  components/    component registry, contracts, ids, fingerprints, manifest builder, package io,
                 trace recorder, closure, retrain pipeline, gates, surgery, rollback, compatibility
  runtime/       engine (beat loop), query lifecycle, sessions, accounting, degradation
  eval/          benchmarks, metrics, baselines adapters, reports
  cli/           commands: build, generate-data, bootstrap, ingest, query, sleep, manifest, retrain, rollback, eval
tests/           unit/, property/, integration/, golden/ (golden traces), regression/
docs/plan/       this plan
```

## 3. Coding conventions (binding)
1. **Config-driven sizes:** no literal capacity, dimension or threshold in code. Everything is read from the Build Configuration object (validated), with F0 defaults in `srm/config/profiles/f0`.
2. **Messages only:** components exchange only the message types from `03` §5. No private tensors cross component boundaries.
3. **Every component** implements the component protocol: `contract()`, `load(shard)`, `save()`, `fingerprint(probe)`, `local_objective()` (if trainable), `validate()`.
4. **Determinism:** global and per-component seeds; generators are deterministic given a seed.
5. **Epistemic invariants are asserted in code** (taint propagation, fact-placement rule, KNOWN-class requires an untainted grounded path), with tests that must never be skipped.
6. **Accounting hooks** on every Heart-funded operation (FLOPs estimate, bytes, wall time).
7. **Traceability:** every claim carries its producer ID and premises; every record carries provenance.

## 4. Milestones

Each milestone ends with its acceptance tests passing in CI. Order is chosen to reach a **vertical slice** (a factual QA + UNKNOWN demo) early.

| # | Milestone | Deliverables | Acceptance |
|---|---|---|---|
| **M0** | Scaffolding & config | Package skeleton, Build Configuration models + validation + F0 profile, canonical JSON + hashing, CI (lint, type check, tests) | Config validation tests; JCS conformance tests |
| **M1** | Interface layer + SEF + generators | Code-space algebra (exact ops), registries, message types, ABI hash; SEF models/readers/writers + validation; ObjectWorld, FactStream, DSLWorld, MathWorld generators (gold SEF + renderings) | Algebra property tests (bind/unbind inverse, overlap statistics); SEF schema round-trip; generator determinism and split-leakage tests |
| **M2** | Memory | Columnar record store, regions/pages, hippocampus, banded content index, triple index, alias index, sketch, association field, lifecycle, WAL | Index recall/false-positive tests against the formulas in `04` §4.1; sketch no-false-negative property test; lifecycle transition tests |
| **M3** | Epistemics core | Workspace, evidence algebra, derived belief, TMS retraction, nogoods, channels, taint/promotion, Support, decision procedure, Answer Record | Unit tests per formula; **taint-leak invariant tests**; decision-procedure table tests (each state reachable) |
| **M4** | Primitives (A-class) + Composer (symbolic) + Body | All A-class primitives, UNIFY/QUERY_JOIN, ALIGN/ABSTRACT with heuristic affinity, COMPARE, circuit graph + executor, type-directed synthesis (uniform policy), Body tools (DSL, Python sandbox, tests, SymPy, units) | DSL tasks solved by synthesis on small depth; ALIGN correctness on synthetic pairs; sandbox isolation tests |
| **M5** | Ingestion + intake + Mouth (templates, unparsers) | Triage (`02` §9), deterministic parsers (code/math/table), controlled-language parser (rule-based first), goal intake, utterance planner, template renderer, unparsers | **Vertical slice:** ingest FactStream → answer questions with KNOWN / UNKNOWN-ABSENT / CONTESTED / INSUFFICIENT correctly; DSL program synthesized, tested and rendered |
| **M6** | Control | Heart (bids, allocation, credit, caps, tiers), Gate, Modulators, Subconscious processes, Selector (heuristic VOC first) | Budget/caps invariants; beat-order tests; active-fraction accounting tests |
| **M7** | Learning: CFE, online rules, sleep, compilation | Concept Formation Engine; Hebbian; verification-as-teacher buffers; envelopes; sleep phases 1–10; skill compilation (shared skill core) | Few-shot concept tests on ObjectWorld held-out concepts; consolidation invariants; compilation acceptance gate tests |
| **M8** | Component system | Component registry, IDs, contracts, fingerprints, Manifest builder, package IO (`srmpkg-1`), trace recorder, closure, retrain pipeline, gates, surgery, rollback, compatibility | ID stability tests (retrain → same logical ID; interface change → new ID); Manifest Merkle verification; end-to-end articulator retrain with rollback; refused operations without a Manifest |
| **M9** | Learned components | S1 interface training (projection, Property Basis, SetEncoder); S2 perception encoder + learned text parser; S3 ALIGN affinity, EXPLAIN heads, verifier, dynamics skills; S4 Composer policy + Selector actor-critic; S5 neural renderer + articulators | Stage acceptance criteria (`08` §2) |
| **M10** | Evaluation harness + baselines | Benchmark suite (`12` §2), metrics (`12` §4), baseline adapters (Transformer, RAG, RAG+adapters+agent, continual fine-tune, PEER/memory-layer, TRM-style) | Reproducible reports with seeds |
| **M11** | SRM-F0 experiments | Library sweep, few-shot, compositional, epistemic, retraining-cost, forgetting, ablations | Continue/stop decision per `12` §5 |

**Vertical-slice priority:** M0 → M1 → M2 → M3 → M5 (factual part) can produce the first demo (ingest facts; answer with correct epistemic states including UNKNOWN) before any neural training. That validates the epistemic architecture early.

## 5. Test strategy
- **Unit tests:** every formula and algorithm in the plan has at least one test referencing its section (e.g. `test_banded_recall_matches_formula  # 04 §4.1`).
- **Property-based tests:** code algebra, sketch, unification, taint propagation, ID stability, Manifest canonicalization.
- **Integration tests:** query lifecycle per worked trace (`11` §6).
- **Golden traces:** stored Answer Records for fixed seeds; diffs reviewed on change.
- **Regression suites:** the same suites used by the retraining gates (`10` §5), so development and surgery share one test infrastructure.
- **Invariant tests (never skipped):** taint-leak = 0; fact-placement rule; sketch has no false negatives; machinery floor present; Manifest root verifies.

## 6. Documentation upkeep
- Each milestone updates `15_readiness_review.md` §6 (status table).
- Any deviation from the plan → a `16_decision_log.md` entry **before** merging the code that deviates.
