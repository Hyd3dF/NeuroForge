# 15 — Implementation Readiness Review

**Date:** 2026-10-07 · **Plan version:** 1.0 · **Scope reviewed:** SRM-F0 (`12` §1)

## 1. Method
1. **Mechanical check:** every cross-reference (`NN §x.y`) in the plan resolves to an existing section. Result: 0 unresolved references.
2. **Layer-by-layer walk** from the lowest layer (raw data) to the output layer. For each subsystem, check that five things are defined:
   - **(D)** data structures with fields and types;
   - **(A)** algorithms or formulas;
   - **(P)** parameters exposed in the Build Configuration;
   - **(T)** a training objective (if learned) or acceptance tests;
   - **(I)** interfaces (messages in and out).
3. **End-to-end walkthroughs:** a factual query (known and unknown), a novel-object recognition, a DSL programming task, a bootstrap training run, and a component retrain. Each was traced through the plan step by step to find missing links.
4. **Gaps found** were fixed in the plan before this verdict (§3).

## 2. Layer-by-layer checklist (lowest layer → output)

| # | Layer / subsystem | D | A | P | T | I | Where |
|---|---|---|---|---|---|---|---|
| 1 | Raw data → SEF conversion (all modalities in F0 scope) | ✓ | ✓ | ✓ | ✓ (validation gates) | ✓ | `02` §2–8 |
| 2 | SEF record kinds (facts, concepts, procedures, examples, skills, causal, contradictions, known unknowns, uncertainty) | ✓ | — | ✓ | ✓ | ✓ | `02` §3–5 |
| 3 | Uncertainty entry (trust, modality, extraction confidence, evidence init) | ✓ | ✓ | ✓ | ✓ | ✓ | `02` §10 |
| 4 | Synthetic generators + dataset sizes | ✓ | ✓ | ✓ | ✓ | ✓ | `02` §14 |
| 5 | Model-side ingestion / triage | ✓ | ✓ | ✓ | ✓ | ✓ | `02` §9, §11 |
| 6 | Code space + algebra | ✓ | ✓ (exact) | ✓ | ✓ (S1) | ✓ | `03` §2 |
| 7 | Property Basis, registries, message types, ABI hash | ✓ | ✓ | ✓ | ✓ | ✓ | `03` §1, §3–5 |
| 8 | Codec | ✓ | ✓ | ✓ | ✓ | ✓ | `03` §6 |
| 9 | Perception encoder, L0, recognition-by-recall | ✓ | ✓ | ✓ | ✓ (S2) | ✓ | `03` §7 |
| 10 | Structural parsers, goal intake, ImplRepr | ✓ | ✓ | ✓ | ✓ (S2) | ✓ | `03` §8–9 |
| 11 | Record store, payloads, regions, routing, tiers | ✓ | ✓ | ✓ | ✓ | ✓ | `04` §2–3 |
| 12 | Indices (banded, signature, triple, alias) | ✓ | ✓ (recall formula) | ✓ | ✓ | ✓ | `04` §4 |
| 13 | Retrieval, cleanup, Association Field, Hebbian | ✓ | ✓ | ✓ | ✓ | ✓ | `04` §5–6 |
| 14 | Knowledge Sketch | ✓ | ✓ (sizing) | ✓ | ✓ | ✓ | `04` §7 |
| 15 | Hippocampal Index, lifecycle | ✓ | ✓ | ✓ | ✓ | ✓ | `04` §8–9 |
| 16 | Workspace, evidence algebra, derived belief, TMS, channels | ✓ | ✓ | ✓ | ✓ | ✓ | `05` §1–3 |
| 17 | Primitive Basis (signatures + implementation class) | ✓ | ✓ | ✓ | ✓ (S3 for learned parts) | ✓ | `05` §4 |
| 18 | Composer (graph, construction, policy) | ✓ | ✓ | ✓ | ✓ (S4) | ✓ | `05` §5 |
| 19 | Two-stage matching (ALIGN, ABSTRACT) | ✓ | ✓ | ✓ | ✓ (S3) | ✓ | `05` §6 |
| 20 | Selector (VOC incl. heuristic fallback, stop rule) | ✓ | ✓ | ✓ | ✓ (S4) | ✓ | `05` §7 |
| 21 | Error Monitor / verifier | ✓ | ✓ | ✓ | ✓ (S3) | ✓ | `05` §8 |
| 22 | Concept Formation Engine | ✓ | ✓ | ✓ | ✓ | ✓ | `05` §9 |
| 23 | Simulator, skills, shared skill core | ✓ | ✓ | ✓ | ✓ (S3, `08` §6) | ✓ | `05` §10–11 |
| 24 | Predictive Hierarchy, prediction records, taint, promotion | ✓ | ✓ | ✓ | ✓ | ✓ | `06` §1–2 |
| 25 | Validity envelopes | ✓ | ✓ | ✓ | ✓ | ✓ | `06` §3 |
| 26 | Support, epistemic lattice, goal decision procedure, UNKNOWN | ✓ | ✓ | ✓ | ✓ (S7) | ✓ | `06` §4–6 |
| 27 | Answer Record | ✓ | — | — | ✓ | ✓ | `06` §7 |
| 28 | Heart, Gate, Modulators, Subconscious | ✓ | ✓ | ✓ | ✓ | ✓ | `07` §1–4 |
| 29 | Mouth (planner, taint check, unparsers, renderer, articulators, templates) | ✓ | ✓ | ✓ | ✓ (S5) | ✓ | `07` §5 |
| 30 | Body (tool interface, sandbox, F0 tools, Python subset) | ✓ | ✓ | ✓ | ✓ | ✓ | `07` §6, `02` §14.2 |
| 31 | Bootstrap S0–S7, losses, acceptance, recorded artifacts | ✓ | ✓ | ✓ | ✓ | ✓ | `08` §1–3 |
| 32 | Online learning, sleep, compilation, anti-forgetting | ✓ | ✓ | ✓ | ✓ | ✓ | `08` §4–7 |
| 33 | Components, IDs, versioning, contracts | ✓ | ✓ | ✓ | ✓ | ✓ | `09` §1–2 |
| 34 | Build Configuration (fields, defaults, validation) | ✓ | ✓ | ✓ | ✓ | ✓ | `09` §3 |
| 35 | Trained Model Manifest (structure, contents, generation) | ✓ | ✓ | ✓ | ✓ | ✓ | `09` §4 |
| 36 | Package format, loading, accounting | ✓ | ✓ | ✓ | ✓ | ✓ | `09` §5–6 |
| 37 | Partial retraining, traces, gates, surgery, rollback, compatibility, third-party | ✓ | ✓ | ✓ | ✓ | ✓ | `10` |
| 38 | Integration: connection matrix, beat order, lifecycle, degradation, concurrency | ✓ | ✓ | ✓ | ✓ | ✓ | `11` |
| 39 | Evaluation, baselines, metrics, safety suite, stop criteria | ✓ | ✓ | ✓ | ✓ | ✓ | `12` |
| 40 | Roadmap, code layout, conventions, tests | ✓ | ✓ | — | ✓ | — | `14` |

## 3. Gaps found during review and resolved in the plan

| # | Gap | Resolution |
|---|---|---|
| G1 | How a goal's hypothesis set is formed, and which Sketch keys a goal implies, were not defined | `06` §6.0 added (D-020) |
| G2 | UNKNOWN-ABSENT could fire although the answer was derivable by a rule or procedure | Derivability check via REGRESS rule lookup (`06` §6.0, D-020) |
| G3 | The shared skill core had no owning component; its changes would break every skill | `core.skill_core` component, frozen after S3 (`05` §11, `08` S3, `09` §1.2, `10` §4, §10, D-019) |
| G4 | No Selector policy existed before learning (milestones M5–M6) | Heuristic VOC defined (`05` §7) |
| G5 | The F0 Python subset was not specified | Allowed/disallowed constructs listed (`02` §14.2) |
| G6 | Dataset sizes for F0 were not specified | Defaults table (`02` §14.6) |
| G7 | Region routing for consolidation was undefined | Routing rule (`04` §3) |
| G8 | Privacy scope was not enforced in consolidation | Scoped regions; no shared consolidation or training without policy (`04` §3) |
| G9 | Interface traces, probe sets, fingerprints and the coverage map had no point of creation | Recorded during S7 (`08` §2) |
| G10 | The safety suite used by retraining gates was not defined | F0 safety suite (`12` §4.7) |
| G11 | SEF gold-state names didn't match the decision states | Aligned (`02` §4.14) |

## 4. Deferred (specified at architecture level; not required for F0)

| Item | Reason for deferral | Plan reference |
|---|---|---|
| Vision and audio parsers | F0 domains are text/code/math/tables; record kinds are already modality-independent | `02` §6.8 |
| Open-domain NL parsing without a teacher | Research risk R2; F0 uses controlled language plus offline teacher extraction | `03` §8.2 |
| NVMe-direct GPU tiering, multi-GPU/distributed training | F0 fits on one GPU with host RAM + disk memory-mapping | `04` §3, `09` §3 |
| Single-file package container | Directory form suffices | `09` §5, D-006 |
| Public skill registry service | The protocol is defined; a service isn't needed for F0 | `10` §8 |
| Scale-up beyond F0 (1B+ configurations, H7) | Requires F0 results first | `12` §3 |

## 5. What remains open (experimental questions, not specification gaps)
These are **hypotheses to be measured**, not missing design:
- whether the Property Basis can be learned well enough (R1);
- whether the Composer's search stays tractable (R3);
- whether active cost is sublinear in practice, including wall-clock (H3);
- whether few-shot and compositional advantages appear (H1, H2);
- whether local learning and sleep avoid drift (R10, R11);
- all threshold values. Every threshold is configurable with a default, and S7 calibrates the epistemic thresholds; the defaults are starting points, not claims.

The plan defines how each of these is measured (`12`) and what result stops the project (`12` §5.2).

## 6. Verdict

**The architecture is sufficiently specified to begin implementation of SRM-F0.**

- Every subsystem from raw data ingestion (the lowest layer) to the Mouth's output (the highest layer) has defined data structures, algorithms or formulas, configurable parameters, training objectives or acceptance tests, and typed interfaces.
- The gaps found during review (§3) have been closed in the plan.
- Remaining uncertainties are experimental, and the plan says how to measure them.

**Preconditions for starting:**
1. The decision log (`16`) is accepted as the default design baseline (D-001 … D-020).
2. Implementation follows the milestone order in `14` §4, starting with M0 (scaffolding and Build Configuration) and aiming for the epistemic vertical slice (M0–M3 + M5 factual path) before any neural training.
3. Every deviation from the plan is recorded in `16` first.

**First implementation steps (next phase):**
1. M0: package skeleton, Build Configuration models with validation and the F0 profile, canonical JSON + hashing, CI.
2. M1: code-space algebra with property tests, registries and message types, SEF models and validation, the four synthetic generators.
3. M2–M3: memory (record store, indices, Sketch, hippocampus) and the epistemic core (Workspace, evidence algebra, taint, Support, decision procedure).
4. M5 (factual path): triage-based ingestion of FactStream, then answering with the correct epistemic states, including UNKNOWN-ABSENT and CONTESTED.

## 7. Milestone status (updated during implementation)

| Milestone | Status |
|---|---|
| M0 Scaffolding & config | not started |
| M1 Interface + SEF + generators | not started |
| M2 Memory | not started |
| M3 Epistemics core | not started |
| M4 Primitives + Composer + Body | not started |
| M5 Ingestion + intake + Mouth (templates/unparsers) | not started |
| M6 Control | not started |
| M7 Learning (CFE, online, sleep, compilation) | not started |
| M8 Component system | not started |
| M9 Learned components | not started |
| M10 Evaluation harness + baselines | not started |
| M11 SRM-F0 experiments | not started |
