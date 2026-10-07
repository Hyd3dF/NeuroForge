# Final SRM — Official Implementation Plan

**Status:** Approved architecture plan, version 1.0 (2026-10-07). Pre-implementation.
**Architecture:** Final SRM (Sparse Recruitment Machine, final revision), which incorporates the surviving parts of EGM (Epistemic Graph Machine) and SRM v1.
**First build target:** SRM-F0, the smallest prototype that can test the central hypotheses (`12_evaluation.md` §1).

This directory is the **single source of truth** for implementation. Code must follow it. Any deviation must first be recorded in `16_decision_log.md`, with the reason and the affected sections.

---

## Reading order

| # | File | Contents |
|---|---|---|
| 01 | `01_overview.md` | Lineage (EGM → SRM v1 → Final SRM), what changed, thesis, the four required problems, principles, **complete input-to-output operation**, where intelligence lives, total vs. active capacity, Diagram A |
| 02 | `02_data_ingestion_format.md` | **SRM Experience Format (SEF)**: how raw data becomes structured records; how facts, concepts, procedures, examples, skills, causal relations, contradictions and uncertainty enter the model |
| 03 | `03_interface_and_perception.md` | Frozen interface layer (code space, algebra, Property Basis, registries, message types), codec, perception and structural parsing |
| 04 | `04_memory.md` | Record store, record kinds, regions/tiers, indices, Knowledge Sketch, Hippocampal Index, Association Field, lifecycle |
| 05 | `05_intelligence_core.md` | Workspace, evidence algebra, truth maintenance, Primitive Basis, Composer, Selector, Error Monitor, Simulator, Concept Formation Engine |
| 06 | `06_prediction_and_epistemics.md` | Predictive Hierarchy, prediction records, taint, validity envelopes, Support, epistemic states, UNKNOWN, Answer Record, Diagram B |
| 07 | `07_control_expression_action.md` | Heart, Thalamic Gate, Modulators, Subconscious, Mouth, Body |
| 08 | `08_training.md` | Bootstrap stages, losses, online local learning, sleep consolidation, skill compilation, data interpretation table |
| 09 | `09_components_packaging_config.md` | Components, Component IDs, versioning, **Build Configuration**, **Trained Model Manifest**, package format, Diagram C |
| 10 | `10_partial_retraining.md` | Training closure, interface traces, validation gates, rollback, surgery, compatibility, third-party skills, Diagram D |
| 11 | `11_system_integration.md` | How all subsystems connect: connection matrix, per-beat execution order, query lifecycle, worked traces |
| 12 | `12_evaluation.md` | SRM-F0 prototype, baselines, benchmarks, metrics, success thresholds, failure criteria, risks, novelty and precedents, references |
| 13 | `13_architecture_QA.md` | Answers to the 50 architecture questions of the final research phase |
| 14 | `14_implementation_roadmap.md` | Python stack, planned code layout, milestones M0–M11, test strategy |
| 15 | `15_readiness_review.md` | Review of whether the plan is sufficient to start implementation, and the verdict |
| 16 | `16_decision_log.md` | Recorded design decisions and defaults (D-001 …) |

---

## Conventions

- **No hardcoded sizes.** Every size, capacity, threshold and budget comes from the **Build Configuration** (`09` §3). Values marked *F0 default* are defaults for the first prototype profile only.
- **"MUST" / "SHOULD" / "MAY"** are used in the RFC 2119 sense.
- **Epistemic state names** (OBSERVED, PREDICTED, UNKNOWN-ABSENT, …) are defined in `06` §5 and must be used exactly as written.
- **Cross-references** take the form `NN §x.y` (file number, section).
- Biological names (Heart, Mouth, Hippocampal Index, Thalamic Gate, …) are **labels for precisely specified mechanisms**. Each section defines the mechanism; the name implies nothing beyond that.

## Global symbols

| Symbol | Meaning | F0 default |
|---|---|---|
| `B` | Blocks per sparse code | 64 |
| `L` | Units per block (each block has exactly one active unit) | 64 |
| `r` | Blocks per index band | 3 |
| `n_b` | Number of index bands | 21 |
| `d` | Internal dense dimension | 256 |
| `d_k` | Dense key dimension for fine scoring | 128 |
| `P_0` | Initial Property Basis size | 512 |
| `N_role` | Initial role vocabulary size | 64 |
| `K` | Workspace claim slots | 64 |
| `R` | Workspace referent slots | 32 |
| `H` | Hypothesis channels | 8 |
| `A` | Maximum premises per derived claim | 3 |
| `N_lib` | Cortical Library capacity (records) | swept 1e4 → 1e7 |
| `N_hip` | Hippocampal Index capacity | 1e5 |
| `T_max` | Maximum beats per query | 64 |
| `k_ret` | Stage-1 retrieval candidates | 64 |
| `k_align` | Stage-2 alignment survivors | 8 |
| `W` | Evidence prior weight (D-021) | 0.15 |
| `θ_commit` | Commit belief threshold | 0.80 |
| `θ_retract` | Retraction disbelief threshold | 0.60 |
| `θ_abstain` | Minimum belief to answer (scaled by stakes) | 0.70 |
| `k_env` | Validity-envelope centroids per generator | 16 |
| `r_skill` | Rank of a compiled-skill modulation | 8 |
| `V_out` | Mouth output vocabulary | 8192 |

## Scope of this version

In scope for SRM-F0: text in a controlled language, synthetic natural-language renderings, a list/integer DSL, a Python subset, arithmetic and algebra, structured tables, synthetic object worlds and fact streams.

Deferred (specified at architecture level, not implementation-ready): vision and audio parsers, open-domain natural-language parsing without a teacher, NVMe-direct GPU tiering, distributed training, single-file package containers, and a public skill registry service. See `15_readiness_review.md` §4.
