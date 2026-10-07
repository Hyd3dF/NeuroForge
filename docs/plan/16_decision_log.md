# 16 — Decision Log

Every design decision that fixes an open choice, and every later deviation from the plan, is recorded here **before** code depends on it.

Format: `D-NNN — title`. Each entry gives the status, the decision, the reason, the alternatives considered, and the affected sections.

---

### D-001 — Implementation language and framework
- **Status:** accepted (2026-10-07)
- **Decision:** Python ≥ 3.11 with PyTorch 2.x.
- **Reason:** user requirement (Python); ecosystem; safetensors and memory-mapping support.
- **Alternatives:** JAX (strong for batching, weaker for irregular memory structures).
- **Affects:** `14` §1.

### D-002 — Sparse block codes as the code space
- **Status:** accepted
- **Decision:** `B` blocks × `L` units, one active unit per block; bind = blockwise addition mod `L`.
- **Reason:** exact and invertible binding that preserves sparsity; one byte per block; compatible with banded hashing.
- **Alternatives:** dense HRR (not sparse, not indexable); binary spatter codes (dense).
- **Affects:** `03` §2.

### D-003 — Banded LSH as the default index
- **Status:** accepted
- **Decision:** `n_b` bands × `r` blocks; candidates capped; exact rescoring. Dense ANN backend is optional.
- **Reason:** analyzable recall (`1−(1−p^r)^{n_b}`), and candidate counts roughly independent of `N` until `N ≈ L^r`.
- **Affects:** `04` §4.

### D-004 — Retrieval uses projected content codes, not sparsified bundles
- **Status:** accepted
- **Decision:** structure bundles are kept as count matrices for exact relational queries; retrieval uses content codes of learned dense embeddings and signature codes.
- **Reason:** sparsified bundles lose per-block agreement, so banded recall collapses.
- **Affects:** `03` §2.6–2.7.

### D-005 — Storage formats
- **Status:** accepted
- **Decision:** safetensors (tensors), Parquet (tables), JSON with RFC 8785 canonicalization (metadata), SHA-256 hashing, base32 logical-ID encoding.
- **Affects:** `09` §2, §5.

### D-006 — Package format `srmpkg-1` is a directory
- **Status:** accepted
- **Decision:** a directory with header, Manifest, sharded component and region directories. A single-file container is deferred.
- **Reason:** simplest memory-mapping and partial loading; easy inspection during development.
- **Affects:** `09` §5.

### D-007 — Formal outputs rendered by deterministic unparsers
- **Status:** accepted
- **Decision:** code, JSON and math output from internal ASTs/structures via unparsers and serializers. The neural renderer is used only for natural language.
- **Reason:** output exactly equals the internal solution; no expression-level hallucination in formal targets.
- **Affects:** `07` §5.2.

### D-008 — Many primitives are algorithmic (A-class) in F0
- **Status:** accepted
- **Decision:** algebra, retrieval, unification, abstraction, ordering, search and control are exact algorithms. ALIGN, COMPARE, EXPLAIN, PREDICT_STEP and REGRESS are hybrid with learned scoring; the verifier, Composer policy, Selector, parsers and Mouth are learned.
- **Reason:** exactness where it is possible; learning where judgment is needed. The architecture is honestly neuro-symbolic at the operation level.
- **Affects:** `05` §4.

### D-009 — Deterministic parsers for code, math and tables; learned parser for controlled-language text; teacher extraction offline for open text
- **Status:** accepted
- **Reason:** formal domains have exact grammars, so F0 can test the architecture without the open NL-parsing risk (R2).
- **Affects:** `02` §6, `03` §8.

### D-010 — Default t-norm `min` for derived belief
- **Status:** accepted
- **Reason:** conservative (never more confident than the weakest premise); `product` is configurable.
- **Affects:** `05` §2.3.

### D-011 — Verifier entailment does not remove taint by default
- **Status:** accepted
- **Reason:** the verifier is learned. Promotion requires observation, test, untainted exact deduction or independent corroboration. Configurable for low stakes only.
- **Affects:** `06` §2.3.

### D-012 — Fact-placement rule
- **Status:** accepted
- **Decision:** world facts live only in ENGRAM records; skills and Core are trained with fact-dropout; fact-like skill outputs are SUGGESTED.
- **Reason:** keeps UNKNOWN-ABSENT exact and knowledge editable.
- **Affects:** `06` §8.

### D-013 — Reported claims are stored as attributed claims
- **Status:** accepted
- **Decision:** "X says P" stores "X asserts P"; P gains no evidence until independently supported.
- **Affects:** `02` §10.2.

### D-014 — F0 domains
- **Status:** accepted
- **Decision:** ObjectWorld, DSLWorld (+ Python subset), FactStream, MathWorld-lite. Vision and audio are deferred.
- **Reason:** full control over ground truth, held-out compositions and unknowns; contamination-free evaluation.
- **Affects:** `02` §14, `12` §1.

### D-015 — The Heart allocation algorithm is deterministic; only value estimators are learned
- **Status:** accepted
- **Reason:** auditability of compute allocation; stable training.
- **Affects:** `07` §1.8.

### D-016 — Template renderer before neural renderer
- **Status:** accepted
- **Decision:** the Mouth ships with deterministic templates per (role, epistemic tag). The neural renderer is added in M9 and keeps the templates as fallback.
- **Affects:** `07` §5.5, `14` M5/M9.

### D-017 — Component retrain class assignments
- **Status:** accepted
- **Decision:** the classes in `10` §10 (independent / with_closure / frozen / derived).
- **Affects:** `10` §10, `09` §1.3.

### D-018 — Records are not components
- **Status:** accepted
- **Decision:** facts and concepts are records inside region components, edited by record operations; regions are versioned components.
- **Reason:** keeps the Manifest tractable while every fact stays addressable.
- **Affects:** `09` §1.1.

### D-019 — Shared skill core is frozen after S3
- **Status:** accepted (added during readiness review)
- **Decision:** `core.skill_core` is pretrained in S3, then frozen and versioned with the interface MAJOR version. Skills are low-rank modulations of it.
- **Reason:** if the core changed, every compiled skill would silently change behavior. Freezing makes skills independently retrainable components.
- **Affects:** `05` §11, `08` S3, `09` §1.2, `10` §4, §10.

### D-020 — Goal hypothesis sets and derivability check before UNKNOWN-ABSENT
- **Status:** accepted (added during readiness review)
- **Decision:** every goal owns a hypothesis set over candidate answer bindings plus RESIDUAL. UNKNOWN-ABSENT requires a missing Sketch key that is also not present in the input and not derivable by any rule or procedure producing its relation.
- **Reason:** prevents premature "I don't know" when the answer can be derived rather than recalled.
- **Affects:** `06` §6.0.

### D-021 — Evidence calibration: W = 0.15, θ_conflict_mass = 0.5
- **Status:** accepted (Batch 1 implementation)
- **Problem found:** with the original defaults `W = 2.0` and `κ = 1.0`, one fully trusted source gives `b = 0.9/(0.9+2) = 0.31`. `θ_commit = 0.80` was then unreachable without about 8 independent trusted sources, so nothing could ever be KNOWN. This contradicted `02` §10 (one curated source should be strong evidence).
- **Correction:** `W = 0.15`. One curated source (t = 0.9) gives `b = 0.857 ≥ θ_answer(0.5) = 0.825`; one web source (0.5) gives `b = 0.77` (INSUFFICIENT alone); two independent web sources give `b = 0.87`. `θ_conflict_mass` is scaled to 0.5 to match. Added `remembered_single_source_min_trust = 0.85` to make the "high-trust single source" clause of `06` §5 configurable.
- **Affects:** `09` §3.1, `04` §9, README symbols. Implemented in `srm/config/build_config.py`; tested in `tests/test_epistemics.py`.

### D-022 — Decision procedure: CONTESTED step, INSUFFICIENT catch-all, OPEN_QUESTION sketch keys
- **Status:** accepted (Batch 1 implementation)
- **Problems found:** (1) CONTESTED was listed as a state but no step produced it. (2) An untainted hypothesis with `θ_predict ≤ b < θ_answer` matched no step. (3) A declared unknown with no facts has no Sketch key, so step 1 (UNKNOWN-ABSENT) fired before step 2 (UNKNOWN-DECLARED) could.
- **Correction:** explicit CONTESTED step; INSUFFICIENT as the final catch-all; OPEN_QUESTION records register their keys in the Sketch.
- **Affects:** `06` §6, `04` §7. Implemented in `srm/prediction/decision.py`, `srm/memory/system.py`.

### D-023 — Environment: Python 3.13, PyTorch from PyPI
- **Status:** accepted (Batch 1 implementation)
- **Decision:** the reference environment is Python 3.13 with PyTorch 2.x from PyPI (the PyTorch CPU wheel index is not reachable from the build environment). Batch 1 uses no torch; learned components (M9) use torch on CPU when no GPU is present.
- **Affects:** `14` §1.

### D-024 — Model-side ingestion lives in `srm/ingest/`
- **Status:** accepted (Batch 1 implementation)
- **Decision:** triage and SEF→memory operations (`02` §9) are a separate package `srm/ingest/` rather than part of `srm/data/` (offline conversion) or `srm/perception/` (raw input). This keeps offline data tooling separate from model components.
- **Affects:** `14` §2.
