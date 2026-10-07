# 08 — Training and Learning

Final SRM learns in four regimes:

| Regime | When | What changes | Mechanism |
|---|---|---|---|
| **Bootstrap** | Once per model generation (and per interface MAJOR version) | Interface layer, learned primitives, Composer, Selector, verifier, parsers, Mouth | Backpropagation on targeted objectives (§2) |
| **Online (local)** | During every query/session | Records, links, evidence, envelopes, calibration, small updates to recruited components | Writes, Hebbian, verification-as-teacher, three-factor gradients (§4) |
| **Sleep** | Idle (micro) or scheduled (deep) | Library organization, schemas, skills, region versions, Manifest | Replay, integration, abstraction, compilation, repair, re-layout (§5–6) |
| **Component retraining** | On demand, after packaging | Selected components only | Training closure on interface traces (`10`) |

**Never used:** a next-token loss over a whole corpus to put knowledge into weights. Knowledge enters through SEF ingestion (`02` §9). Token prediction is trained only inside the Mouth (stage S5).

## 1. Learned components and their objectives

| Component | Objective (loss) |
|---|---|
| Content-code projection + code embedding (`interface.codespace`) | `L_sim` (rank correlation between `ov/B` and cosine) + `L_commit` (quantization commitment) + `L_entropy` (balanced block usage) |
| Property Basis dense vectors (`interface.property_basis`) | `L_prop` (multi-label property prediction from instance embeddings, against generator ground truth) + `L_contrast` (InfoNCE: instances of the same concept vs. others) + `L_disent` (decorrelation penalty across property directions) |
| SetEncoder for signatures | `L_sig` (same relational pattern → close; different → far; synthetic graph pairs) |
| Perception encoder + L0 predictor | `L_pred0` (cosine to the next chunk's encoding) + auxiliary byte reconstruction (small weight) |
| Text parser (`perception.parser.text`) | Cross-entropy on spans, labels, modality/polarity/hedge heads vs. gold SEF |
| ALIGN affinity `f_aff` | Binary cross-entropy on node correspondences (synthetic pairs) + listwise loss on alignment scores |
| COMPARE calibration `φ` | Logistic regression on recognition outcomes |
| EXPLAIN llr heads | Logistic loss: does the evidence item support the true hypothesis? Calibrated (temperature) |
| PREDICT_STEP dynamics skills | MSE / CE on next-state properties (synthetic transitions); envelope initialized from training inputs |
| Verifier | 3-way CE (entail / contradict / neutral) + on-policy negatives + calibration |
| Composer policy `π_θ` | Imitation (CE on reference expansions) → REINFORCE/PPO with reward = verified success − λ·cost (+ ΔU shaping) |
| Selector actor-critic | TD(λ) critic; policy gradient actor; VOC head regression on realized ΔU |
| Gate salience weights | Logistic: admitted item ends up on the goal-support path |
| Mouth renderer | Token CE on `expression_pair` + attribution pointer CE + copy loss |
| Articulators | Same as the Mouth, restricted to target-format pairs |
| Compiled skills | Distillation from circuit traces (§6) |
| Stakes classifier | CE on labeled request stakes (optional) |

## 2. Bootstrap stages (F0)

Each stage has inputs (SEF kinds), outputs (frozen or trained components), and acceptance criteria. All hyperparameters come from the Build Configuration (`learning.bootstrap.*`).

### S0 — Data generation
- Run the ObjectWorld, DSLWorld, FactStream and MathWorld-lite generators (`02` §14) and produce SEF datasets with splits.
- **Accept:** the dataset manifests validate; split leakage checks pass (no held-out composition appears in train).

### S1 — Interface layer
- **Train:** projection/embedding tables, Property Basis vectors, SetEncoder, value encoders.
- **Data:** `concept_definition`, `concept_example`, `fact`, `property_def`, `relation_def`.
- **Accept:**
  - Spearman correlation(`ov/B`, cos) ≥ 0.8 on held-out pairs;
  - block-usage entropy ≥ 0.9·log L;
  - property prediction macro-F1 ≥ target (config);
  - banded-index recall@64 ≥ 0.95 for neighbours with cos ≥ 0.8.
- **Freeze** → interface v1.0. Compute the ABI hash.

### S2 — Perception
- **Train:** perception encoder + L0 predictor; text parser on (rendered text, gold SEF); deterministic parsers are wrapped and validated.
- **Accept:** parser fact-level F1 ≥ target; modality accuracy ≥ target; L0 error decreases on in-distribution text.

### S3 — Learned primitives, shared skill core and verifier
- **Train:** ALIGN affinity, COMPARE `φ`, EXPLAIN heads, the **shared skill core** (`05` §11, multi-task on DSL operations, property mappings and dynamics; frozen at the end of S3), PREDICT_STEP dynamics skills (as modulations of the skill core), verifier.
- **Accept:** alignment F1 on held-out graph pairs; verifier AUROC and ECE targets; dynamics accuracy within envelope.

### S4 — Composer and Selector (meta-learning for composition)
- **Data:** episodes drawn from **task families**, each episode a stream of related tasks (meta-learning for compositionality), using `task` records with verifiers and reference solutions.
- **Phase A:** imitation on reference circuit graphs.
- **Phase B:** RL with verified rewards, cost penalty and nogood usage, with a curriculum on depth and composition novelty.
- **Accept:** success on held-out *compositions* ≥ target; average expansions per solved task ≤ target.

### S5 — Mouth
- **Train:** renderer + articulators on `expression_pair` (template-generated plans with paraphrased targets). Deterministic unparsers are validated by round-trip (parse(unparse(AST)) = AST).
- **Accept:** attribution precision ≥ target; zero literal-copy errors on the test set; fluency judged by reference metrics.

### S6 — Library population (reading)
- Ingest SEF knowledge datasets through triage (`02` §9): writes, evidence, contradictions, sketch, indices. Then run deep sleep cycles (§5).
- **Accept:** index/sketch invariants hold; retrieval recall on held-out queries ≥ target.

### S7 — End-to-end calibration
- Run full queries on `question` and `task` dev sets. Fit thresholds: `θ_answer`, `θ_predict`, `δ_margin`, envelope `θ_env`, Support weights `λ_*`, by maximizing the selective-accuracy / coverage objective at the configured risk target.
- **Accept:** calibration ECE ≤ target; UNKNOWN-detection AUROC ≥ target.
- **Artifacts recorded during S7** (needed for the component system):
  1. **Interface traces** for every retrainable component (reservoir-sampled per task family, `10` §3).
  2. **Probe sets and behavior fingerprints** per component (fixed samples of its input traces; `10` §5.1).
  3. **Coverage map:** for every validation and regression test, the set of components it recruited (`09` §4.2).
  4. **Validity envelopes** initialized for every learned generator from the dev-set verification outcomes.

The bootstrap writes the first **Trained Model Manifest** (`09` §4), including each component's training record and data-category proportions.

## 3. Training infrastructure requirements
- Every stage runs **per component** with explicit input/output messages. Stages S3–S5 can therefore train on recorded interface traces (`10` §3), which is the same mechanism later used for retraining.
- Determinism: seeds for generators, initialization and data order are recorded in the training record.
- Logging: losses, validation metrics, compute used (GPU-hours, peak memory) and data categories per run.

## 4. Online local learning

### 4.1 Writes
Every NOVEL unit is written (`02` §9.1). This is one-shot learning with no gradient.

### 4.2 Hebbian links
After each beat: `04` §6.2.

### 4.3 Verification-as-teacher
- When a claim produced by learned producer `n` (skill, ALIGN, PREDICT_STEP, parser) is later promoted (positive) or retracted / refuted (negative), store `(input message, output, label)` in `n`'s **local replay buffer**.
- At micro-sleep or when the buffer is full, take `n_local_steps` gradient steps on `n` **only**:
  - learning rate `η_local × plasticity(n)`;
  - anchor regularization `λ_anchor · KL/MSE(n_new, n_old)` on a reservoir of earlier inputs.
- Only the recruited component's parameters change.

### 4.4 Envelope and calibration updates
Each verification outcome updates the producer's envelope centroid statistics (`06` §3) and its calibration (temperature, online).

### 4.5 Habit/deliberation conflict
When a skill disagrees with deliberate derivation, the skill gets a negative label for that input, and its envelope region's success count is updated. If accuracy drops below `θ_recompile`, the skill is queued for recompilation in sleep.

### 4.6 Three-factor updates (Selector, Composer)
Eligibility traces over the actions in an episode, multiplied by the DA signal (TD error) at the outcome. Updates are applied in small steps online or batched at micro-sleep, configurable.

### 4.7 Concept Formation
Online: new schemas are written (`05` §9). Variability profiles update by Welford from each new instance.

## 5. Sleep (consolidation)

### 5.1 Triggers
- **Micro-sleep:** idle time > `t_idle` → time-sliced tasks (bounded by `micro_sleep_budget`).
- **Deep sleep:** hippocampal fill ≥ `θ_hip_fill`, or schedule `sleep_every`, or an explicit call.

### 5.2 Deep sleep phases (in order)
1. **Replay sampling:** draw hippocampal records and episodes by priority `surprise × utility × uncertainty`. Interleave **existing-knowledge replay** (stable records and Mouth-free re-derivations of stored facts) at ratio `replay_mix` (F0 1:1).
2. **Integration:** for each record:
   - a fact with lifecycle ≥ CORROBORATED (or high-trust single source, per configuration) → write into the target region (chosen by content-code region routing);
   - a concept → merge into the existing schema (Welford update of profiles; contrast update) or insert as a new schema;
   - a procedure → insert or merge;
   - an episode → keep while it has replay value, otherwise compress (keep the outcome summary, drop the residue).
3. **Abstraction mining:** frequent co-activation sets and recurring aligned subgraphs among recent episodes and instances (ALIGN + ABSTRACT over clusters). Accept a new schema if the **MDL gain** `G = Σ_i DL(x_i) − [DL(S) + Σ_i DL(x_i | S)] > θ_mdl`, where `DL(x) = Σ_properties −log2 P(value | default)` + structure bits.
4. **Skill compilation:** §6.
5. **Truth-maintenance repair:**
   - re-evaluate CONTESTED records with all evidence;
   - propagate DEPRECATED records along `derived_from` and `trained_from` links: flag dependents for re-verification, and queue skills trained on deprecated data for recompilation;
   - merge provisional entities, relations and properties when they are consistent duplicates (`02` §7).
6. **Causal promotion:** an `associated_with` link that holds with consistent sign and strength across ≥ `n_ctx_causal` distinct contexts (an invariance test in the spirit of invariant causal prediction) becomes a **candidate** `causes` link with state PREDICTED, and needs a test or intervention to promote.
7. **Downscaling, pruning, merging:** multiply Hebbian weights by `downscale` (F0 0.95); prune links below `w_min`; merge near-duplicate records (ov ≥ `θ_merge` and compatible payload); decay low-priority hippocampal records.
8. **Re-layout:** re-cluster records into regions and pages by co-activation (graph partitioning over `co_activated` + `used_with` edges, balanced by size) to minimize expected pages per query. A changed region gets a **new region version** (component version), and indices are rebuilt for that region.
9. **Dreaming:** generate synthetic problems from new schemas and procedures (instantiate with sampled bindings), solve them in the Workspace, and use the results as verification evidence for those schemas.
10. **Manifest update:** emergence events (new skills, schemas, regions, splits, merges), region versions and training records are written to a new Manifest version (`09` §4).

## 6. Skill compilation (temporary circuit → persistent skill)

### 6.1 Candidate selection
Recurring circuit subgraphs (by graph hashing up to role renaming) among successful episodes, with frequency ≥ `n_compile` (F0 5) and mean cost ≥ `c_compile`.

### 6.2 Two outputs, in order
1. **Procedure schema:** ABSTRACT across the trace instances → PROCEDURE record (explicit, cheap). Always done.
2. **Compiled skill** (only if frequent and costly):
   - collect `(input messages, output)` pairs from the traces plus fresh Simulator-generated inputs, labeled by running the *slow path* (the circuit);
   - train a new low-rank modulation over the shared skill core (`05` §11) by distillation.

### 6.3 Acceptance
- Held-out agreement with the slow path ≥ `θ_accept` (F0 0.98).
- Cost reduction ≥ `speedup_min` (F0 2×).
- No contract violation.
- The envelope is initialized from the training inputs; its success statistics come from the held-out checks.

### 6.4 Registration
A new **skill component** with a logical ID, lineage (`derived_from`: procedure id, episodes, sleep cycle id), validation suite (held-out pairs), and behavior fingerprint, all recorded in the Manifest (`09`). **Promotion to derived primitive:** reused across ≥ `n_domains_promote` task families with stable accuracy.

### 6.5 Splitting and merging
- **Split:** a skill whose envelope shows bimodal accuracy across clusters is split into two skills with context-restricted envelopes (new IDs, `split_from` lineage).
- **Merge:** two skills that agree on ≥ `θ_accept` of the union of their envelopes are merged (new ID, `merged_from`).

## 7. Anti-forgetting measures (summary)
- New knowledge goes into new records or new components (parameter isolation).
- Sparse, nearly orthogonal codes.
- Context segments isolate versions and domains.
- Lifecycle-dependent plasticity.
- Interleaved replay during sleep.
- Anchor regularization for every local gradient update.
- Record and component versioning instead of overwriting.
- Region-level regression checks after re-layout.

## 8. Data interpretation table (how each information type is learned)

| Type | First destination | Representation | Learning | Validation | Consolidation | Retrieval |
|---|---|---|---|---|---|---|
| Fact | Hippocampus (NEW) | ENGRAM + entity links | Write | Corroboration, consistency, Body check | Library ENGRAM once CORROBORATED and USED | Sketch check → triple index / content index |
| Example | Concept Formation Engine | SceneGraph instance | Align / abstract / inherit | Fits or contrasts with siblings | Updates schema and variability profile | Two-stage matching |
| Concept definition | Library CONCEPT (NEW) | Factored schema | Write + profile init | Consistency with labeled examples | Merged with instance-derived profiles | Content index, is-a traversal |
| Procedure | Hippocampus → PROCEDURE | Circuit template | Write; refined by execution | Execution success | Compiled into a skill if frequent and costly | Goal/type match |
| Code pattern | Hippocampus → CONCEPT (code pattern) | Typed AST template with holes | Mining + anti-unification | Parses, type-checks, passes tests | Generalized schema if MDL gain | Signature/goal match, version-gated |
| Skill | Created in sleep from verified traces | Skill component (low-rank modulation) | Distillation | Agreement with slow path, envelope | Promoted / split / merged | Procedural recruitment; Subconscious habits |
| Causal relation | Link + causal schema (mechanism/intervention) or `associated_with` (observational) | Typed link with conditions | Write; invariance test | Simulation, counterexamples, Body tests | Promoted when invariant and tested | Causal traversal (PREDICT_STEP / REGRESS) |
| Contradiction | Truth-maintenance store | Contradiction record + links | Evidence on both sides | Trust gap, tests | Sleep repair | Surfaces when either side is recalled |
| Hypothesis | Workspace channel | CONJECTURED claim | None until tested | Counterexample search | Stored as DERIVED + TESTED if it survives; else nogood | Hypothesis mode only |
| Known unknown | OPEN_QUESTION record | Pattern + scope | Write | Superseded by new untainted evidence | Kept until resolved | Pattern match → UNKNOWN-DECLARED |
| Repeated pattern | Recognition-by-recall | Statistics bump | None | — | Feeds abstraction mining | Higher prior |
| Noise | Triage | Not stored | — | Low confidence / unstructured | — | — |
