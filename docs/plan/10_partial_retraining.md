# 10 — Partial Retraining, Surgery, Rollback, Compatibility

## 1. Overview of the retraining pipeline

```
SELECT → CLOSURE → STAGE → TRAIN → LOCAL TESTS → INTEGRATION → GATE → VERSION → COMMIT → MONITOR (→ ROLLBACK)
```

Every step operates on the **Manifest** (`09` §4). Without a valid Manifest, retraining is refused (§11).

## 2. Selection and the training closure

### 2.1 Selection
The developer selects one or more logical IDs. Selection is by ID, by type pattern (e.g. `mouth.articulator.*`), or by query over Manifest fields (functional label, usage profile).

### 2.2 Closure algorithm
Inputs: the selected set `S` and the dependency graph `G`.
1. `T ← S` (components to train).
2. **Upstream producers.** For each `c ∈ T`, for each edge `u → c` of type `calls`/data input: if an **interface trace** for `c`'s input ports exists and covers the target data (§3.3), mark `u` as **TRACE-REPLACED** (not loaded). Otherwise mark `u` **LOAD-FROZEN** (inference mode).
3. **Objective dependencies.** For each `c ∈ T`, every dependency named in `c.contract.local_objective.required_frozen` (e.g. the verifier as teacher, the Mouth renderer base for an articulator, the codec) is marked **LOAD-FROZEN**.
4. **Interface layer.** Always **LOAD-FROZEN** (needed to encode targets), memory-mapped read-only.
5. **Consumers.** Every component `v` with an edge `c → v` (`calls`, `renders_with`, `verified_by` reversed) is marked **TEST-ONLY**: loaded only during integration tests (§5), never trained.
6. **Coupled training** (explicit opt-in only). If the developer selects `widen_closure`, or the gate detects that the target metric is unreachable on traces alone (§5.3), add the upstream components to `T`.
7. **Output:** the closure report: `T` (trained), LOAD-FROZEN, TRACE-REPLACED, TEST-ONLY, with estimated GPU memory and the data and traces required.

### 2.3 Guarantee
Gradients exist only for parameters of components in `T`. LOAD-FROZEN components run under `no_grad`. Every other component is not loaded at all.

## 3. Interface traces

### 3.1 What a trace is
The recorded **input messages** at a component's input ports (and optionally its outputs and downstream verdicts), captured while the full system runs on a curated dataset.

### 3.2 Recording
- Enabled per component by runtime policy (`runtime.trace_recording`).
- Per record: query id, task family, data category, timestamp, input messages (serialized: codes as uint8, dense as fp16, typed fields), output messages, downstream verdict (promoted / retracted / test pass/fail), upstream version IDs, interface version.
- **Sampling:** reservoir sampling per task family; a size cap per component.
- **Storage:** `traces/<logical_id>/<interface_version>/shard-*.safetensors` + `index.parquet`.

### 3.3 Validity
- Traces are pinned to the interface MAJOR version and to the upstream version IDs.
- If an upstream component has a newer *minor* version, the traces remain valid but are flagged.
- If an upstream component has a newer *major* version or a new logical ID, the traces must be re-recorded.

### 3.4 Coverage
For each trace set, the Manifest/trace index records task-family and data-category coverage and the envelope clusters represented. The closure algorithm uses this to decide whether traces suffice for the target data.

## 4. Local objectives and anchoring

| Component type | Local objective | Required frozen dependencies |
|---|---|---|
| `mouth.articulator.*` | Token CE + attribution + copy losses on (plan → target) pairs | Renderer base, codec |
| `mouth.renderer_base` | Same, all articulators off | Codec |
| `skill` | Distillation / supervised on (input → verified output) pairs; verification-as-teacher labels | `core.skill_core` (frozen), interface layer |
| `core.primitive.align` | Correspondence CE | Interface layer |
| `core.error_monitor.verifier` | 3-way CE + calibration | Interface layer |
| `core.composer` | Imitation + RL on tasks | Full Core in inference mode (closure is large; classified `with_closure`) |
| `core.selector` | Actor-critic on episodes | Full Core in inference mode (`with_closure`) |
| `perception.parser.text` | Structured extraction CE | Perception encoder (frozen) |
| `memory.region` | Record operations + local payload updates (schemas' numeric profiles) | Interface layer |

**Anchor regularization (mandatory, except for new components):**
`L = L_target + λ_anchor · D(f_new(x), f_old(x))` for `x` sampled from traces **outside** the target data distribution (by task-family/category tags). `D` = KL for distributions, MSE for dense outputs, Hamming-surrogate CE for codes.

## 5. Validation and gates

### 5.1 Local tests
1. **Contract tests:** port types, value ranges, invariants (including epistemic invariants: no untainted claims without provenance; taint flags preserved).
2. **Local validation suite** from the contract.
3. **Behavior fingerprint:** run the component on its fixed probe set; fingerprint = output statistics (per-block code histograms, dense mean/covariance, class distributions). Distance `d_fp` = Jensen–Shannon (discrete) or MMD (dense). Classification: `d_fp < d_patch` → patch; `< d_minor` and contract passes → minor; else **major** (requires a new logical ID or rejection).

### 5.2 Integration tests
- Select from the **coverage map** every system test that recruited the component in recorded validation runs.
- Run the full system on those tests with the candidate version. The runtime pages in only what those tests recruit.
- Then run the **regression subset** (stratified across task families) and the **safety subset**.

### 5.3 Gates (thresholds in the Build Configuration `components` section)

| Gate | Pass condition |
|---|---|
| Target improvement | Target metric gain ≥ `δ_min` |
| Regression | Every coverage-selected and regression suite: drop ≤ `ε_reg` (F0 0.5 percentage points) |
| Safety | No regression on the safety subset; epistemic invariants hold (no increase in taint leaks, abstention-calibration ECE within tolerance) |
| Fingerprint class | Patch or minor (major → reject or re-identify) |
| Shadow | During a shadow run (old and new side by side on live or replayed traffic), disagreement rate ≤ `ε_shadow`, and disagreements are not systematically worse under verification |
| Reachability | If the target metric plateaus below `δ_min` with the trace-only closure, report "closure insufficient" and recommend `widen_closure` |

## 6. Transactional surgery

### 6.1 Operations
Retrain component; replace component (same contract); add component (new skill/articulator/region); remove component (if no non-optional dependents); split/merge components; record operations at scale (bulk edits, tombstones); re-index; re-layout; interface migration (§9).

### 6.2 Procedure
1. Open a **staging manifest** (copy-on-write over the current root).
2. Apply the operations to staged copies only. Production shards are immutable.
3. **Structural invariant checks:**
   - all dependency edges resolve within their version ranges;
   - no cycles among `trained_with` edges;
   - the interface version is consistent for all components;
   - indices match their regions' versions;
   - sketch counts match engram/concept/entity keys;
   - the machinery floor is present;
   - contracts validate;
   - no component outside the staging set was modified (hash comparison).
4. Run the validation pipeline (§5).
5. **Commit:** write the new Manifest root (`parent_root_hash` = previous) and atomically switch the active root pointer (`header.json` update via write-then-rename).
6. Keep the previous root for rollback.

## 7. Versioning and rollback
- Component and region versions are **immutable**. The Manifest is a set of pointers to versions.
- **Manual rollback:** set the component's pointer to an earlier version ID. This creates a new Manifest root (history is never rewritten); it is atomic and needs no consumer change because the contract is unchanged.
- **Automatic rollback triggers** (runtime monitors, configurable windows): rising verifier rejection rate, more habit/deliberation conflicts, higher abstention-probe failure or taint-leak rate, or degradation of a user-defined metric beyond threshold → roll back to the previous version and file an incident record in the Manifest history.

## 8. Third-party components (e.g. skills)
1. The third party obtains the published Manifest and interface specification (the interface layer, read-only).
2. They develop against the contract, using published traces or their own recorded traces.
3. They package: a component shard + contract + validation suite + fingerprint + training record (data categories) + signature.
4. The host runs compatibility checks (§9) and the surgery procedure (§6) in staging.
5. **Trust:** the component enters with `trust_level = third_party`. Its outputs are **SUGGESTED** (tainted; source trust 0.30) until its envelope accrues verified successes (`06` §3). Then it may be raised to `verified` by policy.

## 9. Compatibility checking and interface migration

### 9.1 Compatibility check (old Manifest vs. new model or component)
Compare:
- manifest schema version;
- package format;
- interface MAJOR/MINOR and ABI hash;
- primitive set version;
- each component's interface signature hash;
- dependency version ranges.

The report classifies each component as **compatible**, **migratable** (an adapter exists), or **incompatible** (needs retraining or re-indexing).

### 9.2 Interface migration (MAJOR change)
1. Train a **migration adapter** (old code/dense → new code/dense) on paired encodings of the same SEF records.
2. Re-encode the regions (records re-projected; indices rebuilt).
3. Revalidate the components. Components with private dense ports retrain on re-recorded traces.

This is expensive by design. Interface MAJOR changes are planned events.

## 10. Separability classes

| Class | Components | Meaning |
|---|---|---|
| `independent` | `skill`, `mouth.articulator.*`, `memory.region` (record ops + local payload), `perception.parser.<modality>` (above the frozen encoder), `body.tool_adapter.*`, `core.primitive.*` with local objectives (ALIGN, EXPLAIN heads, PREDICT_STEP skills) | Trainable alone on traces with frozen objective dependencies |
| `with_closure` | `core.composer`, `core.selector`, `control.gate`, `control.heart_policy` estimators, `core.error_monitor.verifier`, `mouth.renderer_base` | Trainable, but system-wide effects require the full integration suite and extended shadow runs |
| `frozen` | `interface.*` (codec, codespace, property basis existing entries, registries, signature encoder), `core.skill_core` | Changed only by MAJOR migration (§9.2); a skill-core change also requires re-distilling all skills |
| `derived` | `memory.index`, `memory.sketch` | Rebuilt, never trained |

## 11. Operations refused without a valid Manifest
Selective retraining; replacement; adding or removing components; splitting or merging; re-indexing; re-layout; rollback; compatibility checks; partial loading for training; per-component quantization; provenance and safety audits; trust attribution. Plain inference MAY run from a package whose Manifest root hash is valid; a package with an invalid or missing Manifest is refused.

## 12. Worked example: improving formatting
- **Select** `mouth.articulator.markdown` and `mouth.articulator.json`.
- **Closure:**
  - trained = {the two articulators};
  - LOAD-FROZEN = {renderer base, codec, interface layer};
  - TRACE-REPLACED = {planner, Core, Library} (recorded UtterancePlans);
  - TEST-ONLY = {nothing else; integration tests run the full system on formatting suites}.
- **GPU-resident:** the renderer base + 2 articulators. The rest of the model is not loaded.
- **Train** on (recorded plan → improved target) pairs, with anchoring on non-formatting traces.
- **Gates** as in §5. Then **commit** a new Manifest root; old versions are kept for rollback.

## 13. Diagram D — Partial retraining

```
 1. SELECT      developer picks LogicalIDs from MANIFEST  (e.g., art:markdown, art:json)
        │
 2. CLOSURE     walk dependency graph
        │         upstream producers  → replaced by RECORDED INTERFACE TRACES (not loaded)
        │         objective deps      → load FROZEN (renderer-base, codec, interface)
        │         consumers           → reserved for integration tests only (via coverage map)
        ▼
 3. STAGE       open copy-on-write STAGING MANIFEST; mmap only selected shards to GPU
        │
 4. TRAIN       local objective + anchor-to-previous-version on out-of-target traces
        │         (no gradients anywhere else; rest of model not resident)
        ▼
 5. LOCAL TESTS contract types/invariants · epistemic invariants · local suite ·
        │         behavior fingerprint diff → classify patch / minor / major(=new LogicalID)
        ▼
 6. INTEGRATION coverage-map-selected system tests (runtime pages in only what they recruit)
        │         A/B vs previous version · safety subset · shadow run
        ▼
 7. GATE ── fail ──► discard staging (old version untouched)
        │
       pass
        ▼
 8. VERSION     new VersionID = H(LogicalID, weights digest, dep versions, training record)
        │         training history + data categories + results appended
        ▼
 9. COMMIT      atomic swap of manifest root (old root retained)
        │
 10. MONITOR    runtime monitors (verifier rejects, conflicts, abstention probes, user metrics)
                  └─ regression ──► ROLLBACK: pointer → previous VersionID (atomic)
```
