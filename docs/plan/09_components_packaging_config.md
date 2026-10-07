# 09 — Components, Component IDs, Build Configuration, Trained Model Manifest, Package

## 1. Component model

A **Component** is the unit of identity, contract, training, versioning and replacement.

### 1.1 Granularity
1. **System components:** each learned or hybrid primitive, Composer, Selector, Heart policy, Gate, Modulators, Error Monitor (verifier), Simulator models, Concept Formation Engine, perception encoder, each parser, Mouth renderer base, each articulator, codec, interface-layer parts, each Body tool adapter.
2. **Skill components:** each compiled skill or derived primitive.
3. **Region components:** Library regions, the Hippocampal store, episodic stores. Indices and the Knowledge Sketch are **derived components**: rebuilt, never trained.

Individual records (facts, concepts, …) are **not** components. They are addressed by record IDs and edited through record operations (`04` §10), not by training.

### 1.2 Component types (registry; extensible by MINOR version)
`interface.codec`, `interface.codespace`, `interface.property_basis`, `interface.role_registry`, `interface.relation_registry`, `interface.signature_encoder`, `perception.encoder`, `perception.parser.<modality>`, `core.primitive.<name>`, `core.skill_core`, `core.composer`, `core.selector`, `core.error_monitor.verifier`, `core.simulator.<kind>`, `core.concept_formation`, `control.heart_policy`, `control.gate`, `control.modulators`, `control.subconscious.<process>`, `memory.region`, `memory.hippocampus`, `memory.index` (derived), `memory.sketch` (derived), `skill`, `mouth.renderer_base`, `mouth.articulator.<target>`, `mouth.planner`, `body.tool_adapter.<tool>`.

### 1.3 Contract (stored per component version)

| Field | Meaning |
|---|---|
| `interface` | Input and output ports with message types (`03` §5) and the interface-layer version range |
| `dependencies` | `[{logical_id, edge_type, version_range}]` |
| `local_objective` | How the component can be trained alone: loss id, required frozen dependencies, data kinds |
| `validation_suite` | Test ids: unit, property and probe tests |
| `behavior_fingerprint` | Probe-set id + fingerprint statistics (`10` §5) |
| `invariants` | Machine-checkable rules (e.g. "outputs are typed `Claim`", "never emits untainted claims without provenance", "output `Code` blocks in range") |
| `resource_class` | Size class, memory footprint, expected FLOPs per call |
| `trust_level` | `core`, `verified`, `third_party`, `experimental` |
| `retrainable` | `independent`, `with_closure`, `frozen`, `derived` (`10` §10) |

## 2. Component IDs

### 2.1 Logical ID (stable identity)
- **Format:** `srm.<family>.<type>.<h>`, where `<h>` is the first 26 characters of the lowercase base32 (no padding) encoding of SHA-256 over the canonical JSON (RFC 8785 JCS) of the **identity metadata**:

| Identity metadata field | Note |
|---|---|
| `model_family_id` | From the Build Configuration |
| `interface_major` | Interface-layer MAJOR version |
| `component_type` | §1.2 |
| `interface_signature_hash` | SHA-256 of the canonical port list (names, message types, interface version range) |
| `structural_signature` | Shapes, ranks, layer counts, code format; **no weight values** |
| `capacity_class` | Bucketed size class (XS/S/M/L/XL by parameter count) |
| `dependency_interface_signature` | SHA-256 of the sorted list of `(dependency logical_id, dependency interface_major)` |
| `lineage_root` | SHA-256 of the creation event (event type, timestamp, parent logical IDs, build-config hash, creator) |
| `model_generation` | Integer generation of the model family |

- **Verification:** anyone holding the Manifest can recompute `<h>` from the stored identity metadata. Weights are never exposed by an ID.

### 2.2 Version ID (integrity)
`sha256:<hex>` over (logical ID ‖ SHA-256 of the canonical weight bytes ‖ sorted dependency version IDs at training time ‖ SHA-256 of the canonical training record). Version IDs form a Merkle chain through dependencies.

### 2.3 Stability rules
| Change | Effect |
|---|---|
| Retraining within the contract | **New version ID**, same logical ID |
| Behavior change within the contract (fingerprint distance between patch and minor thresholds) | New version ID, classified *minor* |
| Interface change (ports or types), or a dependency interface MAJOR change | **New logical ID**, with a `supersedes` edge from the old one |
| Split / merge | New logical IDs with `split_from` / `merged_from` lineage; the alias table maps old to new for resolution |
| Moving bytes, re-sharding, re-packaging | No change: IDs don't depend on file locations or creation order |

### 2.4 Record IDs (not components)
`uint64 = region_id ‖ local_index`, plus a record content hash. Record edits create new record versions (`04` §2.1).

## 3. Build Configuration (input; before training)

Written by the developer, validated before any build, and never mutated by training. Its hash is stored in every Manifest it produces. It is grouped into sections; every scale-dependent quantity lives here.

### 3.1 Fields (types; F0 defaults)

| Section | Field | Type | F0 default |
|---|---|---|---|
| `meta` | `model_family_id`, `model_name`, `model_generation`, `build_config_version` | str/int | `neuroforge-srm`, `srm-f0`, 1, `1.0` |
| | `target_hardware_profile` | enum | `single_gpu_24gb` |
| | `seed` | int | 1234 |
| `capacity` | `total_capacity_budget` (per section: interface, core, library_records, skills, mouth) | dict | sized to profile |
| | `active_capacity_ceiling` (params + record bytes) | int | profile-derived |
| | `max_active_fraction` | float | 0.05 |
| | `machinery_floor` (list of required component types) | list | all Core + Mouth + Error Monitor types |
| `interface` | `B`, `L`, `r`, `n_b`, `d`, `d_k` | int | 64, 64, 3, 21, 256, 128 |
| | `P_0`, `N_role`, `scalar_buckets` | int | 512, 64, 32 |
| `codec` | `input_normalization`, `output_vocab_size`, `special_tokens` | | NFC, 8192, default set |
| `perception` | `chunk_max_bytes`, `n_layers_perc`, `recurrent_state_dim`, `θ_known`, `θ_cos`, `θ_err`, `θ_parse` | | 128, 4, 256, 0.80, 0.90, 0.3, 0.5 |
| `memory` | `N_lib`, `N_hip`, `page_records`, `tiers_enabled`, `S_context`, `cand_max`, `k_ret`, `index_backend` | | sweep, 1e5, 4096, [T1, T2], 4, 4096, 64, banded_sparse |
| | `sketch_fpr`, `h_spread`, `k_spread`, `η_H`, `η_decay`, `w_max` | | 1e-3, 2, 256, 0.05, 0.001, 1.0 |
| `workspace` | `K`, `R`, `H`, `A`, `G_max`, `HS_max`, `NG_max` | int | 64, 32, 8, 3, 8, 16, 1024 |
| `epistemics` | `W`, `κ`, `κ_v`, `κ_t`, `θ_commit`, `θ_retract`, `θ_abstain`, `θ_max`, `θ_predict`, `δ_margin`, `π_cap`, `r_extrap`, `t_norm` | | 2.0, 1.0, 1.0, 2.0, 0.80, 0.60, 0.70, 0.95, 0.5, 1.0, 1.0, 0.3, min |
| | `trust_priors` (table), `modality_factors` (table), `θ_trust_gap`, `θ_conflict_mass` | | `02` §10, 0.3, 1.0 |
| | `allow_verifier_promotion_low_stakes` | bool | false |
| `core` | `D_max`, `E_max`, `k_align`, `t_align`, `θ_schema`, `θ_analogy`, `θ_parent`, verifier width/layers, composer width/layers | | 8, 512, 8, 5, 0.6, 0.6, 0.5, 256/3, 256/3 |
| `prediction` | `k_env`, `n_min`, `θ_env`, `λ_res`, `λ_res0` | | 16, 20, 0.8, 1.0, 1.0 |
| `control` | `T_max`, `base_budget`, `max_budget_per_query`, per-beat caps, floors (subconscious, error monitor), credit `η_c`, `[c_min, c_max]`, modulator gains | | 64, profile, profile, profile, 0.10/0.15, 0.1, [0.1, 2.0], defaults |
| `skills` | `d_s`, `n_layers_skill`, `r_skill`, `θ_accept`, `speedup_min`, `n_compile`, `n_domains_promote`, `θ_recompile` | | 256, 4, 8, 0.98, 2.0, 5, 3, 0.9 |
| `mouth` | `n_layers_mouth`, `d_mouth`, `V_out`, `θ_attr`, articulator set | | 4, 256, 8192, 1.5 nats, [prose, markdown, json, python, dsl, latex] |
| `body` | Enabled tools, sandbox limits, tool costs | | DSL, python_sandbox, tests, sympy, units |
| `learning.bootstrap` | Stage list, per-stage optimizer settings, acceptance targets, data mixture by category, compute budget | | S0–S7 defaults |
| `learning.online` | `η_local`, `n_local_steps`, `λ_anchor`, buffer sizes, plasticity by lifecycle | | 1e-4, 8, 1.0, 512, `04` §9 |
| `sleep` | `t_idle`, `micro_sleep_budget`, `θ_hip_fill`, `sleep_every`, `replay_mix`, `θ_mdl`, `downscale`, `w_min`, `θ_merge`, `n_ctx_causal`, re-layout settings | | defaults |
| `components` | Component granularity options, fingerprint probe sizes, validation thresholds (`ε_reg`, `δ_min`, `ε_shadow`, `d_patch`, `d_minor`) | | `10` §5 |
| `packaging` | Package format version, shard target size, compression, signing key id | | `srmpkg-1`, 256 MB, zstd, none |
| `runtime` | Batch size, precision (fp16/bf16), device map, logging level, trace recording policy | | 1–32, bf16, single |
| `evaluation` | Benchmark suite ids, baselines, seeds | | `12` |

### 3.2 Validation rules (checked before build)
- `r · n_b ≤ B`.
- `L ≤ 256` (one byte per block).
- The `machinery_floor` component types are all present.
- `K ≥ 2·A + G_max`.
- The sum of floors < 1.
- Every threshold lies in its valid range.
- `active_capacity_ceiling ≥ resident Core + Mouth size estimate`.
- The interface MAJOR version is compatible with any loaded components.

## 4. Trained Model Manifest (output; generated, never hand-edited)

### 4.1 Structure
Hierarchical, content-addressed (a Merkle tree):
- `manifest/root.json`: identity, capacity summary, interface version, build-config hash, hashes of the sub-manifests, and the previous root hash (history chain).
- `manifest/components/<type_group>.json`: component entries.
- `manifest/graph.json`: the dependency graph.
- `manifest/regions.json`: region summaries.
- `manifest/history.json`: version graph and rollback pointers.
- `manifest/validation.json`: suites, probes, coverage map.
- `manifest/safety.json`: integrity and safety metadata.

**Canonicalization:** RFC 8785 JCS; hashing: SHA-256. `root_hash` is SHA-256 over the canonical root, which includes the sub-manifest hashes.

### 4.2 Contents

| Group | Fields |
|---|---|
| Identity | `architecture: "final-srm"`, `architecture_version`, `model_family_id`, `model_name`, `model_generation`, `interface_version`, `abi_hash`, `build_config_hash`, `created_at`, `root_hash`, `parent_root_hash` |
| Capacity | Realized total capacity per section (parameters, records, bytes); active-capacity ceiling; measured active-capacity percentiles (p50/p95/p99) on the evaluation suites; per-beat caps |
| Components | For each component: `logical_id`, identity metadata (§2.1), `version_id`, `type`, `contract` (§1.3), `size`, `retrainable`, `frozen`, `trust_level`, `envelope_summary` (centroid count, coverage, accuracy), `fingerprint_hash`, `shard_paths` |
| Dependency graph | Edges `{from, to, type, version_range}`. Types: `calls`, `reads_interface`, `trained_with`, `verified_by`, `renders_with`, `indexes`, `derived_from`, `merged_from`, `split_from`, `supersedes` |
| Emergence and specialization | For emerged components: `created_by` (sleep cycle id / compilation event), `source_trace_families`, `parents`, `usage_profile` (task families → recruitment frequency), `activation_cluster`, `functional_label` (auto-generated, descriptive only), `promotion_history` |
| Training history | Per version: objectives, data-category proportions (`02` §13), compute (GPU-hours, peak memory), seeds, validation results, the version it replaced, the operator (bootstrap / sleep / retrain job id) |
| Regions | Region id, record counts by kind and lifecycle, page count, tier hints, index versions, sketch version |
| Coverage map | For each validation/regression test: the component logical IDs it recruited (recorded during validation runs) |
| Compatibility | Interface version, primitive set version, manifest schema version, `package_format`, migration adapters available |
| Integrity and safety | Per-shard SHA-256; signatures; safety evaluation results per component and per system; license; provenance summary; red-team notes |
| History | Previous root hashes, component version pointers, rollback targets |

### 4.3 Generation
- The Manifest is produced by the build system at the end of bootstrap, after each deep sleep that changes components or regions, and after each committed surgery (`10` §6).
- It is **only** written by these processes. Hand edits invalidate the root hash and are refused by the loader.

### 4.4 Separation from the Build Configuration
| | Build Configuration | Trained Model Manifest |
|---|---|---|
| Written | Before training, by people | After training/sleep/surgery, by the system |
| Describes | Intent and limits | The realized structure |
| Mutability | Immutable per build | Versioned chain |
| Relationship | Its hash is referenced by the Manifest | Never copied into the Build Configuration |

## 5. Package format (`srmpkg-1`, directory form)

```
<model>.srmpkg/
├─ header.json                      # format version, root_hash, signatures, created_at
├─ manifest/                        # §4 (root.json, components/*.json, graph.json, regions.json,
│                                   #     history.json, validation.json, safety.json)
├─ interface/<version_id>/          # frozen interface layer
│    ├─ codespace.safetensors       # projection W/β, embedding tables E
│    ├─ registries/*.parquet        # roles, properties, relations, types (+ codes, dense)
│    └─ signature_encoder.safetensors
├─ components/<logical_id>/<version_id>/
│    ├─ weights.safetensors
│    ├─ contract.json
│    ├─ fingerprint.safetensors
│    └─ training_record.json
├─ regions/<region_id>/<version_id>/
│    ├─ records.parquet             # columnar record table (04 §2.1)
│    ├─ payloads.safetensors        # dense, keys, schema numeric payloads
│    ├─ payloads.parquet            # structured payloads (schemas, engrams, procedures, episodes)
│    ├─ links.parquet               # CSR association edges
│    ├─ provenance.parquet
│    └─ wal/                        # write-ahead log segments (live models only)
├─ derived/
│    ├─ index/<region_id>/<version_id>/{content,signature,triple,alias}.*
│    └─ sketch/<version_id>.bin
├─ validation/
│    ├─ suites/<suite_id>/…         # tests (SEF task/question records)
│    ├─ probes/<component>/…        # probe sets for fingerprints
│    └─ coverage.parquet
├─ traces/        (optional)        # interface traces for retraining (10 §3); not distributed by default
├─ episodic/      (optional)        # privacy-scoped; separable
└─ history/       (optional)        # older component/region versions or references to them
```

- **Formats:** tensors in safetensors; tables in Parquet; metadata in JSON (canonical when hashed).
- **Loading procedure:**
  1. Read `header.json`; verify the signature (if any) and the root hash.
  2. Load the Manifest; check compatibility (interface version, package format).
  3. Memory-map the interface layer and the resident Core and Mouth shards.
  4. Register regions (records lazily paged by tier policy).
  5. Load indices and sketch (or rebuild if missing or mismatched).
  6. Verify each shard's hash on first access.
- A single-file container (aligned tar) is deferred. The directory form is normative for F0.

## 6. Runtime activation accounting
For each query, the runtime records:
- resident parameters;
- recruited records/pages (bytes);
- invoked skills (parameters);
- Mouth parameters used;
- FLOPs (estimated and profiled);
- bytes moved.

Active fraction = (active parameters + active record bytes in parameter-equivalents) / total. These measurements produce the Manifest's measured active-capacity percentiles and the efficiency metrics (`12` §4).

## 7. Diagram C — Packaging and component architecture

```
 SRM PACKAGE ───────────────────────────────────────────────────────────────────────────
 │ HEADER: format ver · root-manifest hash · signatures
 │ MANIFEST (Merkle root) ─┬─ identity · capacity · build-config hash · interface ver · abi hash
 │                         ├─ COMPONENTS: LogicalID ⟂ VersionID · contract · trust · envelope
 │                         ├─ DEPENDENCY GRAPH (calls · reads-interface · trained-with · verified-by ·
 │                         │                    renders-with · indexes · derived/merged/split/supersedes)
 │                         ├─ emergence/specialization · training history (data categories) · coverage map
 │                         └─ compatibility · integrity/safety · version history (rollback pointers)
 │ INTERFACE (frozen ABI): codec · code space · role vectors · Property Basis · primitive signatures
 │ CORE:      [prim:ALIGN] [prim:EXPLAIN] [prim:PREDICT_STEP] … [composer] [selector] [heart] [gate]
 │            [error-monitor/verifier] [simulator] [concept-formation]
 │ SKILLS:    [skill:list-reorder v3] [skill:unit-convert v1] [skill:dsl-fold-pattern v2] …
 │ LIBRARY:   [region:objectworld-vehicles] [region:dsl-lists] [region:factstream-geo] …  (records inside)
 │ DERIVED:   index shards · knowledge sketch      MOUTH: [renderer-base] [art:markdown] [art:json] [art:python]
 │ VALIDATION: contracts · suites · probe sets · fingerprints · coverage map   EPISODIC (optional)
 └───────────────────────────────────────────────────────────────────────────────────────

 RUNTIME (one query):  resident = INTERFACE + CORE (small)
                       recruited = [region:dsl-lists] [skill:dsl-fold-pattern] [art:python] (+ prefetch)
                       dormant   = everything else (on host RAM / disk, never touched)
                       ↑ Heart enforces active-capacity ceiling from Build Configuration
```
