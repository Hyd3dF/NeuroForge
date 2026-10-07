# 04 — Memory

## 1. Memory systems at a glance

| System | Holds | Write speed | Lifetime | Component type |
|---|---|---|---|---|
| **Workspace** | Active claims, goals, referents, hypotheses | Every beat | One query/session | (part of Core, `05` §1) |
| **Hippocampal Index** | NEW records and episodes (one-shot) | Immediate | Until consolidated or decayed | `memory.hippocampus` |
| **Cortical Library** | Concepts, engrams, procedures, skills (refs), entities, open questions | Sleep (plus record edits) | Long-term | `memory.region` (one per region) |
| **Association Field** | Typed weighted links between records | Hebbian (online) + sleep | Long-term | Stored inside regions (link tables) |
| **Indices** | Content, signature, structured-triple and alias indices | Rebuilt or updated incrementally | Derived | `memory.index` (derived) |
| **Knowledge Sketch** | Counting Bloom filter of stored keys | On write/tombstone | Derived | `memory.sketch` (derived) |
| **Provenance store** | Source registry, evidence events (compacted) | On ingestion | Long-term | Part of the `memory.region` metadata |

## 2. Record kinds and payloads

### 2.1 Common record columns (columnar storage per region)

| Column | Type | Meaning |
|---|---|---|
| `record_id` | uint64 | `region_id (16 bits) ‖ local_index (48 bits)` |
| `kind` | enum | ENTITY, CONCEPT, ENGRAM, PROCEDURE, SKILL_REF, EPISODE, OPEN_QUESTION, PROPERTY_REF |
| `identity_code` | uint8[B] | Random seeded code (`03` §2.3) |
| `content_code` | uint8[B] | Projected code |
| `dense_key` | float16[d_k] | Fine scoring (learned key head over `dense`) |
| `dense` | float16[d] | Content embedding |
| `signature_code` | uint8[B] or null | Relational signature (concepts, procedures, episodes) |
| `context_segments` | uint8[S, B] | Up to `S` context codes (version, domain, time bucket, world); F0 `S` = 4 |
| `e_plus`, `e_minus` | float32 | Evidence accumulators |
| `state` | enum | Epistemic state (`06` §5) |
| `lifecycle` | enum | NEW, CORROBORATED, USED, CONSOLIDATED, STABLE, CONTESTED, DEPRECATED, DECAYED |
| `provenance_ptr` | uint64 | Pointer into the provenance table (source ids, spans, root sources) |
| `usage_count`, `last_used`, `value_sum`, `cost_sum` | numeric | Economics |
| `plasticity` | float16 | Learning-rate multiplier (by lifecycle stage) |
| `page_id`, `tier` | uint32, enum | Physical location |
| `version`, `supersedes` | uint32, record_id | Record versioning (edits create new versions) |
| `payload_ptr` | uint64 | Kind-specific payload (below) |

### 2.2 CONCEPT payload (factored schema)

| Field | Meaning |
|---|---|
| `parents`, `children` | is-a links (record ids) |
| `slots` | List of `(role_id, filler)`. A filler is `PropertyConstraint`, `PartSpec` (role, concept, count range), `RelationPattern` or `FunctionSpec` |
| `PropertyConstraint` | `property_id`, constraint (value / range / enum set / bool), **variability profile** (§2.2.1), `w_def` (defining weight), evidence |
| `invariances` | `(transform_id, preserves_identity, confidence)` |
| `contrasts` | `[(sibling_concept_id, [(property_id, discriminative_weight)])]` |
| `overhypothesis` | For parent concepts: per-property distribution of children's variability (§2.2.2) |
| `instance_stats` | Count of instances, last update |
| `structure_bundle` | Sparse COO of the role⊗filler bundle (for exact relational queries) |

#### 2.2.1 Variability profile (per property `π` in concept `C`)
| Value type | Profile |
|---|---|
| Scalar | Log-space running mean `μ`, std `σ`, min/max, count `n` (Welford updates) |
| Enum | Dirichlet counts `α_v` |
| Bool | Beta(`a`, `b`) |

- **Normalized variability:** `v_C(π) = σ_C(π) / σ_global(π)` for scalars, normalized entropy for enum/bool.
- **Defining weight:** `w_def(π, C) = clip(1 − v_C(π), 0, 1) × presence_rate(π, C)`.

#### 2.2.2 Overhypothesis (for a parent `P`)
For each property `π`, store mean and variance of `v_child(π)` across `P`'s children. This is the prior variability for a *new* child concept of `P`, used by the Concept Formation Engine (`05` §9).

### 2.3 ENGRAM payload (fact)
`subject (record_id)`, `relation_id`, `object (record_id or Value)`, qualifiers (time interval, version, condition pattern, location), polarity, modality, attribution (record_id or null), and `derived_from` links when the engram came from consolidated reasoning.

### 2.4 PROCEDURE payload
- A `CircuitGraph` template (`05` §5.1) with typed `Hole`s.
- Goal pattern, input/output types, pre/postconditions, invariants.
- Usage statistics: executions, success rate, mean cost.

### 2.5 SKILL_REF payload
Points to a skill **component** (`09` §1) by logical ID and version ID, plus the interface signature, validity-envelope summary and track record. The skill's weights live in the component shard, not in the region.

### 2.6 EPISODE payload
Context code, timestamp, source, list of involved record ids, novel residue (scene fragments not yet integrated), outcome (for task episodes: goal, circuit-graph trace, verified success, cost), and replay priority.

### 2.7 ENTITY payload
Canonical name, aliases, type (concept id), external ids, provisional flag.

### 2.8 OPEN_QUESTION payload
Pattern with variables, scope, as-of date and source (from `known_unknown`, `02` §4.11).

## 3. Regions, pages and tiers
- **Region** = a cluster of records that tend to be co-activated. Initially it is assigned by data category and concept subtree; it is re-clustered during sleep (`08` §5). A region is a component (`memory.region`) with its own versioned shard.
- **Page** = a fixed-size slice of a region (`page_records`, F0 default 4096 records). Pages are the unit of tier movement and prefetch.
- **Region routing** (where a consolidated record goes): each region keeps a centroid (mean dense vector) and a data-category set. A record goes to the region maximizing `cos(dense, centroid) + λ_cat · [category ∈ region categories]`. A new region is created when the best score < `θ_new_region` or the region exceeds `region_max_records`. All values are configurable.
- **Privacy scoping:** records with `privacy_scope ∈ {user_private, org_private}` are routed only to regions with the same scope (per user/org). They are never consolidated into shared regions and never used to train shared components unless policy explicitly permits it.
- **Tiers:** `T0` = GPU memory, `T1` = host RAM, `T2` = disk (memory-mapped). F0 implements T1 and T2 with optional T0 caching. Promotion and demotion follow Heart policy (`07` §1.6).

## 4. Indices (all derived; rebuildable from regions)

### 4.1 Content index (banded LSH over content codes)
- **Band layout:** `n_b` bands of `r` blocks each (F0: 21 × 3 = 63 of the 64 blocks; the layout is fixed by the interface version).
- **Key** for band `j`: `hash(j, c[b_j1], …, c[b_jr])`. Posting list = record ids.
- **Query:** union of postings over all bands → candidate set → cap at `cand_max` (F0 4096) by vote count (number of matching bands) → exact `ov` and `dense_key` scoring → top `k_ret`.
- **Recall model:** if each block matches with probability `p` (a function of cosine similarity), then `P(retrieved) = 1 − (1 − p^r)^{n_b}`. For `p = 0.7`, `r = 3`, `n_b = 21`: ≈ 0.9999. For random records (`p = 1/L`), the false-candidate rate per band is `L^{−r}` ≈ 3.8e-6.
- **Expected candidates per query** ≈ `N · n_b · L^{−r}` + true neighbours. For `N = 1e7`, about 800, so lookup cost is roughly independent of `N` until `N` approaches `L^r`. When that happens, increase `r` or `L` (a MAJOR interface change; planned as a capacity class).
- **Context filter:** candidates whose context segments contradict the query context (version mismatch) are dropped before scoring.

### 4.2 Signature index
Same banded structure over `signature_code`, for relational-signature (analogy) retrieval.

### 4.3 Structured triple index (exact)
Hash maps per region:
- `(subject, relation) → [engram ids]`
- `(relation, object) → [engram ids]`
- `subject → [engram ids]`
- `concept → [instances]`
- `parent → [children]`

These provide exact factual lookup and contradiction detection, independent of approximate codes.

### 4.4 Alias index
Normalized surface form → entity/concept/property/relation ids (exact), plus a small dense ANN over alias embeddings (fuzzy).

### 4.5 Backend abstraction
The index interface has two implementations: `banded_sparse` (default, as above) and `dense_ann` (an approximate nearest-neighbour library over `dense_key`; optional in F0, used as a baseline and fallback).

## 5. Retrieval

### 5.1 Stage 1 (cheap, many)
`RETRIEVE(query_code, query_dense, kind_filter, context, k)` → content index → context filter → score `s = α·ov/B + (1−α)·cos(dense_key)` (α configurable, F0 0.5) → top-k.

### 5.2 Structured retrieval
`RETRIEVE_PATTERN(pattern)` with variables → structured triple index → candidate bindings. Used for exact facts and joins.

### 5.3 Signature retrieval
`RETRIEVE_SIG(signature_code, k)` → signature index (analogy candidates across domains).

### 5.4 Cleanup (attractor completion)
Given a noisy query `x` and candidate set `C` (keys `K_C`, values `V_C`), iterate `x ← V_Cᵀ softmax(β · K_C x)` for `t_clean` steps (F0: β = 8, t = 3). This is a modern-Hopfield update restricted to `C`. It is used after unbind-by-role and for partial cues.

### 5.5 Stage 2
Structural alignment (`05` §6) runs on at most `k_align` survivors.

## 6. Association Field

### 6.1 Edge types

| Type | Admission | Used by |
|---|---|---|
| `is_a`, `part_of`, `has_property` | Definitions, examples, consolidation | Inheritance, recognition |
| `causes`, `prevents`, `enables` | **Mechanism or intervention evidence only**, or invariance across ≥ `n_ctx_causal` contexts at sleep (`08` §5) | PREDICT-STEP, REGRESS |
| `associated_with` | Co-occurrence and observational evidence | Priming only (never causal prediction) |
| `analogous_to` | Verified analogies | Analogy retrieval |
| `used_with` | Co-use in successful circuits | Composer retrieval |
| `contradicts` | Contradiction records | Truth maintenance |
| `derived_from`, `trained_from` | Provenance and lineage | Correction propagation |
| `co_activated` | Hebbian | Priming, region clustering |

### 6.2 Storage and update
- **Storage:** CSR adjacency per region (`dst`, `type`, `weight`, `evidence`), plus a cross-region edge table.
- **Spreading activation:** `a_j^{t+1} = clip(γ·a_j^t + Σ_{i→j} w_ij · g_type · a_i^t, 0, 1)`, then keep the top `k_spread` (F0 256), for `h_spread` hops (F0 2). Gains `g_type` are configurable; `contradicts` and `derived_from` don't spread priming.
- **Hebbian update** after each beat: `Δw_ij = η_H · m · a_i · a_j − η_decay · w_ij`, clipped to `[0, w_max]`. Here `m` comes from the modulators (`07` §3): positive after verified success, scaled by ACh during encoding.

## 7. Knowledge Sketch
- **Structure:** a counting Bloom filter over keys `(entity_id)`, `(entity_id, relation_id)`, `(concept_id)`, `(concept_id, property_id)`, plus normalized alias strings.
- **Sizing:** `m = ⌈−n ln p / (ln 2)²⌉` counters and `k = ⌈(m/n) ln 2⌉` hashes, for the target false-positive rate `p` (F0 1e-3) and expected key count `n`. 4-bit counters.
- **Semantics:** if any counter is 0, the key was **never stored** → UNKNOWN-ABSENT (no false negatives). Writes increment and tombstones decrement.
- **Invariant:** every ENGRAM, CONCEPT and ENTITY write updates the sketch in the same transaction (`10` §6 checks this). Facts may live **only** in engrams (the fact-placement rule, `06` §8).

## 8. Hippocampal Index
- **Same record format** as the Library, in a dedicated fast-write store with capacity `N_hip`.
- **Pattern separation:** the episode context code is `sp(expand(h_context))`. `expand` is a fixed random sparse projection to a higher dimension followed by keeping the top-k, then re-blocked into `B × L`. Similar contexts therefore get distinct episode codes.
- **Writes are append-only** and become retrievable immediately: the content and triple indices of the hippocampal store are updated synchronously.
- **Decay:** `priority = surprise × utility × uncertainty`. Records that are not consolidated and fall below `θ_decay` after `T_decay` sleep cycles are DECAYED (removed, with a sketch decrement).
- **Consolidation** moves records into Library regions during sleep (`08` §5).

## 9. Lifecycle transitions

| From → To | Criterion (configurable thresholds) |
|---|---|
| NEW → CORROBORATED | ≥ 2 independent root sources, **or** a verification/test event with belief ≥ `θ_commit` |
| CORROBORATED → USED | Contributed to ≥ `n_use` (F0 2) committed answers that were not later retracted |
| USED → CONSOLIDATED | Integrated during sleep (written into a Library region) |
| CONSOLIDATED → STABLE | Age ≥ `n_sleep_stable` sleep cycles with no contrary evidence, and usage ≥ `n_stable_use` |
| any → CONTESTED | Both `e⁺` and `e⁻` ≥ `θ_conflict_mass` (F0 1.0), or an unresolved contradiction |
| CONTESTED → (prior stage) | Contradiction resolved with a trust gap ≥ `θ_trust_gap` |
| any → DEPRECATED | `d ≥ θ_retract` with strong evidence, or superseded by a corrected version (kept for provenance) |
| NEW → DECAYED | Not consolidated, low priority, past `T_decay` |

**Plasticity multiplier by stage:** NEW 1.0, CORROBORATED 0.7, USED 0.5, CONSOLIDATED 0.2, STABLE 0.05 (configurable).

## 10. Memory operations API (logical)

| Operation | Effect |
|---|---|
| `write(record, region or hippocampus)` | Append; update indices and sketch; log an event |
| `add_evidence(record_id, EvidenceEvent)` | Update `e⁺`/`e⁻`, state, lifecycle |
| `link(src, dst, type, weight)` / `hebbian_update(active_set)` | Association Field |
| `tombstone(record_id, reason)` | Mark DEPRECATED; decrement sketch; keep provenance |
| `new_version(record_id, changes)` | Create a successor record (`supersedes`) |
| `retrieve`, `retrieve_pattern`, `retrieve_sig`, `cleanup`, `spread` | §5–6 |
| `consolidate(batch)` | Sleep only (`08` §5) |
| `prefetch(page_ids)` / `promote` / `demote` | Tier movement (Heart-directed) |

## 11. Consistency and persistence
- **Write-ahead log:** every mutation is appended to a region event log. Region shards are snapshots plus a log. Sleep compaction folds the log into a new region version (the component version changes).
- **Read/write separation:** queries read a consistent snapshot. Writes from the current query become visible to that query immediately (read-your-writes) and to other sessions after commit.
- **Formats:** record columns are stored as Parquet (Arrow); dense payloads and skill weights as safetensors; links as Parquet CSR (`09` §5).
