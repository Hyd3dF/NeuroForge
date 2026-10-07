# 11 — System Integration: How All Subsystems Connect

## 1. Connection matrix

All connections pass typed messages (`03` §5). "Calls" means synchronous invocation inside a beat; "writes/reads" refers to shared stores.

| Subsystem | Consumes | Produces | Calls / uses | Read/written by |
|---|---|---|---|---|
| Codec | Bytes | Normalized bytes; output tokens (Mouth) | — | Perception, Mouth |
| Perception encoder + L0 | Bytes/chunks | `h_chunk`, `ε0` | Content index (recognition) | Gate (salience), Modulators (ACh) |
| Parsers | Chunks/segments | SceneGraph, Goals, triage tags | Alias index, registries | Hippocampus (writes), Gate |
| Hippocampal Index | Records, episodes | Records (retrieval) | Indices, Sketch | Triage, Sleep, Retrieval |
| Cortical Library (regions) | Consolidated records | Records | Indices, Sketch, tiers | Retrieval, Sleep, record ops |
| Indices / Sketch | Record writes | Candidates; absent/present | — | RETRIEVE, triage, decision procedure |
| Association Field | Activations | Spread activations | Region link tables | Subconscious, Hebbian updates |
| Subconscious | Workspace cues, perception fragments | SUGGESTED items, prefetch bids | Spread, RETRIEVE, skills, ALIGN | Preconscious buffer → Gate |
| Thalamic Gate | Buffer items, goals | Admissions, routing decisions | — | Workspace |
| Workspace | Admissions, node outputs, evidence events | Claims, hypothesis sets, goal states | Evidence algebra, TMS | All Core processes |
| Composer | Goals, workspace | CircuitGraph expansions | RETRIEVE (procedures, signatures), ALIGN, policy | Selector |
| Selector | Workspace state, ready nodes | Actions + bids, STOP | VOC head, critic | Heart |
| Primitives / skills | Bound inputs | Claims, PredictionRecords | Memory, Body (via TEST) | Composer nodes |
| Error Monitor | Claims, prediction records | EvidenceEvents, contradictions, conflict flags | Verifier, memory, Body | Workspace |
| Simulator | States, artifacts | Predictions, counterexamples | PREDICT_STEP, Body | Error Monitor, Composer |
| Concept Formation Engine | Examples, residual-dominant recognitions | CONCEPT records, provisional properties | ALIGN, ABSTRACT, RETRIEVE | Hippocampus |
| Heart | Bids, budget, modulators | Funding decisions, tier moves, halting | Cost model | Every process |
| Modulators | Beat statistics | DA/ACh/NE/5-HT scalars | — | Heart, Gate, Selector, learning |
| Mouth | UtterancePlan | Text/code/data | Articulators, unparsers | Output |
| Body | Tool calls | OBSERVED results | Sandboxes | Error Monitor, Simulator |
| Learning (online) | Verification outcomes, beat activity | Component micro-updates, link updates | Local buffers | Components, regions |
| Sleep Engine | Hippocampus, episodes, regions | New region/component versions, Manifest updates | All memory ops, ALIGN/ABSTRACT, distillation | Manifest |
| Component system | Manifest, Build Configuration | Loaded components, surgery, rollback | Package loader, validation harness | Runtime |

## 2. Per-beat execution order (normative)

For beat `t` of an active query:

| Phase | Steps |
|---|---|
| 0. Modulate | Update DA/ACh/NE/5-HT from beat `t−1` statistics |
| 1. Perceive | If input remains: chunk → encode → L0 predict → recognize (known → pointer) → parse novel → triage → hippocampal writes → novel or salient items to the preconscious buffer |
| 2. Background | Subconscious (within its floor): priming spread, prefetch bids, stage-1 matching on new fragments, habit proposals → buffer |
| 3. Diastole | Collect bids: Gate admissions, Selector actions (with VOC), Composer expansions, verifications, Simulator/Body calls, retrievals, Concept Formation Engine, Mouth (if ready) |
| 4. Systole | Heart allocates (`07` §1.4), then dispatches in this fixed order: (a) Gate admissions → Workspace; (b) memory retrievals and tier moves; (c) Composer expansions; (d) circuit node executions (primitives and skills, batched); (e) Simulator and Body calls; (f) Error Monitor verifications |
| 5. Epistemic update | Apply evidence events → Support recomputation for affected hypothesis sets → belief propagation over the justification DAG → taint propagation and promotion → retraction and nogoods → channel pruning |
| 6. Maintain | Deduplicate and evict workspace slots; update node states (READY/DONE/FAILED); queue local repairs |
| 7. Learn (online) | Credit updates (`c_p`), Selector TD update (if enabled), Hebbian update for co-active records, verification-as-teacher buffer appends, envelope updates |
| 8. Decide | Goal decision procedure (`06` §6) for each goal; Selector STOP check; Heart budget check |
| 9. Trace | Append beat events to the Answer Record; interface-trace recorder hooks (if enabled) |

When every goal is decided (or the query must halt): build the UtterancePlan → Mouth → output. Then post-processing.

## 3. Query lifecycle (state machine)

```
RECEIVED → INTAKE (codec, perception, parse, goals) → REASONING (beats 1..T)
   → DECIDED (per goal: KNOWN-class / PREDICTION-class / ambiguous / INSUFFICIENT / UNKNOWN-* / CONTESTED / UNRESOLVED)
   → PLANNED (UtterancePlan + taint check) → RENDERED (Mouth) → DELIVERED
   → POST (episode write, local learning flush, consolidation queue, trace sampling)
```

- **Multi-turn sessions:** the Workspace persists (claims decay to DORMANT across turns unless referenced). Hippocampal writes from earlier turns are retrievable immediately.
- **Background:** micro-sleep runs only when no query is active (or within its budget share). Deep sleep runs offline and blocks surgery on the same model.

## 4. Degradation and error handling

| Condition | Behavior |
|---|---|
| Index unavailable or stale | Fall back to structured triple index + hippocampal store; mark retrieval-based beliefs with lower `hit_strength`; rebuild queued |
| Budget exhausted | Undecided goals → UNRESOLVED; render partial results |
| Tool failure | Error signature recorded as an observation; goal may become UNRESOLVED or trigger repair |
| Neural renderer failure / attribution failure | Template renderer fallback (`07` §5.5) |
| Learned component outside envelope | Outputs EXTRAPOLATED; decision procedure treats them as no basis for factual answers |
| Contract violation at runtime | Component output discarded; incident logged; automatic rollback evaluated (`10` §7) |
| Manifest invalid | Load refused (`10` §11) |

## 5. Concurrency model (F0)
- **One query per worker process.** Regions are shared read-only snapshots. Writes go to a per-session hippocampal overlay and are merged into the shared hippocampal store at commit (append-only, so conflicts are only duplicates and are deduplicated by content hash).
- Sleep and surgery take an exclusive lock on the model version they produce (copy-on-write; readers keep the old root).

## 6. Worked traces

### 6.1 Factual question, known
"What is the capital of Freedonia?" (FactStream)
1. Parse → Goal `(ent:freedonia, rel:capital, ?x)`.
2. Sketch: `(freedonia, capital)` present.
3. Composer: a trivial circuit `RETRIEVE_PATTERN`.
4. Engram with `b = 0.93`, REMEMBERED, untainted.
5. Decision: KNOWN-class (REMEMBERED).
6. Plan: answer + citation.
7. Mouth: template or neural render.

Cost: perception + 1–3 beats + a small render.

### 6.2 Factual question, unknown
"What is the population of Zembla?" where Zembla is a withheld entity.
1. Sketch: `(zembla)` absent → **UNKNOWN-ABSENT** at beat 1.
2. Mouth: "I don't know. I have no information about Zembla." No reasoning budget is spent.

If Zembla exists but has no population fact: the Sketch shows `(zembla)` present and `(zembla, population)` absent → UNKNOWN-ABSENT for that attribute, and the Mouth reports known facts about Zembla plus "its population is not in my knowledge".

### 6.3 Novel object (wheel-style)
1. Parse the scene → stage-1 retrieval → ALIGN with `wheel` (defining structure matched) → **DERIVED** membership.
2. ALIGN with `car_wheel` → size z-score far outside → rejected, with the DiffRecord kept.
3. REGRESS/PREDICT_STEP over causal schemas (`high load → heavy vehicle`, `shock strut + airport → landing gear`) → hypotheses `{aircraft_wheel, truck_wheel, RESIDUAL}`, each PREDICTED with evidence lists.
4. Support ranking → `aircraft_wheel`. Taint is removed only if a part-of chain `aircraft → landing_gear → wheel` is REMEMBERED and the alignment matches (DERIVED); otherwise it is rendered as a prediction.
5. Episode written. If similar residuals recur, the Concept Formation Engine creates `aircraft_wheel`.

### 6.4 Programming task (DSL / Python subset)
"Write a function returning the k largest distinct values."
1. Goal: `ImplRepr: List[int] × int → List[int]` + generated property tests.
2. Composer: retrieve procedure schemas (`dedupe → sort → take`) → instantiate; holes filled via type-directed synthesis.
3. Simulator: generate counterexamples (duplicates, k > n, empty list) → Body test runner.
4. A failure (k > n) → local repair of the `take` node (bounds handling) → tests pass → **TESTED**.
5. Mouth: deterministic unparser → Python code.
6. Episode stored; recurring pattern → procedure schema (and later a skill).

### 6.5 Mathematics
1. Parse the word problem → SceneGraph (quantities, relations).
2. Retrieve a procedure (rate problem) → circuit of TRANSFORM (SymPy) nodes.
3. Each step is DERIVED (exact). Body numeric check → TESTED.
4. Mouth renders the solution steps from the Answer Record.

### 6.6 Planning
1. Goal state → REGRESS recursively → plan graph (procedure under construction).
2. Simulator rollout checks preconditions; risky steps (high `u`) get verification budget; contingencies go into channels.
3. Output: a plan graph rendered as steps with flagged assumptions (CONJECTURED) and predictions.

### 6.7 Creative task
1. The goal is marked `fictional` context: claims live in a FICTION channel and are never asserted as world facts.
2. NE raises exploration; recombination (analogy, blending via BUNDLE + CLEANUP) proposes candidates.
3. The verifier checks coherence with user constraints, not truth.
4. Mouth renders in creative style; no factual claims are made outside the frame.

## 7. Startup and shutdown
- **Startup:** load package (`09` §5) → verify → build the runtime graph of components from the Manifest dependency graph → warm resident shards → ready.
- **Shutdown:** flush hippocampal overlays, logs and traces; optionally run micro-consolidation; no Manifest change unless sleep or surgery ran.
