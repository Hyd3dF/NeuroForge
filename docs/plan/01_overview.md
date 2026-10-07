# 01 — Overview: Final SRM from Input to Output

## 1. Lineage

| Generation | Central idea | What survives in Final SRM |
|---|---|---|
| **EGM** (Epistemic Graph Machine) | Reasoning is the construction of a bounded workspace of *claims*, each carrying justification, evidence and assumption labels; output renders only committed claims | Claim workspace; evidence algebra (belief / disbelief / uncertainty); justification graph; truth maintenance (retraction, nogoods); hypothesis channels; information-bottlenecked verifier; render-from-commitments; value of computation |
| **SRM v1** (Sparse Recruitment Machine) | Recruit a tiny, need-proportional set of stored units from a huge dormant library under an explicit compute economy; learn locally, consolidate offline | Heart (compute economy); Hippocampal Index + Cortical Library; content-addressed sparse recruitment; Association Field; Knowledge Sketch; Subconscious; Thalamic Gate; Selector; Modulators; Mouth; Body; sleep consolidation; habit compilation; recognition-by-recall |
| **Final SRM** | Intelligence lives in reusable machinery (primitives + composer + epistemics + learning), not in stored pattern volume; predictions are hypotheses until promoted; the model is a set of versioned components | Everything in this plan |

## 2. What changed from SRM v1

| Area | SRM v1 | Final SRM |
|---|---|---|
| **Where intelligence lives** | Implicit; spread across circuits and the library | An explicit **Intelligence Core**: domain-general **primitives**, a **Composer** that assembles them into temporary circuits per task, EGM workspace dynamics, learning mechanisms (`05`) |
| **Representation** | An assembly is a sparse code plus an opaque payload | **Factored concept schemas** over a shared **Property Basis** using sparse block-code binding (`03`, `04`) |
| **Few-shot learning** | Writes plus consolidation | A dedicated **Concept Formation Engine**: parse, generalize across examples, inherit from a parent, apply variability priors and the size principle (`05` §9) |
| **Prediction** | Mainly perceptual prediction plus EGM claims | A **seven-level Predictive Hierarchy** with typed **Prediction records**, **taint tracking** and **validity envelopes** (`06`) |
| **Choosing an answer** | Commit by belief threshold | **Best-supported, not most probable.** Generators only propose; evidence decides; every hypothesis set has an explicit residual "novel" hypothesis (`06` §4) |
| **Specialization** | Circuits recruited from a library | The specialist is **assembled during the task** as a circuit graph, and compiled into a skill only after verified repeated success (`05` §5, `08` §6) |
| **Modularity** | Not addressed | A frozen **interface layer**, Components with persistent IDs and contracts, Build Configuration, generated Trained Model Manifest, sharded package, transactional surgery and rollback (`09`, `10`) |
| **Small models** | Not addressed | A **machinery floor**: every scale gets the full machinery; scale changes knowledge, skills, workspace size and depth (§8) |
| **Data** | Generic "input" | The **SRM Experience Format** (SEF): typed structured records with provenance and uncertainty (`02`) |
| **Merged or removed** | — | Sensorium and Recognition Gate become **Perception & Parsing**; Conflict Monitor is folded into the **Error Monitor**; "circuits as low-rank modulations" survive only as **compiled skills** |

## 3. Thesis and the four required problems

**Thesis.** Final SRM solves problems by:
1. **parsing them into structure**;
2. **matching that structure** against a large but mostly dormant library of factored concepts and skills;
3. **assembling a temporary computation circuit** from a small set of domain-general primitives.

Predictions stay typed hypotheses until evidence promotes them to knowledge. Everything runs under an explicit compute economy, and the system is packaged as versioned components with contracts.

| Required problem | Mechanisms (location) |
|---|---|
| 1. Strong generalization from little data | Factored schemas over the Property Basis (`03` §3, `04` §2.2); structural parsing (`03` §8); two-stage matching (`05` §6); Concept Formation Engine with variability priors (`05` §9); primitives + Composer (`05` §4–5) |
| 2. Enormous total capacity, small active compute | Content-addressed recruitment (`04` §5); small resident Core with dormant library (§7); Heart hard caps (`07` §1); recognition-by-recall (`03` §7) |
| 3. Knowledge separated from prediction; a real UNKNOWN | Typed prediction records and taint (`06` §2–3); validity envelopes (`06` §3); Support ranking (`06` §4); epistemic lattice (`06` §5); Knowledge Sketch (`04` §7) |
| 4. Safe component-level retraining after packaging | Frozen interface layer (`03` §1); Components, IDs and contracts (`09` §1–2); Manifest and package (`09` §4–5); retraining on interface traces (`10` §2–3); transactional surgery and rollback (`10` §6–7) |

## 4. Design principles

1. **Pay only for surprise.** Known input becomes pointers; only novel content is processed deeply.
2. **Recruit; don't broadcast.** Computation follows content addressing; dormant structures cost nothing.
3. **Communication is the cost.** Optimize bytes moved per query, not just FLOPs.
4. **Amortize.** Verified, repeated deliberation is compiled into cheap skills.
5. **Learn fast and locally; consolidate slowly.** Write now, integrate during sleep.
6. **Hard budget, central regulation.** The Heart enforces compute and memory caps.
7. **Redundant codes with cleanup.** Partial cues still retrieve; noise is snapped back to stored patterns.
8. **Small serial workspace, large parallel background.** Coherence in the Workspace; preparation in the Subconscious.
9. **Factor; don't memorize.** Concepts are bundles of reusable properties, parts, functions and relations.
10. **Compose; don't pre-specialize.** The expert for a task is assembled during the task.
11. **Predictions are hypotheses until promoted.** Taint is removed only by evidence events.
12. **Every learned part has an identity and a contract.** Nothing is an anonymous blob.

## 5. Layered system overview

| Layer | Subsystems | File |
|---|---|---|
| L-A Interface (frozen after bootstrap) | Codec, code space and algebra, Property Basis, role/relation registries, message types | `03` |
| L-B Perception & Parsing | Chunker, perception encoder, L0 predictor, recognition-by-recall, structural parsers, triage | `03` |
| L-C Memory | Record store, Cortical Library, Hippocampal Index, indices, Association Field, Knowledge Sketch | `04` |
| L-D Intelligence Core | Workspace, evidence/truth maintenance, Primitive Basis, Composer, Selector, Error Monitor, Simulator, Concept Formation Engine | `05` |
| L-E Prediction & Epistemics | Predictive Hierarchy, prediction records, taint, envelopes, Support, Answer Record | `06` |
| L-F Control | Heart, Thalamic Gate, Modulators, Subconscious | `07` |
| L-G Expression & Action | Mouth (planner, articulators, renderer), Body (tools) | `07` |
| L-H Learning & Lifecycle | Bootstrap, online learning, sleep, compilation, component system, manifest, surgery | `08`, `09`, `10` |

## 6. Complete operation from input to output

This is the canonical path of one query. `11` gives the exact per-beat ordering and worked traces.

1. **Raw input arrives** as user text, a file, code or a tool result. It is wrapped in a **Source Envelope** (provenance, trust class) and, if it is a document for learning, converted to **SEF records** (`02`).
2. **Codec** normalizes bytes (`03` §6).
3. **Perception** splits the input into chunks, encodes them, and predicts each next chunk (L0). **Recognition-by-recall** looks each chunk up in the Library: known content becomes pointers; novel content continues (`03` §7).
4. **Structural parsing** turns novel content into a **SceneGraph** (referents, properties, relations, claims with modality) and turns the request into **GOAL claims** with typed answer variables (`03` §8–9).
5. **Triage** classifies the parsed units (fact, example, procedure, causal, contradiction, noise, …). Novel facts are written to the **Hippocampal Index**, with immediate retrieval possible (`02` §9, `04` §8).
6. The **Subconscious** primes related records by spreading activation, prefetches their pages, runs stage-1 retrieval of candidate concepts and schemas, and proposes habit (skill) answers. Everything it produces is tagged SUGGESTED (`07` §4).
7. The **Thalamic Gate** admits the most salient items into the small **Workspace** (`07` §2).
8. **The beat loop.** The **Heart** runs bulk-synchronous beats (`07` §1). In each beat:
   - processes bid for resources; the Heart funds the most valuable within caps;
   - the **Composer** builds or extends a **temporary circuit graph** from primitives, procedure schemas, analogies and skills (`05` §5);
   - the **Selector** picks which ready circuit nodes, verifications or retrievals run (`05` §7);
   - nodes execute (primitives, skills, retrievals, Body tools). Their outputs are **claims** or **prediction records**;
   - the **Error Monitor** verifies claims, detects contradictions and prediction errors, and emits evidence events (`05` §8);
   - **epistemic update:** evidence is applied, **Support** recomputed, beliefs propagated, taint propagated, failed claims retracted, nogoods recorded (`05` §2–3, `06`);
   - the **Concept Formation Engine** runs when the residual "novel" hypothesis dominates (`05` §9).
9. **Halting.** The goal is decided when its best-supported answer is committed, or when the best remaining action's value of computation drops below 0, or when the budget runs out. The goal's **epistemic state** (KNOWN class, PREDICTED class, UNKNOWN-*, CONTESTED, UNRESOLVED) is fixed (`06` §5–6).
10. **Utterance plan.** The committed claims, the labeled predictions and any abstention content are ordered into an **UtterancePlan**, and the **taint check** runs: tainted content must carry a prediction label (`07` §5).
11. **The Mouth** renders the plan:
    - formal targets (code, JSON, math) through deterministic articulators (unparsers);
    - natural language through the small neural renderer with attribution.
    This is the only place token prediction happens (L6).
12. **Answer Record.** The justification graph, evidence, executed circuit, rejected hypotheses and taint map are stored. "Why?" is answered from this record (`06` §7).
13. **Post-processing.** The episode (with its circuit trace and outcome) is written to the Hippocampal Index; local learning updates are applied (Hebbian links, verification-as-teacher, envelope and calibration updates); the consolidation queue is updated (`08` §5).
14. **Sleep**, idle or scheduled: replay, integration, abstraction, circuit compilation into skills, truth-maintenance repair, pruning and merging, re-layout. **Emergence events are written to the Manifest** (`08` §6, `09` §4).

## 7. Where intelligence lives

Intelligence is **not** the Library. It is:
- **(a)** the Primitive Basis;
- **(b)** the Composer and Selector, which assemble and steer temporary circuits;
- **(c)** the epistemic workspace dynamics: hypotheses, verification, Support, retraction;
- **(d)** the learning mechanisms: concept formation, compilation, consolidation.

The Library is the material this machinery reasons with. A model with an empty library and a complete Core is like a capable child: little knowledge, intact machinery.

## 8. Total capacity vs. active capacity; small vs. large models

**Total capacity** = Interface layer + Core + Library + Skills + Indices + Mouth + Episodic memory.

**Active capacity per query** = Interface layer (encoding) + Core (resident) + recruited Library pages + invoked skills + Mouth slice.

- The Heart enforces the active-capacity ceiling from the Build Configuration.
- The Library and skills scale by orders of magnitude; the Core scales slowly.
- **Machinery floor:** every scale includes the complete primitive set, Composer, Selector, Error Monitor, epistemic system, Concept Formation Engine, Heart and Mouth.
- Scaling down shrinks the Library, Skills, `K`, `R`, `H`, search budgets and Mouth fluency. It never removes a mechanism.

## 9. Diagram A — complete intelligence architecture

```
                 ┌──────────── MODULATORS (DA · ACh · NE · 5-HT) ─────────────┐
                 ▼                                                            │
 ┌──────────────────────── HEART: bid → fund → batched beat · caps · tiers ─────────────────────┐
 └──────┬───────────────┬───────────────┬─────────────────┬────────────────┬──────────────┬─────┘
        │ budget        │ budget        │ budget          │ budget         │ budget       │ budget
INPUT ──▼──────────┐    │               │                 │                │              │
(SEF / raw)        │    │               │                 │                │              │
BODY ─►│ CODEC → PERCEPTION & PARSING: L0 predict → recognize-by-recall ─(known)→ pointers/stats
(tools)│                     │ novel                                                      
       │                     ▼                                                            
       │        SCENE GRAPH (referents · properties · relations · claims+modality) + GOALS
       │                     │                                                            
       │      ┌──────────────┼───────────────────────────────┐                             
       │      ▼              ▼                               ▼                             
       │ HIPPOCAMPAL    TWO-STAGE MATCHING: stage-1 ◄── CORTICAL LIBRARY (dormant, tiered regions)
       │ INDEX (NEW)    sparse retrieval → stage-2 ALIGN  concept schemas · engrams · procedures · skills
       │      │         → hypothesis sets + residual    ▲  ASSOCIATION FIELD (typed; causes ≠ similar)
       │      │                │                        │  KNOWLEDGE SKETCH (UNKNOWN-ABSENT)
       │      │                ▼                        │
       │      │   SUBCONSCIOUS (priming · prefetch · habits · background matching) → SUGGESTED
       │      │                │                                                            
       │      │                ▼                                                            
       │      │        THALAMIC GATE (salience · suppression · routing)                     
       │      │                │                                                            
       │      │   ┌────────────▼─────────────── INTELLIGENCE CORE ────────────────────────┐
       │      │   │ WORKSPACE (K slots): goals · referents · hypothesis channels · claims   │
       │      │   │ COMPOSER → temporary CIRCUIT GRAPH of PRIMITIVES (BIND·ALIGN·ABSTRACT·  │
       │      │   │   TRANSFORM·PREDICT-STEP·REGRESS·TEST·SEARCH·ITERATE…) + skills/schemas  │
       │      │   │ SELECTOR (next node / stop) · SIMULATOR (rollouts, counterexamples)     │
       │      │   │ PREDICTIVE HIERARCHY L1–L5 → typed PREDICTION records (tainted)          │
       │      │   │ ERROR MONITOR: verifier · prediction errors · conflicts · TMS retraction│
       │      │   │ SUPPORT RANKING: best-supported, grounded path required to commit       │
       │      │   │ CONCEPT FORMATION ENGINE (few-shot schemas)                             │
       │      │   └──────────────┬────────────────────────────────────┬────────────────────┘
       │      │                  │ committed / tainted / UNKNOWN      │ traces, outcomes
       │      │                  ▼                                    ▼
       │      │   MOUTH: taint check → utterance plan → articulators → small renderer (L6) → OUTPUT
       │      │                                                       │
       │      └──────────► SLEEP ENGINE: replay · integrate · abstract · compile circuits→skills ·
       │                   TMS repair · prune/merge · re-layout · dream · write emergence → MANIFEST
       └── local learning: writes · Hebbian · verification-as-teacher · envelope/calibration updates
```
