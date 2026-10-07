# 05 — Intelligence Core

The Core is resident for every query and is subject to the **machinery floor** (`01` §8). It consists of:
- the **Workspace** with its evidence and truth-maintenance dynamics (§1–3);
- the **Primitive Basis** (§4);
- the **Composer** (§5);
- **two-stage matching** (§6);
- the **Selector** (§7);
- the **Error Monitor** (§8);
- the **Concept Formation Engine** (§9);
- the **Simulator** (§10);
- skill invocation (§11).

## 1. Workspace

### 1.1 Contents (fixed-capacity tensors; sizes from the Build Configuration)

| Structure | Capacity | Fields |
|---|---|---|
| Claim slots | `K` | `Claim` message fields (`03` §5) + `status ∈ {OPEN, CHECKED, COMMITTED, RETRACTED, DORMANT}` + `relevance ρ` + `spent κ` + `structure_bundle` |
| Referent slots | `R` | `Referent` messages + bindings to records |
| Goal stack | `G_max` (F0 8) | `Goal` messages with parent/child links (subgoals) |
| Hypothesis sets | Up to `HS_max` (F0 16) | `PredictionRecord`s with hypotheses + residual |
| Channels | `H` | Channel id, assumption claims, alive flag, weight |
| Active circuit | 1 per goal | `CircuitGraph` (§5.1) with node states |
| Nogood set | `NG_max` (F0 1024) | Hashes of (premise-content set, producer) |
| Answer trace | Unbounded (log) | Events for the Answer Record (`06` §7) |

### 1.2 Admission and eviction
- Items enter through the Thalamic Gate (`07` §2) or as outputs of executed nodes.
- When full, the slot with the lowest `ρ × activity` that is not on any goal's support path is evicted to DORMANT. It is written to the episodic trace, not deleted from the provenance log.

## 2. Evidence algebra

### 2.1 Belief, disbelief, uncertainty
For each claim or hypothesis: `b = e⁺/(e⁺+e⁻+W)`, `d = e⁻/(e⁺+e⁻+W)`, `u = W/(e⁺+e⁻+W)`.

### 2.2 Evidence events
`EvidenceEvent`s add `Δe⁺` / `Δe⁻`:

| Event kind | Magnitude |
|---|---|
| Observation match | `κ · t_source` |
| Memory support | `κ · hit_strength · b(record) · plasticity-independent reliability` |
| Verifier entail / contradict | `κ_v · p_entail` / `κ_v · p_contra` |
| Test pass / fail | `κ_t` (F0 2.0) |
| Corroboration | `κ · t_source` per new root source |
| Prediction error | `κ · min(1, ε/ε_ref)` against the prediction |

**Independence:** events sharing a `source_root_id` are combined by **max**, not sum.

### 2.3 Derived claims
A claim `c` produced by node `n` from premises `p_1..p_k`:
- `b_inh(c) = r_n · T(b(p_1), …, b(p_k))`, where `T` is the t-norm (default `min`; `product` configurable) and `r_n` is the reliability of the producer:
  - exact algorithmic primitives: 1.0;
  - learned components: track-record accuracy within the envelope;
  - outside the envelope: `r_extrap` (F0 0.3), and the claim is tagged EXTRAPOLATED.
- **Initialization:** `e⁺ = W · b_inh / (1 − b_inh)` (capped at `e_max`), `e⁻ = 0`.
- **Recomputation:** when a premise's belief changes, the claim's inherited component is recomputed and its own direct evidence is retained.
- **Taint:** `taint(c) = OR(taint(p_i)) OR producer_is_predictive(n)` (`06` §2).

### 2.4 Channels
`channel_mask(c) = AND` of the premises' masks.
- A claim whose mask is all ones holds in every channel (shared work is done once).
- CONJECTURED claims open a channel. When a contradiction occurs inside a channel, the channel is killed (`alive = false`) and its exclusive claims become DORMANT.

## 3. Truth maintenance (retraction and nogoods)
1. **Trigger:** `d(c) ≥ θ_retract`, or a contradiction is detected between `c` and `c'` in the same channel.
2. **Contradiction culprit:** compute the CONJECTURED ancestors of both claims in the justification DAG. If they share assumptions, record a nogood over the minimal shared assumption set and kill the channels containing it. If there are no assumptions, retract the claim with the lower Support.
3. **Retraction:** status RETRACTED. Walk dependents in topological order (`K` is small, so a full recompute is acceptable) and recompute `b_inh`. Dependents whose belief drops below `θ_dormant` become DORMANT.
4. **Nogood recording:** the hash of (sorted premise content codes, producer id) goes into the nogood set. The Selector and Composer mask matching actions.
5. **Learning signal:** the producer of a retracted claim receives a negative verification-as-teacher signal (`08` §4.3).

## 4. Primitive Basis

Signatures use the message types of `03` §5. *Impl* is **A** (exact algorithm, no weights), **L** (learned component) or **H** (hybrid: algorithm with learned scoring). Each learned or hybrid primitive is a component `core.primitive.<name>`.

| Primitive | Signature | Impl | Specification |
|---|---|---|---|
| `BIND`, `UNBIND`, `BUNDLE`, `PERMUTE`, `SPARSIFY` | Code × Code → Code; [Code] → Bundle | A | `03` §2.2 |
| `SIM` | Code/Dense × Code/Dense → score | A | `ov/B` and cosine |
| `RETRIEVE` | query (Code, Dense, filters) → [record] | A (+ index) | `04` §5.1–5.3 |
| `CLEANUP` | noisy Dense × candidates → Dense | A | `04` §5.4 |
| `COMPARE` | SceneGraph/Schema × SceneGraph/Schema → DiffRecord | H | For each property in the union: `match`, `mismatch(z)` with `z = |x−μ_C| / σ_C` (scalar) or `−log p` (enum/bool), or `missing_a` / `missing_b`. Score `= Σ w_def·[match] − Σ w_def·φ(z) − λ_miss·Σ w_def·[missing]`, where `φ` is a learned monotone calibration (small MLP, 1 input). Output: DiffRecord (lists + score) |
| `ALIGN` | SceneGraph × SceneGraph → (Correspondence, score, CandidateInferences) | H | §6.2 |
| `ABSTRACT` | [SceneGraph] (aligned) → Schema | A | §6.3 (anti-unification) |
| `SPECIALIZE` | Schema × bindings → SceneGraph | A | Substitute variables; unfilled properties get the schema's typical value as **INHERITED** (tainted) claims |
| `TRANSFORM` | Operator × args → Value/SceneGraph | A or L | Symbolic operators (DSL primitives, arithmetic, SymPy) are A; neural operators are skills (§11) |
| `ORDER`, `COUNT`, `AGGREGATE` | [Value/Claim] → Value/[…] | A | Sort by comparator; count; sum/min/max/mean over typed values |
| `PREDICT_STEP` | State(SceneGraph) × Action/Event → [PredictionRecord L3] | H | Match causal schemas whose cause pattern and conditions unify with the state (structured retrieval, `04` §5.2), then emit effect claims as PREDICTED. If none match, fall back to the learned dynamics skill for the domain (if in envelope), otherwise return UNKNOWN-NO-BASIS |
| `REGRESS` | Goal pattern → [(action/operator, preconditions as subgoals)] | H | Retrieve causal schemas and procedures whose effects/postconditions unify with the goal; rank by learned prior × past success |
| `DECOMPOSE` | Goal → [Goal] | H | Instantiate a matching procedure schema's step structure, or REGRESS |
| `EXPLAIN` | Evidence set × Hypothesis set → Support per hypothesis | H | `06` §4. A learned log-likelihood-ratio head per evidence kind (small MLP over [evidence features, hypothesis features]) is trained for calibration |
| `TEST` | Claim/constraint → EvidenceEvent (+ witness) | A/L | Dispatch: exact constraint check (A); Body execution (A); verifier (L, §8); memory consistency (A + retrieval) |
| `SEARCH` | Generator × scorer × budget → [candidates] | A | Best-first / beam with nogood masking and budget accounting |
| `ITERATE`, `BRANCH` | Control | A | Circuit-graph control nodes (§5.1) |
| `UNIFY` | Pattern × Claim → bindings or ⊥ | A | Typed unification with qualifier compatibility (time overlap, version match) |
| `QUERY_JOIN` | [Pattern] → bindings | A | Conjunctive query over `RETRIEVE_PATTERN` results (multi-hop facts) |

- **Promotion to derived primitives:** sleep may promote a skill to a derived primitive when it is reused across ≥ `n_domains_promote` task families. It keeps its component ID and gains `primitive` status in the Manifest (`08` §6.4).
- **The set is versioned** with the interface layer: adding a primitive is a MINOR change; changing a signature is a MAJOR change.

## 5. Composer (temporary computation circuits)

### 5.1 Circuit graph

| Element | Fields |
|---|---|
| `CircuitNode` | `node_id`, `op` (primitive name / skill logical ID / procedure id / `RETRIEVE` / `TEST` / `BODY:<tool>`), `in_ports: [(name, type)]`, `out_ports`, `bindings` (to workspace claims/referents or literals), `budget`, `state ∈ {PENDING, READY, RUNNING, DONE, FAILED, SKIPPED}`, `produced: [claim_id]`, `reliability` |
| `Edge` | `(src_node, out_port) → (dst_node, in_port)`; types must match (`03` §5) |
| Control | `ITERATE(body_subgraph, state_port, termination_test, max_iter)`, `BRANCH(test_port, then_subgraph, else_subgraph)` |
| `CircuitGraph` | Nodes, edges, goal id, origin (`schema` / `analogy` / `synthesized`), cost-so-far, nogoods-hit |

**Execution semantics:**
- Dataflow: a node is READY when all of its input ports are bound to claims with status ≥ CHECKED, or literals.
- Every output is a claim (or prediction record) with `justification.producer_id = node_id` and premises = its inputs.

### 5.2 Construction algorithm (per goal)
1. **Goal typing:** derive the output type from `Goal.answer_vars` or the output spec, and the available input types from the workspace.
2. **Schema retrieval:** `RETRIEVE(kind=PROCEDURE)` by goal content code, then `ALIGN(goal pattern, schema.goal pattern)`. If the score ≥ `θ_schema`, instantiate the template with bindings (holes become sub-goals).
3. **Analogical transfer:** `RETRIEVE_SIG` over successful EPISODEs and procedures from any domain, then ALIGN. If the score ≥ `θ_analogy`, transfer the circuit with roles rebound. The transferred nodes carry the analogy tag; their outputs are ANALOGICAL until verified.
4. **Type-directed synthesis:** best-first search over partial graphs.
   - Expansions add a node whose input types are available and whose output brings the frontier closer to the goal type.
   - Priority `= log π_θ(op | state) − λ_cost·est_cost − λ_depth·depth`.
   - Limits: depth `D_max` (F0 8), expansions `E_max` (F0 512).
   - Nogood-masked; each expansion is a Heart-funded action.
5. **Execution and local repair:** execute READY nodes as the Selector chooses. On a FAILED node (a test fails or the output is retracted), re-synthesize only the subgraph rooted at the failure, with the failure recorded as a nogood.
6. **Completion:** the goal is satisfied when a COMMITTED claim binds all answer variables (`06` §6).

### 5.3 Composer policy network `π_θ` (component `core.composer`)
- **Inputs:** goal embedding, a workspace summary (attention pooling over claim embeddings), a partial-graph embedding (a 2-layer message-passing network over the graph's nodes, using operation embeddings), and the candidate operation embedding.
- **Output:** a logit per candidate.
- **Training:** imitation on reference solutions (SEF `task.reference_solution`), then RL on verified success minus cost (`08` §2, S4).

### 5.4 After the task
- The graph is dissolved.
- The trace (graph, bindings, node outcomes, cost, verified result) is written as an EPISODE.
- Failed subgraphs leave nogood traces.
- Compilation candidates are queued for sleep (`08` §6).

## 6. Two-stage matching (recognition and analogy)

### 6.1 Stage 1
Content retrieval (`RETRIEVE`) for recognition; signature retrieval (`RETRIEVE_SIG`) for analogy. Return `k_ret` candidates, cut to `k_align` by stage-1 score.

### 6.2 Stage 2: `ALIGN(G_s, G_t)`
1. **Node affinity:** `A0[i, j] = σ(f_aff(dense_i, dense_j, type_compat_ij))`, with `f_aff` a learned MLP. This is the learned part of the hybrid.
2. **Structural refinement:** for `t_align` iterations (F0 5), `A[i, j] ← A0[i, j] + λ_s · Σ_{(i,i')∈E_s, (j,j')∈E_t, same rel} A[i', j']`, then Sinkhorn normalization. Higher-order relations get more weight (systematicity).
3. **Hard correspondence:** Hungarian assignment on `A`, keeping pairs with `A ≥ θ_pair`.
4. **Score:** `= Σ_matched A + λ_rel · #consistent_relations − λ_unmatched · #unmatched_defining_slots`.
5. **Candidate inferences:** source relations and properties whose arguments are all mapped but which are absent in the target. They are emitted as **ANALOGICAL** (alignment with a stored case) or **INHERITED** (alignment with a schema) predictions.
6. **Output:** the correspondence, the score, the DiffRecord (via COMPARE on matched pairs), and the candidate inferences.

### 6.3 `ABSTRACT` (anti-unification over aligned graphs)
- For aligned node sets: equal values become constants; differing values become variables with the observed value set or range.
- Relations present in all graphs are kept; relations present in some become optional (with a frequency).
- The output schema's variability profiles are initialized from the observed spread.

## 7. Selector (component `core.selector`)
- **Action space per beat:** execute a READY node; verify a claim; retrieve for a subgoal; expand the Composer; run the Simulator; open or close a channel; invoke the Concept Formation Engine; **STOP**.
- **Features per action:** action type embedding, target claim/goal embedding, its `u`, its centrality (number of goal-support paths through it), estimated cost, nogood proximity, modulator state.
- **Critic:** `V(s)` predicts final goal success (probability of a committed correct answer). **Actor:** a score per action.
- **Value of computation:** `VOC(a) = Ê[ΔU_goal | a] − λ_cost · cost(a)`.
  - `U_goal` = entropy over the goal's hypotheses + mean `u` of the leading hypothesis.
  - `Ê[ΔU]` comes from a learned head regressed on realized ΔU; it bids to the Heart (`07` §1).
- **Heuristic VOC (used before the learned head is trained, milestone M6):** `VOC_h(a) = u(target) · (1 + centrality(target)) · (0.5 + stakes) · type_prior(a) − λ_cost · cost(a)`. `type_prior` is a configurable table (verify on the goal-support path > execute READY node > retrieve for open subgoal > Composer expansion > Simulator > channel ops). The learned head replaces it after S4; the heuristic remains the fallback when the learned head is outside its envelope.
- **Stop rule:** STOP when the goal is decided (`06` §6), or `max_a VOC(a) < 0`, or the budget is exhausted.
- **Learning:** TD error `δ = r + γV(s') − V(s)` (r = verified success − cost at episode end; shaped by ΔU per beat) is the **DA** signal (`07` §3).

## 8. Error Monitor (component `core.error_monitor.*`)

### 8.1 Verifier (component `core.error_monitor.verifier`)
- **Information bottleneck:** input = **canonicalized** claim and premises only: content codes, relation ids, value encodings, qualifier codes, re-embedded through the verifier's **own** embedding tables. It has no access to workspace dense states, generator confidences or channel weights.
- **Architecture:** a set encoder over premises plus a cross-attention to the claim (2–4 layers, width configurable) → `(p_entail, p_contradict, p_neutral)`.
- **Training:** synthetic entailment/contradiction from the worlds, corrupted facts, **on-policy negatives** (the system's own retracted claims), with calibration via temperature scaling on a held-out set.

### 8.2 Other checks
| Check | Mechanism |
|---|---|
| Memory consistency | `RETRIEVE_PATTERN` on the claim's (subject, relation) → COMPARE → contradiction or support events |
| Prediction errors | Level-wise errors from `06` §1 |
| Habit/deliberation conflict | When a skill's output disagrees with a deliberately derived result for the same subgoal, flag the skill (`08` §4.5) |
| Redundancy disagreement | Disagreement among independent derivations (disjoint justification sets) → `e⁻` on both, plus a Selector bid to resolve |
| Contradiction scan | Pairwise check among claims sharing (subject, functional relation) in the same channel |

### 8.3 Verification budget
Claims are verified in priority order `stakes × centrality × u`. Claims below `θ_verify_priority` are not verified if the budget is short. Their beliefs then remain at inherited levels, so they cannot reach `θ_commit` unless their inherited support is already sufficient and untainted.

## 9. Concept Formation Engine (component `core.concept_formation`)

**Trigger:** labeled examples of an unknown label; or recognition where the residual hypothesis has the highest Support; or recurring unexplained instances found during sleep.

**Algorithm**, for examples `X = {x_1..x_n}` (positive), optional negatives `X⁻`:
1. **Parse** each example into a SceneGraph (`03` §8).
2. **Align and abstract:** pairwise ALIGN against a pivot example, then ABSTRACT → candidate schema `S_0`, with constants, variables and observed ranges.
3. **Parent selection:** retrieve concepts by `S_0` content code; ALIGN `S_0` against each; choose the parent `P` that maximizes `score − λ_exc · exceptions`. If none exceeds `θ_parent`, attach to the nearest top-level type.
4. **Hypothesis space by generalization level.** For each property `π` not constant across examples, or observed only once:
   - narrow: the observed range;
   - medium: the observed range widened by the parent overhypothesis's expected variability `E[v_child(π)]`;
   - broad: the parent's full range.
   A hypothesis `h` is a choice of level per property, pruned to a beam.
5. **Size-principle scoring:** `log P(h | X) ∝ log prior(h) − n · log |h|`.
   - `|h| = Π_π width_h(π) / width_global(π)` (normalized volume).
   - `prior(h)` favours widening properties the overhypothesis marks as variable and keeping defining-property ranges narrow.
   - Negative examples in `h` give `−∞`.
6. **Choose `h*`**; set the variability profiles from `h*` and the parent's overhypothesis; set `w_def`.
7. **Inheritance:** the parent's properties not observed in the examples are copied as **INHERITED** defaults (tainted predictions about members).
8. **Contrast:** COMPARE `h*` with the parent's other children; store the discriminative properties.
9. **Store** as a CONCEPT record, lifecycle NEW, in the Hippocampal Index, with evidence count `n`; update the Sketch.
10. **New properties:** if the examples share a regularity that no property expresses (high residual in a property-prediction head), create a provisional property (`03` §3).

**Output:** a new or updated schema, plus a classification of each example.

## 10. Simulator (components `core.simulator.*`)
- **Rollout:** a chain of `PREDICT_STEP`s from a state through an action sequence (plans) or an execution trace (code). Each step's predictions are PREDICTED (L3/L5).
- **Counterexample generation:**
  - DSL/code: property-based input generation (boundary values, random typed values, shrinking);
  - math: random numeric substitution checks of symbolic identities;
  - claims: retrieval of potentially conflicting facts plus perturbation of qualifiers;
  - concepts: generated instances at the edges of the variability ranges.
- **Body bridge:** executable artifacts (code, math) are run through the Body (`07` §6). Results are OBSERVED evidence and **promote** (or refute) predictions.

## 11. Skill invocation (habits)
- A compiled skill (`08` §6) is invoked as a circuit node.
- **Envelope check first** (`06` §3): inside the envelope its outputs carry reliability `r = track-record accuracy`; outside, they are EXTRAPOLATED with `r_extrap`.
- Skills invoked by the **Subconscious** produce SUGGESTED outputs.
- Skill outputs that are factual claims are SUGGESTED until checked against engrams (fact-placement rule, `06` §8).
- **Skill execution:** shared skill core network (`d_s`, `n_layers_skill`, configurable) with the skill's low-rank modulation `ΔW = U Vᵀ` (rank `r_skill`) applied per layer. Input and output heads are typed by the skill's interface signature. Multiple skills are batched with grouped matrix multiplications.
- **Shared skill core** (component `core.skill_core`): an MLP-residual network over interface-layer message embeddings, pretrained in S3 on a broad mixture of transformation tasks (DSL operations, property mappings, dynamics). **Frozen after S3** and versioned with the interface MAJOR version, so that every compiled skill (a modulation of it) stays valid. Changing it is a MAJOR migration that requires re-distilling the skills (`10` §9.2).
