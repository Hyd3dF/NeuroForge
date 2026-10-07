# 07 — Control (Heart, Gate, Modulators, Subconscious), Expression (Mouth), Action (Body)

## 1. Heart: the resource circulation controller (component `control.heart_policy`)

### 1.1 The beat
A beat is one bulk-synchronous step. All funded work in a beat is batched into as few kernel launches as possible. The per-beat order of execution is fixed (`11` §2).

### 1.2 Resources
| Resource | Unit | Measured by |
|---|---|---|
| `flops` | Estimated FLOPs | Per-op cost model (calibrated by profiling) |
| `bytes` | Bytes moved from T1/T2 into compute | Page loads |
| `slots` | Workspace slots | Count |
| `verifier_calls` | Calls | Count |
| `tool_calls` | Body calls | Count (with per-tool cost) |
| `tokens` | Mouth tokens | Count |
| `wall_ms` | Milliseconds | Timer (soft constraint) |

### 1.3 Per-query budget
`Budget_q = base_budget × (1 + stakes_gain · stakes) × NE_gain(NE)`, capped by the Build Configuration's `max_budget_per_query`. The maximum number of beats is `T_max`.

### 1.4 Bidding and allocation
1. **Diastole (bid):** each awake process submits bids `(resources, v̂, conf)`. Processes: Gate admissions, Selector actions (with VOC), Composer expansions, Error Monitor verifications, Simulator runs, memory retrievals, Concept Formation Engine, Subconscious tasks, Mouth (when ready).
2. **Adjustment:** `v_adj = v̂ · c_p`, where `c_p` is the process credit (§1.5).
3. **Price:** `cost_scalar = Σ_r price_r · amount_r`. Prices come from the hardware profile; the 5-HT signal scales the overall cost weight `λ_cost`.
4. **Allocation:** greedy by `v_adj / cost_scalar`, subject to per-beat caps (`max_flops_per_beat`, `max_bytes_per_beat`, `max_active_records`, `max_active_fraction`), the remaining query budget, and **floors** (minimum shares) for the Subconscious and the Error Monitor (F0: 10% and 15%).
5. **Systole:** dispatch the funded work as batches.

### 1.5 Credit (bid honesty)
- After execution, measure the realized value: `ΔU_goal` caused, verification survival of produced claims, or prefetch hits.
- Update `c_p ← (1−η_c)·c_p + η_c·clip(realized / max(v̂, ε), 0, 2)`, bounded to `[c_min, c_max]` (F0 [0.1, 2.0]).

### 1.6 Tier management
- Page heat = Σ recent activation × value. Pages with heat above `θ_promote` move T2→T1 (and T1→T0 when available); cold pages are demoted.
- Prefetch requests from the Subconscious are funded as `bytes` bids with predicted value = P(needed) × value-if-needed.

### 1.7 Halting authority
The Heart ends the beat loop when the Selector issues STOP, the budget is exhausted (the goal becomes UNRESOLVED), or `T_max` is reached.

### 1.8 Learned parts
Only the bid-value estimators of the Selector and the Subconscious, and the price calibration, are learned. The allocation itself is a deterministic algorithm, which keeps it auditable.

## 2. Thalamic Gate (component `control.gate`)
- **Candidates:** items in the preconscious buffer (Subconscious outputs, perception novelties, retrieval results).
- **Salience:** `sal(x) = σ(w_1·precision(x)·|ε(x)| + w_2·relevance(x, goals) + w_3·novelty(x) + w_4·trust(x)) · gain_ACh`. Relevance = max cosine to goal embeddings; precision = inverse running variance of that error type. The weights `w` are learned (logistic regression trained on whether admitted items ended up on goal-support paths).
- **Admission:** top-k by salience into free workspace slots (k funded by the Heart). Competing items with similarity above `θ_dup` are deduplicated (lateral inhibition).
- **Routing:** for each admitted query need, the Gate selects the index (content / signature / triple / episodic) with a small learned classifier, falling back to all.

## 3. Modulators (component `control.modulators`)

| Signal | Computation (per beat) | Consumers |
|---|---|---|
| **DA** | Selector TD error `δ` (`05` §7) | Selector/Composer learning rate sign; Hebbian `m`; habit compilation priority |
| **ACh** | `EMA(normalized L0/L2 prediction-error magnitude)` (model unreliability) | Encoding mode (raises hippocampal write rate and bottom-up weight); Gate gain |
| **NE** | `max(z-score of surprise/contradiction events in last n beats)`, clipped | Budget multiplier `NE_gain`; number of open channels; Composer exploration temperature |
| **5-HT** | `f(stakes, remaining budget, user patience setting)` | `λ_cost` (cost weight), `θ_answer` scaling, maximum beats |

All formulas and gains come from the Build Configuration. The modulators have no other hidden effects.

## 4. Subconscious (component group `control.subconscious.*`)
Background processes funded from the Subconscious floor. They **cannot commit** anything; all outputs are SUGGESTED.

| Process | Mechanism |
|---|---|
| Priming | Spreading activation (`04` §6) from workspace content codes |
| Prefetch | Pages of the top primed records → `bytes` bids |
| Background stage-1 matching | `RETRIEVE` on new scene fragments → candidate concepts/schemas into the buffer |
| Habit proposals | Skills whose envelopes contain the current subgoal's input → run → SUGGESTED answers with track record |
| Anomaly monitors | Threshold on L0/L2 errors and contradiction-scan hits → salience boost |
| Background association | ALIGN between pairs of active items of different domains, at low budget → analogy candidates |
| Incubation | Continue low-priority SEARCH for unsolved goals across turns or idle time |
| Micro-consolidation | Idle-time small sleep tasks (`08` §5) |

## 5. Mouth (expression; L6 only)

### 5.1 UtterancePlan
| Field | Meaning |
|---|---|
| `items` | Ordered `PlanItem`s |
| `PlanItem` | `role ∈ {answer, support, hedge, alternative, abstention, missing_info, explanation, code_block, data_block, plan_step}`, `content_ref` (claim / ImplRepr / plan graph / value), `epistemic_tag` (§5.4), `citations` (source ids), `format_target` |
| `format` | Target format (prose, markdown, json, python, dsl, latex, …) |
| `style` | Register, length limit, language |

Built by the **Utterance Planner** (deterministic): ordering rules by role, plus grouping by referent.

### 5.2 Rendering pipeline
1. **Taint check (MUST):** every PlanItem whose `content_ref` is tainted must carry a PREDICTION-class tag. A violation is a hard error, logged; the item is re-tagged.
2. **Formal targets → deterministic articulators:** complete `ImplRepr` → unparser (Python `ast.unparse` or the DSL printer; the SymPy printer for math); structured data → serializer (JSON with schema). No neural decoding, so the output exactly matches the internal solution.
3. **Natural language → neural renderer** (component `mouth.renderer_base`): a small encoder–decoder.
   - The encoder takes PlanItem embeddings (claim content embedding + tag embedding + role embedding + citation slots).
   - The decoder is a causal Transformer decoder (`n_layers_mouth`, `d_mouth`, `V_out`, all from config) cross-attending to plan items.
   - A **pointer/attribution head** points to the source plan item for each content span.
4. **Articulators** (components `mouth.articulator.<target>`): low-rank adapters on the renderer for style or language, plus a grammar constraint (context-free grammar / regular-expression masks) for semi-formal targets (Markdown tables, JSON-in-prose).
5. **Constrained decoding:** grammar masks; copy/pointer for literal values (numbers, names) from plan items, so values cannot be mistyped.
6. **Attribution check:** spans whose attribution distribution has entropy above `θ_attr` or points to no item are regenerated (once). If that fails, the system falls back to the template renderer (§5.5).

### 5.3 What the Mouth cannot do
It cannot add content that is not in the plan. Literal values must be copied. It has no access to the Library, Workspace or Core states beyond the plan.

### 5.4 Epistemic tag → rendering policy

| Tag (from `06` §5–6) | Rendering |
|---|---|
| OBSERVED / REMEMBERED / DERIVED / TESTED | Plain assertion; citations optional by format |
| PREDICTED / INHERITED / ANALOGICAL | "Likely / probably …" plus the main supporting evidence |
| EXTRAPOLATED | "I'm extrapolating beyond what I've verified: …" |
| SUGGESTED | Not rendered as an answer; may appear as "one possibility" only if the user asks for brainstorming |
| CONJECTURED | "If we assume …, then …" |
| CONTESTED | "Sources disagree: A says …, B says …" |
| INSUFFICIENT | "I'm not sure. The best-supported option is …, but the evidence is weak because …" |
| UNRESOLVED | "I couldn't determine this within the budget; partial results: …" |
| UNKNOWN-ABSENT / UNKNOWN-NO-BASIS | "I don't know." + what is known about the surrounding entities + **what specific information is missing** (from the open subgoals with the highest `u`) |
| UNKNOWN-DECLARED | "This is not known (per <source>, as of <date>)." |

### 5.5 Template renderer
A deterministic template per (role, tag) is always available. It is used as the fallback and in F0 before the neural renderer is trained.

### 5.6 Isolation
The Mouth's token probabilities are logged for training but never fed back as evidence.

## 6. Body (action and grounding)

### 6.1 Tool interface (components `body.tool_adapter.<tool>`)
| Field | Meaning |
|---|---|
| `name`, `version` | Identity |
| `input_schema`, `output_schema` | Typed messages (`03` §5) |
| `sandbox` | Isolation level (process, container, none for pure functions) |
| `limits` | Time, memory, network policy |
| `cost` | Heart price |
| `determinism` | Deterministic, or seeded with variance |
| `trust_class` | `tool_execution` (`02` §10.1) |

### 6.2 F0 tools
- DSL interpreter.
- Restricted Python sandbox (subprocess; no network; resource limits).
- Test runner (unit + property tests).
- SymPy evaluator.
- Unit converter.

### 6.3 Semantics
- Tool results are **OBSERVED** claims (`source_type: tool_execution`) and serve as promotion events (`06` §2.3).
- A tool error is itself an observation (the error signature), which feeds debugging goals and bug-fix episodes.
