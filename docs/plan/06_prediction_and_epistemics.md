# 06 — Prediction and Epistemics

Prediction exists at every level of the system, and **prediction is never the same internal state as knowledge**.

## 1. The Predictive Hierarchy

| Level | Predicts | Producer (F0) | Error signal | Error drives |
|---|---|---|---|---|
| **L0 Perceptual** | Next chunk's encoding | Perception L0 predictor (`03` §7.3) | `ε0 = 1 − cos` | Novelty gating, Gate salience, ACh |
| **L1 Recognition** | What this is: hypothesis set over concepts + residual | Two-stage matching + EXPLAIN (`05` §6, §4) | Rejection of the leading hypothesis by new evidence | Reweighting; Concept Formation Engine trigger |
| **L2 Completion** | Missing parts and properties | Candidate inferences, SPECIALIZE defaults | Observed value outside the predicted range | Tests/queries; concept profile updates |
| **L3 Causal/dynamic** | Effects of actions and events; next state | PREDICT_STEP | Observed next state ≠ predicted | Causal schema evidence; envelope update |
| **L4 Plan/value** | Which action reaches the goal; cost | REGRESS, Composer policy, Selector critic | Plan step fails; realized cost ≠ predicted | Replanning; Selector TD learning; Heart credit |
| **L5 Execution** | Output of code/computation | Simulator; Body checks reality | Body result ≠ predicted | Skill recalibration; debugging goals |
| **L6 Expression** | Next output token | **Mouth only** | Rendering loss (training only) | Mouth training only |

- **World Prediction** = L0–L5 (Core, Library, Simulator).
- **Expression Prediction** = L6 (Mouth).
- The Mouth's token probabilities are **never** read as evidence about the world (`07` §5.6).

## 2. Prediction records and taint

### 2.1 PredictionRecord
Defined in `03` §5. Key properties:
- **Hypothesis sets.** Each hypothesis carries `evidence_for`, `evidence_against`, `assumptions`, `generator_id/version` and `in_envelope`.
- **Residual hypothesis.** Every hypothesis set contains an explicit `RESIDUAL` ("none of these / novel"). Its Support is `S_res = λ_res · unexplained_evidence_mass − λ_res0`, where unexplained mass is the evidence (observed properties, relations) not accounted for by the best hypothesis's alignment. So the open world is always represented.
- **Status:** `PREDICTED` while tainted; `PROMOTED` (→ knowledge claim) or `REJECTED` (→ retracted).

### 2.2 Taint rules (information-flow typing)
- **T1.** Claims produced by any predictive producer are tainted: candidate inferences, SPECIALIZE defaults (INHERITED), PREDICT_STEP, analogical transfer, skills outside envelope or invoked subconsciously, the residual, and the Composer's guesses.
- **T2.** `taint(c) = OR(taint(premises))` for derived claims. Deduction does **not** remove taint.
- **T3.** CONJECTURED claims are tainted and also carry channel restrictions.
- **T4.** Taint is removed **only** by a **promotion event** (§2.3) applied to the claim itself.
- **T5.** The Mouth MUST render tainted claims only with a prediction label (`07` §5.4). Final answers in the KNOWN class require an untainted claim.

### 2.3 Promotion events (taint removal)
1. **Observation match:** an OBSERVED claim (from input or Body) unifies with the predicted claim.
2. **Test pass:** an executed test or Body check with a definite verdict.
3. **Untainted deduction:** re-derivation of the same content from **only untainted** premises by exact (A-class) primitives.
4. **Independent corroboration:** an untainted memory record with `b ≥ θ_commit` (REMEMBERED) unifies with the claim.

Verifier entailment alone adds evidence but does **not** remove taint, because the verifier is learned. Configuration MAY allow it for low-stakes goals, but the default is disallowed.

## 3. Validity envelopes (per learned generator)
- **Data:** input embeddings (`Dense`) of the generator's runs whose outputs were later verified (passed or failed).
- **Model:** online k-means with `k_env` centroids. Per centroid: count `n_c`, successes `s_c`, radius `ρ_c` = 95th percentile of member distances.
- **In-envelope test** for input `x`: the nearest centroid `c` must satisfy `dist(x, c) ≤ ρ_c` and `n_c ≥ n_min` (F0 20) and the lower credible bound `Beta(s_c+1, n_c−s_c+1).ppf(0.05) ≥ θ_env` (F0 0.8).
- **Effects:**
  - inside: producer reliability `r_n` = posterior mean `(s_c+1)/(n_c+2)`;
  - outside: the output is tagged **EXTRAPOLATED**, with `r_n = r_extrap`;
  - for factual answers, EXTRAPOLATED alone counts as **no basis** (§5).
- **Updates:** after each verification of the generator's output (`08` §4.4). The envelope shrinks where failures accumulate.

## 4. Support: best-supported, not most probable

**Support(h) = Σ_{g ∈ groups} max_{e∈g} [ w(e) · llr(e, h) ] − λ_c · C(h) + λ_v · V(h) − λ_mdl · DL(h) + min(π_cap, λ_p · log p_gen(h))**

| Term | Meaning |
|---|---|
| groups | Evidence grouped by `source_root_id` (dependent evidence counts once) |
| `llr(e, h)` | Calibrated log-likelihood ratio from EXPLAIN heads (match/mismatch scores, observations, memory hits) |
| `w(e)` | Source trust × extraction confidence |
| `C(h)` | Number of unresolved contradictions involving `h` |
| `V(h)` | Number of verification/test passes on `h`'s claims |
| `DL(h)` | Description length of `h` (Occam) |
| `p_gen(h)` | Generator probability (the proposal), with **capped** influence `π_cap` (F0 1.0 nat) |

- **Mapping into evidence:** positive contributions → `Δe⁺`, negative → `Δe⁻` on the hypothesis's claim.
- **Normalized weights for display:** `softmax(Support)` over the hypothesis set including the residual. These are presentation weights only and are never used as belief.

## 5. Epistemic state lattice

| Class | State | Definition |
|---|---|---|
| Knowledge (untainted) | **OBSERVED** | Grounded in current input or a Body result |
| | **REMEMBERED** | Retrieved untainted record with `b ≥ θ_commit`, lifecycle ≥ CORROBORATED (or a high-trust single source, configurable) |
| | **DERIVED** | Exact deduction from untainted premises, `b ≥ θ_commit` |
| | **TESTED** | Passed a test or Body check |
| Prediction (tainted) | **PREDICTED** | Output of a predictive producer within envelope |
| | **INHERITED** | Default from a parent concept |
| | **ANALOGICAL** | Candidate inference from analogy |
| | **EXTRAPOLATED** | Producer outside its validity envelope |
| | **SUGGESTED** | Subconscious or third-party skill output |
| Hypothetical | **CONJECTURED** | Assumption inside a channel |
| Insufficient / negative | **CONTESTED** | Strong evidence both ways, or unresolved contradiction |
| | **INSUFFICIENT** | Relevant records exist; best belief `< θ_abstain` |
| | **UNRESOLVED** | Derivation not finished within budget |
| | **UNKNOWN-ABSENT** | Knowledge Sketch: key never stored |
| | **UNKNOWN-DECLARED** | Matches an OPEN_QUESTION record |
| | **UNKNOWN-NO-BASIS** | No applicable record and no in-envelope generator |
| Memory lifecycle (orthogonal) | NEW, CORROBORATED, USED, CONSOLIDATED, STABLE, CONTESTED, DEPRECATED, DECAYED | `04` §9 |

## 6. Goal decision procedure (evaluated when the Selector checks halting)

### 6.0 Goal hypothesis sets and goal keys
- **Goal hypothesis set:** every goal owns one PredictionRecord at its level (L1 for recognition goals, L2/L3 for completion/causal goals, L4 for plan goals, L5 for execution goals; factual goals use L2). Each **distinct candidate binding** of the answer variables, from any producer (retrieval, derivation, analogy, skill, Simulator), becomes one Hypothesis. Bindings equal within value tolerance are merged and their evidence pooled under the independence rule (§4). Plus the RESIDUAL. Hypotheses are created lazily as producers emit candidate bindings.
- **Goal keys:** the Sketch keys implied by the goal pattern: each bound entity `(e)`, each `(e, relation)` pair, each `(concept)` and `(concept, property)` mentioned, and the normalized alias of every unlinked surface form.
- **Derivability check:** before declaring a key absent-and-unsuppliable, REGRESS is queried with the key's relation. If any procedure, causal schema or rule exists whose output can produce that relation (a rule-index lookup by `relation_id`), the key is treated as *derivable* and step 1 does not fire.

Given goal `g` with answer variables and its hypothesis set (evaluated in this order; D-022):
1. If any goal key is absent in the Knowledge Sketch, is not present in the current input, and is not derivable (§6.0) → **UNKNOWN-ABSENT**.
2. If the pattern unifies with an OPEN_QUESTION and no untainted hypothesis has `b ≥ θ_commit` → **UNKNOWN-DECLARED**.
3. If there are no non-residual hypotheses → **UNRESOLVED** if the budget ran out with derivations pending, otherwise **UNKNOWN-NO-BASIS**.
4. Let `h*` = the hypothesis with the highest Support and `h_2` the runner-up (the RESIDUAL included).
   - **CONTESTED** if `h*` is involved in an unresolved contradiction.
   - **KNOWN-class answer** if: `b(h*) ≥ θ_answer(stakes)`, **and** `h*` is untainted, **and** its justification DAG has a path to OBSERVED / REMEMBERED / TESTED leaves, **and** `Support(h*) − Support(h_2) ≥ δ_margin`. The state is OBSERVED, REMEMBERED, DERIVED or TESTED accordingly.
   - **AMBIGUOUS** if other non-residual hypotheses lie within `δ_margin` of `h*` and `b(h*) ≥ θ_predict`: render the leading hypotheses with their evidence, or a conditional answer if they lie in different channels.
   - **PREDICTION-class answer** if `h*` is tainted but within its envelope and `b(h*) ≥ θ_predict`.
5. If every non-residual hypothesis is out of envelope (EXTRAPOLATED only) → **UNKNOWN-NO-BASIS**.
6. If the budget is exhausted with derivations pending → **UNRESOLVED**.
7. Otherwise (relevant hypotheses exist but none qualifies, e.g. untainted with `θ_predict ≤ b < θ_answer`) → **INSUFFICIENT**. This catch-all closes a gap in the original procedure, which left untainted hypotheses below `θ_answer` without a state.

**Stakes scaling:** `θ_answer(stakes) = θ_abstain + (θ_max − θ_abstain) · stakes` (F0 `θ_abstain` = 0.70, `θ_max` = 0.95).

**Difference between UNKNOWN and a low-confidence prediction:** UNKNOWN-* means there is no basis at all. PREDICTION-class or INSUFFICIENT means there is a basis, but it is weak or unpromoted. They are rendered differently (`07` §5.4).

## 7. Answer Record (why the model answered)
Stored for every decided goal:

| Field | Content |
|---|---|
| `goal`, `decision_state` | From §6 |
| `answer_claims` | Committed or predicted claims bound to the answer variables |
| `justification_subgraph` | All ancestor claims with producers, premises and evidence events |
| `sources` | Root sources with trust |
| `circuit_trace` | The executed CircuitGraph with node outcomes and costs |
| `rejected_hypotheses` | Each with Support and the decisive evidence against |
| `taint_map` | Taint status per claim and the promotion events applied |
| `envelope_flags` | Generators used and whether they were in-envelope |
| `budget` | Beats, resources used, why halting happened |

"Why?" questions are answered by rendering this record. Explanations are never generated after the fact.

## 8. Fact-placement rule
- Facts live **only** in ENGRAM records.
- Skills and Core components are trained with **fact-dropout**: training targets that require recalling world facts are masked, or supplied through retrieval inputs. Their outputs that look like factual claims are SUGGESTED until unified with engrams.
- This keeps the Knowledge Sketch's UNKNOWN-ABSENT meaningful and keeps knowledge addressable and editable.

## 9. Diagram B — Prediction and generalization

```
 new input / example(s)
        │
        ▼
 PARSE → scene graph over Property Basis ──────────────► L0/L2 prediction errors → salience
        │
        ▼
 STAGE 1: sparse retrieval of candidate concepts/cases (cheap, many)
        │
        ▼
 STAGE 2: ALIGN (structure-mapping) → match score + structured difference + candidate inferences
        │
        ├─► defining structure matched, mismatches within variability profile ──► DERIVED class membership
        │
        ▼
 HYPOTHESIS SET  {H1: support, evidence±, assumptions, generator} … {RESIDUAL "novel": mass}
        │   (candidate inferences, inherited defaults, causal predictions = PREDICTED / tainted)
        ▼
 TEST: observe · query memory · simulate (PREDICT-STEP) · counterexample search · Body execution
        │
        ├─ supported + grounded path ──► PROMOTE (taint removed) → knowledge record
        ├─ contradicted ──► RETRACT → TMS cascade · nogood · generator envelope shrinks
        └─ unresolved ──► stays PREDICTED / INSUFFICIENT / UNKNOWN-NO-BASIS
        │
        ▼
 RESIDUAL dominant? ──yes──► CONCEPT FORMATION ENGINE:
        │                     ABSTRACT across examples → attach parent (inherit as INHERITED)
        │                     → variability priors + size principle → contrast siblings
        │                     → NEW schema in Hippocampal Index
        ▼
 SLEEP: corroboration + use → CONSOLIDATED concept · variability priors updated ·
        recurring successful circuits → procedure schema → compiled skill (envelope set)
```
