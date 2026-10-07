# 03 — Interface Layer and Perception & Parsing

# Part A — Interface layer (the shared language of all components)

## 1. Role, versioning and freezing

### 1.1 Role
Every component communicates **only** through typed messages defined here (§5), expressed in the code space (§2) and the registries (§3–4). The interface layer is the model's binary interface between components (its "ABI"). It is what makes components independently retrainable (`10`).

### 1.2 Contents
- Code-space definition (`B`, `L`, band layout).
- Content-code projection and code embedding tables.
- Role registry.
- Property Basis.
- Relation registry.
- Concept-type and value-type registries.
- Message schemas.
- Primitive signatures (`05` §4).
- Codec (§6).

### 1.3 Versioning
`interface_version = MAJOR.MINOR`.
- **MAJOR** changes break compatibility: changing `B`, `L` or `d`; retraining the projection or embedding tables; changing role codes or message schemas incompatibly.
- **MINOR** changes are additive and compatible: new properties, relations, roles or message optional fields, appended with fresh codes. Existing entries are never modified.
- **ABI hash** = SHA-256 over the canonical description of: `B`, `L`, `d`, band layout, the digest of the projection and embedding weights, role table digest, registry digests up to the frozen MAJOR baseline, message schema version and primitive signature set.

### 1.4 Freeze rule
After bootstrap stage S1 (`08` §2), the projection and embedding tables, existing registry entries and message schemas are **frozen**. Later learning may only *append* registry entries (MINOR) or create a new MAJOR version through migration (`10` §9).

## 2. Code space

### 2.1 Format
A **sparse block code** `c ∈ {0,…,L−1}^B`: `B` blocks, each with exactly one active unit. Dense view: a `B·L` binary vector with `B` ones (sparsity `1/L`). Storage: `B` bytes when `L ≤ 256`.

### 2.2 Algebra (exact)

| Operation | Definition | Properties |
|---|---|---|
| **bind** `a ⊗ b` | `(a ⊗ b)[i] = (a[i] + b[i]) mod L` | Exact, invertible, commutative, keeps sparsity, output dissimilar to both inputs |
| **unbind** `c ⊘ b` | `(c ⊘ b)[i] = (c[i] − b[i]) mod L` | `(a ⊗ b) ⊘ b = a` exactly |
| **permute** `ρ_k(a)` | `ρ_k(a)[i] = a[(i + k) mod B]` | Encodes order or position (sequence roles) |
| **overlap** `ov(a, b)` | `|{i : a[i] = b[i]}|` | Similarity in `[0, B]`; for random codes `ov ~ Binomial(B, 1/L)` |
| **bundle** `⊕{a_j}` | Count matrix `M ∈ ℕ^{B×L}`, `M[i, a_j[i]] += w_j` | Keeps all members (a *soft* bundle) |
| **sparsify** `sp(M)` | `sp(M)[i] = argmax_l M[i, l]` (ties broken by seeded hash) | Lossy; overlap with each member is about `B/n` for `n` equal members |
| **match to bundle** `ovM(q, M)` | `Σ_i M[i, q[i]]` | Membership score of `q` in bundle `M` |
| **unbind bundle by role** `M ⊘ ρ` | Row-wise cyclic shift of `M[i,·]` by `−ρ[i]` | Gives a filler distribution per block; cleaned up against a codebook (`04` §5.4) |

### 2.3 Two kinds of codes
- **Identity codes:** random, seeded per symbol (entities, concepts, records, roles), nearly orthogonal. Used for exact binding and record identity.
- **Content codes:** produced from dense vectors by the learned projection (§2.4), so similar meanings give overlapping codes. Used for retrieval.

Every Library record has **both**.

### 2.4 Content-code projection (learned, frozen after S1)
- For dense `x ∈ ℝ^d`, compute per-block logits `z_i = W_i x + β_i ∈ ℝ^L`; `code[i] = argmax z_i`.
- Training uses Gumbel-softmax with a straight-through estimator (temperature annealed), plus a similarity-preservation loss so that `ov(code(x), code(y))/B` tracks `cos(x, y)` monotonically (`08` §2, S1).

### 2.5 Code embedding (learned, frozen after S1)
- `emb(c) = normalize(Σ_i E_i[c[i]])`, with `E_i ∈ ℝ^{L×d}`.
- Bundles embed as `normalize(Σ_i Σ_l M[i,l] · E_i[l])`.

### 2.6 Structure codes
- A structure (scene, schema, claim) is the bundle `⊕_j (role_j ⊗ code(filler_j))`, kept as a count matrix in the Workspace and as a sparsified code for storage.
- Exact relational queries use unbind-by-role followed by cleanup.
- **Retrieval never relies on sparsified bundles.** Retrieval uses content codes of learned dense embeddings (§2.4), because sparsified bundles don't survive banded indexing.

### 2.7 Relational-signature code
- For a graph `G`, the signature dense vector is `s(G) = SetEncoder({emb(edge_type) ⊙ emb(type(src)) ⊙ emb(type(dst))})`, a content-free relational pattern.
- Its content code `code(s(G))` is stored in a separate **signature index** (`04` §5) for analogy retrieval.
- The SetEncoder is part of the interface layer (frozen after S1).

## 3. Property Basis

Each property entry:

| Field | Meaning |
|---|---|
| `property_id` | Stable registry ID |
| `name`, `description` | Human-readable labels |
| `kind` | geometric, dynamic, functional, relational, computational, contextual, other |
| `value_type` | bool, int, float, range, enum, entity_ref, text |
| `unit`, `enum_values`, `range` | Value constraints |
| `dense` | `ℝ^d` embedding (learned in S1, frozen) |
| `content_code`, `identity_code` | Codes (§2.3) |
| `value_encoder` | How values become fillers: thermometer/log-scale bucket codes for scalars (bucket count configurable), identity codes for enums, `bool → {TRUE, FALSE}` codes |
| `provisional` | True if created after the freeze and not yet promoted |

- **Growth:** new properties are appended (MINOR interface version). Their dense vector is initialized by the Concept Formation Engine as the mean of the instances that motivated it, projected to a code with the frozen projection. Projection and embedding tables do not change.
- **Grounding:** in S1, the synthetic worlds supply ground-truth property values, so each property direction is trained against real factors (`08` §2).

## 4. Registries

| Registry | Entry fields | Notes |
|---|---|---|
| **Role registry** | `role_id`, name, identity code, arity position | Initial set (F0 `N_role` = 64): subject, object, agent, patient, instrument, part, whole, property, value, function, cause, effect, condition, before, after, input, output, state, goal, container, element, index, … plus sequence positions via `ρ_k` |
| **Relation registry** | From SEF `relation_def` (`02` §4.4) plus identity/content codes | Cardinality and temporal flags drive contradiction detection |
| **Concept-type registry** | Top-level types (physical object, artifact, agent, event, process, quantity, code construct, math object, place, time, abstract) | For type compatibility in alignment and linking |
| **Value-type registry** | int, float(unit), bool, enum, date, range, text, entity_ref, code_ref, expr_ref | For typed message ports |

## 5. Message types (the only things components exchange)

All messages are Python dataclasses with tensor fields (implementation note: batched as structures of arrays).

| Message | Fields |
|---|---|
| `Code` | `blocks: uint8[B]` |
| `Bundle` | `counts: uint16[B, L]` (sparse COO allowed) |
| `Dense` | `vec: float16/32[d]` |
| `Value` | `type`, `payload`, `unit`, `tolerance` |
| `Referent` | `ref_id`, `type_id`, `dense`, `content_code`, `identity_code`, `bound_record: record_id or null` |
| `Claim` | `claim_id`, `subject: Referent ref`, `relation_id`, `object: Referent ref or Value`, `qualifiers`, `polarity`, `modality`, `content_code`, `dense`, epistemic block (`e_plus`, `e_minus`, `state`, `taint: bool`, `lifecycle`), `justification: {producer_id (node/component), premises: [claim_id] ≤ A, weights}`, `provenance: [source_id]`, `channel_mask: bitset[H]` |
| `Hypothesis` | `hyp_id`, `content: Claim or SceneGraph ref`, `support`, `evidence_for: [EvidenceEvent ref]`, `evidence_against`, `assumptions: [claim_id]`, `generator_id`, `generator_version`, `in_envelope: bool` |
| `PredictionRecord` | `pred_id`, `level: L0…L5`, `target`, `hypotheses: [Hypothesis]`, `residual_mass`, `horizon`, `status` (`06` §2) |
| `EvidenceEvent` | `event_id`, `target_claim/hyp`, `delta_plus`, `delta_minus`, `kind: observation\|memory\|verification\|test\|derivation\|corroboration\|contradiction\|prediction_error`, `source_root_id`, `producer_id` |
| `SceneGraph` | `nodes: [Referent]`, `properties: [(ref_id, property_id, Value, confidence)]`, `relations: [(relation_id, [ref_id])]`, `claims: [Claim]`, `parts: [(whole_ref, part_ref, role_id)]`, `context`, `structure_bundle: Bundle`, `signature_code: Code` |
| `Goal` | `goal_id`, `pattern: Claim with variables or typed output spec`, `answer_vars: [(var, type)]`, `constraints`, `stakes: float`, `deadline_beats` |
| `CircuitNode` / `CircuitGraph` | `05` §5.1 |
| `UtterancePlan` | `07` §5.1 |
| `Bid` | `process_id`, `resources: {flops, bytes, slots, verifier_calls, tokens, tool_calls}`, `predicted_value`, `confidence` |
| `ImplRepr` | Typed AST with `Hole(type)` nodes for code/DSL/math (§9.3) |

---

# Part B — Codec

## 6. Codec
- **Input:** UTF-8 bytes → NFC normalization → newline normalization. The perception chunker operates on bytes (no tokenizer dependency for understanding).
- **Output vocabulary:** the Mouth's neural renderer uses a byte-level BPE vocabulary of size `V_out` (F0 default 8192), trained on `expression_pair` targets. Special tokens: `<bos> <eos> <pad> <cite:k> <hedge> <abstain> <code> </code>`. Formal targets bypass the vocabulary through deterministic unparsers (`07` §5).
- **Versioning:** the codec is part of the interface layer. A vocabulary change is a MAJOR change for the Mouth only (articulator compatibility).

---

# Part C — Perception & Parsing

## 7. Perception pipeline

### 7.1 Chunker
Splits the byte stream into chunks. Text uses sentence boundaries (rule-based), capped at `chunk_max_bytes` (F0 128). Code uses top-level statements or functions. Math uses expressions. Tables use rows.

### 7.2 Perception encoder (component `perception.encoder`)
- Byte embedding (256 + specials) → a local-window attention stack (`n_layers_perc` = 4, window = chunk) interleaved with a gated linear recurrence across chunks (state size `d`).
- Output per chunk: `h_chunk ∈ ℝ^d`, plus per-byte states for the parser.
- All sizes come from the Build Configuration.

### 7.3 L0 predictor
- A small MLP on the recurrent state predicts the next chunk's `h`.
- Prediction error: `ε0 = 1 − cos(ĥ, h)`.
- Trained by the local prediction loss (`08` §4).

### 7.4 Recognition-by-recall
1. `q = code(h_chunk)`; stage-1 lookup (`04` §5).
2. Familiarity `f = max_c ov(q, c)/B` over the candidates. Null model: `ov ~ Binomial(B, 1/L)`, `z = (f·B − B/L)/sqrt(B·(1/L)(1−1/L))`.
3. **KNOWN** if `f ≥ θ_known` (F0 0.80) **and** `cos(h_chunk, dense(c*)) ≥ θ_cos` (F0 0.90) **and** `ε0 ≤ θ_err`. The chunk is represented as a pointer to `c*`; usage statistics are updated; deep parsing is skipped.
4. Otherwise **NOVEL**: send to the structural parser.

## 8. Structural parsing

### 8.1 Output
A `SceneGraph` (§5) with claims carrying modality, polarity and qualifiers.

### 8.2 Parser components (one per modality; component type `perception.parser.<modality>`)

| Modality | F0 implementation |
|---|---|
| Code (DSL, Python subset) | **Deterministic**: grammar parser / Python `ast` → typed AST → SceneGraph (nodes = AST constructs and symbols; relations = calls, data flow, contains, defines). Wrapped as a component for uniformity, but has no learned weights |
| Math | **Deterministic**: SymPy parse → expression tree → SceneGraph |
| Tables | **Deterministic**: schema mapping → facts |
| Controlled-language text (synthetic) | **Learned**: sequence-to-graph parser (encoder states → pointer network producing referent spans, property/relation labels from the registries, modality tags). Trained on (rendered text, gold SEF) pairs (`08` S2) |
| Open text | F0: offline teacher extraction (SEF conversion, `02` §6.1) for data. At inference, the learned parser runs with lower extraction confidence; if parser confidence is below `θ_parse`, the content is treated as an opaque segment (stored, but not converted to claims) |

### 8.3 Entity linking at inference
Same algorithm as `02` §7.1, using the Library's alias index and entity content codes.

### 8.4 Epistemic tagging at inference
The learned parser includes heads for modality, polarity, hedge strength and attribution, matching SEF semantics (`02` §10).

## 9. Request intake: goals

### 9.1 Question → Goal
- A question is parsed to a `Goal` whose pattern has answer variables. Example: "What vehicle does this wheel belong to?" becomes the pattern `(wheel_ref, rel:part_of, ?v)` with `?v : concept(vehicle)`.
- Yes/no questions become patterns with a boolean answer variable over a claim.
- "Why" questions request the Answer Record of a prior answer, or a causal explanation goal `(?cause, rel:causes, event)`.

### 9.2 Task → Goal
A task request ("write a function that…", "solve…", "plan…") becomes a Goal with a typed output specification (`ImplRepr` of a given type, a plan graph, or a value) plus constraints (tests, requirements).

### 9.3 Implementation representation (`ImplRepr`)
- A typed AST in the target formal language (DSL, Python subset, math), whose nodes may be `Hole(type, constraints)`.
- The Composer fills holes. The Mouth's deterministic unparser renders complete ASTs.
- Type checking uses the language's type rules: the DSL has static types; the Python subset uses annotation-guided checking in F0.

### 9.4 Stakes
`Goal.stakes ∈ [0,1]` comes from an explicit user/API parameter or a learned stakes classifier (default 0.5). It feeds the 5-HT modulator (`07` §3).

## 10. Triage tags emitted by Perception
Each parsed unit carries a provisional kind tag (fact, example, procedure, causal, contradiction candidate, question/goal, instruction, noise) and a novelty flag, which feed `02` §9.1 triage and the Thalamic Gate salience (`07` §2).
