# 02 — Data Ingestion Format: SRM Experience Format (SEF)

## 1. Why Final SRM does not consume data like a conventional LLM

A conventional LLM consumes data as `text → tokenizer → batches → next-token loss`. Every token costs the same, facts are not separated from hedges, and provenance is discarded. Final SRM needs to know, for each piece of input:
- **what kind** of information it is (fact, example, procedure, causal rule, …);
- **who asserts it** and how trustworthy that source is;
- **how certain** the assertion is (asserted, hedged, reported, hypothetical, negated, explicitly unknown);
- **what it refers to** (entities, properties, relations in canonical form);
- **whether it is already known, new, or conflicting.**

**SEF** is the structured format that carries this. All learning data, and all knowledge that enters the Library, passes through SEF. At inference time, raw user input is parsed by Perception into the *same* structures in memory (SceneGraph, `03` §8). SEF is the persisted, offline form of those structures.

SEF serves **three consumers**:
1. **Library population.** Facts, concepts, procedures, causal relations and contradictions are written into memory (§9).
2. **Training the machinery.** Tasks, skill demonstrations, examples and expression pairs train the parser, primitives, Composer, verifier and Mouth (§12).
3. **Evaluation.** Questions with gold answer states, including UNKNOWN (§4.14).

## 2. Pipeline overview

```
RAW SOURCE (text · code · math · tables · dialogs · tool outputs · synthetic generators)
   │
   ▼  2.1 Acquisition        → Source Envelope (provenance, license, data_category, trust_class)
   ▼  2.2 Normalization      → Unicode NFC, whitespace, encoding repair, language id, exact/near dedupe
   ▼  2.3 Segmentation       → segments (sentences/paragraphs · functions/modules · expressions · rows)
   ▼  2.4 Structural parsing → candidate units per modality (§6)
   ▼  2.5 Canonicalization   → entity linking · relation/property registry mapping · units/dates/values (§7)
   ▼  2.6 Epistemic tagging  → modality · polarity · attribution · hedge strength · extraction confidence (§10)
   ▼  2.7 Classification     → record kind (§4)
   ▼  2.8 Validation         → schema checks, quality gates, rejection log (§8)
   ▼
SEF SHARDS (JSON Lines, versioned, by data category) + DATASET MANIFEST
   │
   ▼  model-side ingestion (§9): encode → triage (known / novel / conflicting / noise) → memory operations
```

Stages 2.1–2.8 run **offline** in the data pipeline. They are deterministic tools plus optional teacher models; they are **not** part of the model and **not** components.

The model's own learned parser (`03` §8) is trained to reproduce stage 2.4–2.7 output from raw segments, so that at inference time the model can parse raw input itself.

## 3. Container format

| Item | Specification |
|---|---|
| Encoding | UTF-8 JSON Lines; one record per line |
| Shard | `sef/<data_category>/<shard_index>.jsonl` (optionally zstd-compressed `.jsonl.zst`); target shard size configurable (F0 default 64 MB) |
| Dataset manifest | One per dataset. Fields: `dataset_id`, `sef_version`, creation time, generator/converter versions, shard list with SHA-256 hashes and record counts per kind, data-category proportions, license summary, registry snapshot versions (entity/relation/property) |
| Versioning | `sef_version` (semantic version) in every record header and in the dataset manifest. Readers MUST reject unknown major versions |
| Large binary payloads | Not inline. Referenced by `blob_ref` (content hash plus path in a `blobs/` directory) |
| Ordering | Records within a shard follow source order. `source` and `entity` records appear before records that reference them, or in a dataset-level registry shard |

### 3.1 Common record header (every record)

| Field | Type | Meaning |
|---|---|---|
| `sef_version` | string | Format version, e.g. `1.0` |
| `record_id` | string | Globally unique within the dataset: `<dataset_id>:<kind>:<ordinal>` |
| `kind` | enum | One of §4 |
| `source_id` | string | Reference to a `source` record (required except for `source` itself) |
| `span` | object or null | `{segment_id, start, end}`: character offsets into the segment |
| `data_category` | string | Taxonomy path from §13, e.g. `code/python/stdlib-docs` |
| `extraction` | object | `{method: deterministic\|teacher\|human\|generator, method_version, confidence ∈ [0,1]}` |
| `epistemic` | object | §10.2: `{modality, polarity, hedge, attribution, declared_confidence}` |
| `context` | object | Qualifiers: `{time: interval or null, version: string or null, domain: string or null, locale: string or null, world_id: string or null}` |
| `links` | array | Optional typed references to other records: `{rel: supports\|contradicts\|example_of\|part_of\|derived_from\|answers, target: record_id}` |
| `content_hash` | string | SHA-256 of canonical content, used for deduplication |

## 4. Record kinds

| Kind | Purpose | Main consumer |
|---|---|---|
| `source` | Provenance: who or what produced the information | Evidence algebra, manifest data categories |
| `segment` | Raw text/code/math span the other records came from | Parser training, attribution |
| `entity` | A referent with canonical ID, type and aliases | Entity records in the Library |
| `property_def` | A Property Basis entry (name, kind, value type, unit) | Interface layer |
| `relation_def` | A relation registry entry (arity, argument types, cardinality, temporal flag, inverse) | Interface layer, contradiction detection |
| `fact` | An atomic claim: subject–relation–object/value with qualifiers | Engrams |
| `concept_definition` | An explicit definition: parent (genus), defining properties (differentiae), parts, functions | Concept schemas |
| `concept_example` | An instance description: parts, properties, relations, optional label, positive/negative | Concept Formation Engine |
| `procedure` | An ordered or partially ordered recipe with steps, preconditions and postconditions | Procedure schemas |
| `causal` | Cause → effect with conditions, mechanism and evidence type | `causes` links, causal schemas |
| `contradiction` | An explicit statement that two claims conflict, or a known dispute | Truth maintenance |
| `known_unknown` | An explicit statement that something is not known ("the cause of X is unknown") | Open-question records |
| `skill_demo` | An input → output demonstration, optionally with a trace and a checker | Composer, skill compilation |
| `task` | A problem with a goal, inputs, a **verifier** and optionally a hidden reference solution | Composer, Selector, Simulator training; evaluation |
| `question` | A query with a gold answer **state** (value, UNKNOWN-*, CONTESTED, …) | Evaluation and calibration training |
| `expression_pair` | An UtterancePlan ↔ surface text (or code) pair | Mouth training |

### 4.1 `source`

| Field | Type | Meaning |
|---|---|---|
| `source_type` | enum | `generator_truth`, `tool_execution`, `curated_kb`, `reference_doc`, `textbook`, `code_repository`, `api_docs`, `web_document`, `forum`, `user_statement`, `model_output`, `third_party_component` |
| `trust_class` | enum | Maps to a trust prior (§10.1) |
| `uri` / `title` / `author` | string or null | Provenance |
| `published_at` / `retrieved_at` | timestamp or null | Time context |
| `license` | string | SPDX identifier or a policy tag |
| `root_source_id` | string or null | For derivative sources (a copy, quotation or summary of another source). **Used to deduplicate evidence**: claims sharing a root count once |
| `privacy_scope` | enum | `public`, `user_private`, `org_private` |

### 4.2 `segment`
`{segment_id, modality: text|code|math|table|dialog|tool_output, language, text or blob_ref, parent_segment_id, position}`

### 4.3 `entity`
`{entity_id, canonical_name, entity_type (concept id), aliases: [string], disambiguation: string, external_ids: {...}, provisional: bool}`

`provisional = true` means the linker created it without a confident match (§7.1).

### 4.4 `property_def` / `relation_def`
- `property_def`: `{property_id, name, kind: geometric|dynamic|functional|relational|computational|contextual|other, value_type: bool|int|float|range|enum|entity_ref|text, unit (canonical SI or domain unit), enum_values, description}`
- `relation_def`: `{relation_id, name, arity, arg_types: [concept ids or value types], cardinality: functional|multi, temporal: bool, symmetric: bool, inverse_of: relation_id or null, description}`

`cardinality = functional` (one value per subject at a given time and context) is what enables contradiction detection (§11).

### 4.5 `fact`

| Field | Type | Meaning |
|---|---|---|
| `subject` | entity_id | Who or what the claim is about |
| `relation` | relation_id | Canonical relation |
| `object` | entity_id or typed value | `{type: int\|float\|bool\|enum\|text\|date\|range\|entity_ref\|code_ref, value, unit, tolerance}` |
| `qualifiers` | object | Time, version, location, condition (`condition` holds a fact pattern; the fact is then conditional) |
| `epistemic` | header | Modality and polarity are the key fields here (§10.2) |

### 4.6 `concept_definition`
`{concept_id, name, parent_ids: [concept_id], defining_properties: [{property_id, constraint: value|range|enum-set|bool, weight: optional}], typical_properties: [...] (non-defining defaults), parts: [{role, concept_id, count: range}], functions: [{property_id or relation pattern}], relations: [patterns], invariances: [{transform: scale|color|material|rotation|..., preserves_identity: bool}], contrast_with: [concept_id]}`

### 4.7 `concept_example`
`{example_id, label: concept_id or null, polarity: positive|negative, scene: SceneGraph-JSON (referents, properties with values, relations, parts), context, salient_features: optional [property_id]}`

Unlabeled examples are allowed; they feed concept discovery.

### 4.8 `procedure`
`{procedure_id, goal: fact pattern or task type, inputs: [{name, type}], outputs: [{name, type}], steps: [{step_id, action: text + optional structured op (primitive/skill/procedure reference with argument bindings), pre: [fact pattern], post: [fact pattern], depends_on: [step_id]}], invariants: [fact pattern], domain}`

### 4.9 `causal`

| Field | Meaning |
|---|---|
| `cause` / `effect` | Fact patterns or event descriptions (with variables) |
| `conditions` | Fact patterns that must hold (enabling conditions) |
| `polarity` | `causes`, `prevents`, `enables` |
| `strength` | Optional probability or effect size |
| `mechanism` | Optional structured chain of intermediate causal steps, or free text |
| `evidence_type` | `mechanism`, `intervention`, `observational`, `statement` |
| `scope` | Context restrictions (domain, conditions) |

**Admission rule:** `observational` alone produces only an `associated-with` link, never `causes` (§9, `04` §6).

### 4.10 `contradiction`
`{claims: [record_id or fact pattern], nature: value_conflict|polarity_conflict|temporal_conflict|definitional_conflict|disputed, resolution: null|{preferred: record_id, reason}, declared_by: source_id}`

### 4.11 `known_unknown`
`{pattern: fact pattern with variable(s), scope: global|source|domain, as_of: date, note}`

Example: "the boiling point of compound Z has not been measured." This creates an **open-question record**. Queries matching it return **UNKNOWN-DECLARED**, a subtype of UNKNOWN with provenance (`06` §5).

### 4.12 `skill_demo`
`{skill_hint: optional name, input: typed value(s) or SceneGraph, output: typed value(s), trace: optional [circuit steps], checker: optional verifier spec, domain}`

### 4.13 `task`

| Field | Meaning |
|---|---|
| `task_id`, `family` | The task family drives curriculum and evaluation splits |
| `goal` | Goal description: a fact pattern with answer variables, or a typed output specification |
| `inputs` | Typed inputs (values, scenes, code, I/O examples) |
| `verifier` | `{type: exact\|tolerance\|unit_tests\|property_tests\|proof_check\|constraint_set\|reference_compare, spec}`. Every training task MUST have a verifier |
| `reference_solution` | Optional circuit graph, DSL program or proof (used for imitation, never shown at test time) |
| `split` | `train`, `dev`, `test`, `heldout_composition`, `heldout_concept`, … |
| `difficulty` | Optional generator-provided difficulty |

### 4.14 `question`
`{query: text and/or fact pattern, gold_state: KNOWN|PREDICTION|UNKNOWN-ABSENT|UNKNOWN-NO-BASIS|UNKNOWN-DECLARED|CONTESTED|INSUFFICIENT, gold_answer: value or null, acceptable_hedges, split}`

- `KNOWN` = any knowledge-class decision (`06` §5).
- `PREDICTION` = a prediction-class answer is the correct behavior (e.g. a novel object whose class can only be inferred).
- The other values match the decision states of `06` §6 exactly.

### 4.15 `expression_pair`
`{plan: UtterancePlan-JSON (`07` §5.1), target_text, target_format: prose|markdown|json|python|dsl|latex|..., style: optional}`

## 5. Illustrative records

These examples are illustrative, not normative. Field sets follow §3–4.

```text
{"sef_version":"1.0","record_id":"ow1:fact:1042","kind":"fact","source_id":"ow1:source:gen",
 "data_category":"synthetic/objectworld","extraction":{"method":"generator","method_version":"ow-0.1","confidence":1.0},
 "epistemic":{"modality":"asserted","polarity":"positive","hedge":0.0,"attribution":null,"declared_confidence":null},
 "context":{"world_id":"ow1"},"subject":"ow1:ent:wheel_17","relation":"rel:diameter_m",
 "object":{"type":"float","value":1.2,"unit":"m","tolerance":0.05},"qualifiers":{}}

{"sef_version":"1.0","record_id":"web7:fact:88","kind":"fact","source_id":"web7:source:3",
 "data_category":"web/encyclopedic","extraction":{"method":"teacher","method_version":"x","confidence":0.82},
 "epistemic":{"modality":"reported","polarity":"positive","hedge":0.4,"attribution":"ent:dr_smith","declared_confidence":null},
 "subject":"ent:compound_z","relation":"rel:boiling_point_c","object":{"type":"float","value":212,"unit":"degC"}}

{"sef_version":"1.0","record_id":"ow1:causal:12","kind":"causal","source_id":"ow1:source:gen",
 "cause":{"pattern":"(?v rel:load_kg ?l) & (?l > 20000)"},"effect":{"pattern":"(?v rel:requires_part concept:heavy_duty_wheel)"},
 "conditions":[],"polarity":"causes","evidence_type":"mechanism","mechanism":"load per wheel exceeds passenger-wheel rating"}
```

## 6. Raw → SEF conversion by modality

### 6.1 Text
1. **Segmentation:** paragraphs, then sentences (rule-based splitter, configurable).
2. **Claim extraction:**
   - *F0 synthetic data:* the generator emits gold SEF directly, plus the rendered text (pairs for parser training).
   - *Natural language:* a teacher extractor (rule-based for controlled language; an optional teacher LLM for open text, recorded as `method: teacher` with its version) produces atomic facts, definitions, procedures and causal statements.
3. **Epistemic tagging:**
   - a hedge lexicon with learned scoring ("may", "probably", "it is believed") sets `hedge` and `modality: hedged`;
   - attribution patterns ("X says", "according to") set `modality: reported` and `attribution`;
   - conditionals set `qualifiers.condition`;
   - negations set `polarity: negative`;
   - questions become `question` records (no gold) or are dropped;
   - statements of ignorance become `known_unknown`.
4. **Definitions** ("A wheel is a circular component that rotates on an axle…") become `concept_definition` (genus = parent, differentiae = defining properties).
5. **Instructions and recipes** become `procedure`.
6. **Explicit causal language** ("because", "causes", "leads to", "prevents") becomes `causal` with `evidence_type: statement`, upgraded to `mechanism` when a mechanism is stated and to `intervention` when an experiment is described.

### 6.2 Code (F0: list/integer DSL and a Python subset)
1. **Parse** with deterministic parsers (Python `ast`; the DSL's own grammar), producing a typed AST with symbol tables and data-flow edges.
2. **API facts** from signatures, docstrings and package metadata: `fact(subject=api_symbol, relation=rel:signature|rel:returns|rel:raises|rel:deprecated_in, …)` with `context.version` from package metadata. Version context is mandatory for API facts.
3. **Idioms and patterns:** recurring AST subtrees across files are mined (offline frequent-subtree mining) into candidate `concept_definition` records of kind *code pattern* with typed holes.
4. **Tests** become `task` records (verifier `unit_tests`) and `skill_demo` records (input → output pairs from assertions).
5. **Algorithms** (documented functions with docstrings) become `procedure` records (steps derived from the AST control-flow outline) plus facts about complexity and invariants when stated.
6. **Bug-fix commits** (when history is available) become a `causal` record (error signature → cause → fix) and a `skill_demo` (before → after).

### 6.3 Mathematics
- Expressions are parsed (SymPy) into expression trees.
- Theorem and identity statements become `fact` records (subject = expression pattern, relation = `rel:equals|rel:implies`).
- Worked solutions become `procedure` + `skill_demo`.
- Exercises with answers become `task` records (verifier: `tolerance` or symbolic `reference_compare`).

### 6.4 Tables and structured data
Each row becomes an entity (if the row key is an entity) plus one `fact` per column (relation from a header → relation registry mapping). Units are taken from headers or schema.

### 6.5 Dialog and user statements
`source_type: user_statement`, trust class `user`. Requests become goals at inference time, not SEF facts. User-provided facts are SEF facts with `privacy_scope: user_private` when persisted.

### 6.6 Tool outputs (Body)
`source_type: tool_execution` (trust class `tool`). Execution results are OBSERVED evidence at inference time. Persisted runs become `skill_demo` or `task` evidence.

### 6.7 Synthetic generators (F0)
Specified in §14. They emit gold SEF directly with `method: generator`, `confidence: 1.0`, plus rendered raw text and code for parser and Mouth training.

### 6.8 Images and audio
Deferred. When added, they produce `concept_example` scenes via part/attribute parsers; the record kinds stay the same.

## 7. Canonicalization

### 7.1 Entity linking
1. Normalize the surface form (casefold, strip punctuation, NFC).
2. Generate candidates from an alias index (exact, then fuzzy within edit distance or embedding top-k).
3. Score = alias match × type compatibility × context similarity (dense embedding of the surrounding segment vs. entity description).
4. Link if the score ≥ `θ_link` (config, F0 default 0.85). Otherwise create a **provisional** entity, flagged for merge review during sleep (`08` §5).

### 7.2 Relation and property mapping
Map extracted predicate strings to the registries by alias lookup, then embedding nearest neighbour.
- If similarity ≥ `θ_rel` (F0 default 0.80), use the existing ID.
- Otherwise create a **provisional relation/property** (`relation_def` with `provisional: true`). Provisional entries are merged or promoted during sleep when used consistently.

### 7.3 Values
- Units are converted to canonical units (a unit library, e.g. pint).
- Dates are converted to ISO-8601 intervals.
- Numbers get explicit tolerance from stated precision (significant digits) when not given.
- Enumerations are mapped to registry enum values.

### 7.4 Deduplication
- Exact: `content_hash`.
- Near-duplicate: MinHash over normalized segment text (threshold configurable).
- Duplicates from the **same root source** are merged and don't add evidence. Duplicates from **different root sources** are kept as corroboration.

## 8. Validation and quality gates
- **Schema validation** for every record (types, required fields, registry references resolve).
- **Range sanity:** values within the property's declared range when one exists.
- **Extraction confidence:** records with `extraction.confidence < θ_noise` (F0 default 0.3) are dropped and logged as noise.
- **Licensing and privacy:** records whose license or privacy scope is incompatible with the dataset's intended use are excluded.
- **Rejection log:** every dropped record is logged with a reason code, for auditing.
- **Dataset manifest** totals are recomputed and hashed.

## 9. Model-side ingestion: how each kind enters the architecture

Every record is first **encoded** with the frozen interface layer (`03`):
- referents and values become dense vectors and content codes;
- the structure becomes a SceneGraph fragment.

Then **triage** runs (§9.1), and finally the operation for the kind (§9.2).

### 9.1 Triage algorithm (per claim-bearing record)
1. **Sketch check** (`04` §7) on `(subject)` and `(subject, relation)`.
2. **Exact lookup** through the structured triple index (`04` §4.3) for `(subject, relation)` under compatible qualifiers (overlapping time, compatible version and context).
3. **Outcomes:**
   - **KNOWN-SAME:** an existing record has an equal object (within tolerance) and the same polarity. If the root source is new, add corroborating evidence (§10.3); otherwise only update usage statistics. *No new record and no gradient.*
   - **CONFLICT:** the relation is functional, the qualifiers overlap, and the objects differ beyond tolerance, or the polarity is opposite. Create a contradiction record, add `contradicts` links, and add evidence against the weaker side when the trust difference exceeds `θ_trust_gap`; otherwise both become CONTESTED.
   - **NOVEL:** write to the Hippocampal Index with lifecycle stage NEW.
   - **NOISE:** failed validation, or extraction confidence below the threshold. Drop.
4. **Linking:** Hebbian links to co-active records from the same segment and episode; `example_of`, `part_of` and similar links come from the record's `links`.

### 9.2 Operation by record kind

| Kind | Destination | Representation created | Initial epistemic state | Notes |
|---|---|---|---|---|
| `source` | Source registry (part of the provenance store) | Source entry with trust prior | — | Required before claims from it |
| `entity` | Library ENTITY records (+ alias index) | Identity code, content code, dense | REMEMBERED (generator / curated) or NEW | Provisional entities flagged |
| `property_def` / `relation_def` | Registries (interface layer) | Property/relation entries | — | After interface freeze: only additive minor versions (`03` §1.3) |
| `fact` (asserted) | Hippocampal Index → Library ENGRAM | Claim payload + structured triple index entries + Sketch keys | NEW → (lifecycle) | Evidence initialized by §10.3 |
| `fact` (reported) | ENGRAM of the form *attribution asserts P* | The attributed claim is the knowledge; P itself gets no evidence | NEW | P becomes REMEMBERED only via independent support |
| `fact` (hedged) | ENGRAM with hedge | Evidence scaled by modality factor | NEW, belief limited | — |
| `fact` (conditional) | ENGRAM with condition qualifier | Conditional claim | NEW | Used only when the condition holds in context |
| `fact` (negated) | ENGRAM, polarity negative | Negative claim | NEW | Explicit negatives can answer "no" questions |
| `concept_definition` | Library CONCEPT schema (via CFE integration) | Factored schema: parents, defining/typical properties with constraints, parts, functions, invariances | NEW (REMEMBERED once corroborated) | Variability profiles initialized from constraints (`04` §2.2) |
| `concept_example` | Concept Formation Engine (`05` §9) | Instance record; updates or creates schema | Instance: OBSERVED in its source; schema changes NEW | Labeled examples update the labeled concept; unlabeled ones go to the residual and discovery path |
| `procedure` | Library PROCEDURE schema | Circuit-graph template with typed holes | NEW | Execution success raises evidence |
| `causal` | `causes` / `prevents` / `enables` link + causal schema (if mechanism or intervention) or `associated-with` link (if observational) | Link with conditions and strength | NEW (hypothesis-level for observational) | Admission rule from §4.9 |
| `contradiction` | Truth-maintenance store + `contradicts` links | Contradiction record | Both claims → CONTESTED unless a resolution is given | Resolution with a strong reason adds evidence |
| `known_unknown` | OPEN-QUESTION record | Pattern + scope + as-of date | UNKNOWN-DECLARED | Matching queries are answered "not known (per source)" |
| `skill_demo` | Episodic trace store → compilation candidates | Demonstration episode | OBSERVED (demonstration) | Feeds Composer imitation and skill compilation |
| `task` | Training/eval stream (not the Library) | — | — | Used by bootstrap stages S3–S4 and evaluation |
| `question` | Eval / calibration stream | — | — | Gold states train abstention calibration |
| `expression_pair` | Mouth training stream | — | — | Stage S5 |

## 10. How uncertainty enters

### 10.1 Trust priors by source class (configurable table; F0 defaults)

| Trust class | Prior `t` | Notes |
|---|---|---|
| `generator_truth` | 1.00 | Synthetic ground truth (training worlds only) |
| `tool_execution` | 0.99 | Body observations |
| `curated_kb` | 0.90 | — |
| `reference_doc` / `textbook` / `api_docs` | 0.85 | Version-scoped |
| `code_repository` | 0.75 | — |
| `user_statement` | 0.70 | Scoped to the user/session unless corroborated |
| `web_document` | 0.50 | — |
| `forum` | 0.40 | — |
| `third_party_component` | 0.30 | Until its track record accrues (`10` §8) |
| `model_output` | 0.20 | The model's own past outputs **never** corroborate themselves (same root source) |

### 10.2 Modality mapping

| `modality` | Meaning | Factor `μ` | Storage rule |
|---|---|---|---|
| `asserted` | Plain statement | 1.0 | Claim P |
| `hedged` | "probably", "may" (`hedge ∈ (0,1]`) | `1 − 0.7·hedge` | Claim P with hedge recorded |
| `reported` | "X says P" | 0 for P; 1.0 for "X asserts P" | Attributed claim |
| `hypothetical` | "if …", "suppose …" | 0 | CONJECTURED pattern (not knowledge) |
| `conditional` | P holds under condition C | 1.0 (conditional) | Claim with condition qualifier |
| `negated` | not P | 1.0 | Polarity negative |
| `unknown_declared` | "it is unknown whether P" | — | `known_unknown` record |
| `fictional` | Inside fiction, examples, jokes | 0 | Stored only in a FICTION context (`world_id`), never as a world claim |

### 10.3 Evidence initialization
For a claim from source `s` with extraction confidence `c` and modality factor `μ`:

**Δe⁺ = κ · t(s) · c · μ**   (κ = evidence unit, config, F0 default 1.0)

- Negated claims add the same amount to the negative claim's own e⁺; they add evidence *against* the positive claim only through contradiction detection.
- **Corroboration:** additional sources with **different root sources** each add their own Δe⁺. Same-root duplicates add nothing.
- The resulting belief follows the evidence algebra (`05` §2): `b = e⁺/(e⁺+e⁻+W)`.

### 10.4 Extraction uncertainty vs. world uncertainty
`extraction.confidence` measures how sure the converter is that the source *says* this. `hedge` and `modality` measure how sure the *source* is. Both reduce evidence, but they are stored separately so that a better parser can later re-extract and recompute evidence.

### 10.5 What never enters as knowledge
- Hypothetical, fictional or reported content (as P itself).
- Low-confidence extractions.
- The model's own unverified outputs.

## 11. Contradiction detection at ingestion

| Conflict type | Rule |
|---|---|
| Value conflict | Functional relation, overlapping qualifiers, objects differ beyond the combined tolerance |
| Polarity conflict | Same (s, r, o) with opposite polarity and overlapping qualifiers |
| Temporal conflict | Same functional (s, r) with overlapping intervals and different values. If the intervals don't overlap, there is **no conflict** (the value changed over time) |
| Definitional conflict | A `concept_definition` whose defining constraint is violated by a labeled positive `concept_example` (or vice versa) |
| Version conflict | API facts differing across `context.version`: **not** a conflict (version-scoped); stored as separate context segments |

Handling:
1. Create a contradiction record.
2. If `|t(s1)·c1 − t(s2)·c2| ≥ θ_trust_gap` (F0 default 0.3), evidence goes against the weaker claim.
3. Otherwise both claims become CONTESTED and stay so until further evidence arrives or sleep repair resolves them (`08` §5).

## 12. Which SEF kinds train which components

| Component (`09` §1) | Trained on |
|---|---|
| Interface layer (Property Basis, content-code projection) | `concept_definition`, `concept_example`, `fact` (property-bearing), `property_def`, `relation_def` |
| Perception parsers | (`segment`, gold structured records) pairs |
| ALIGN affinity, COMPARE weighting | `concept_example` pairs with known correspondence (synthetic) |
| PREDICT-STEP dynamics, causal schemas | `causal`, state-transition `task` records |
| Verifier (Error Monitor) | `fact`, `contradiction`, corrupted facts, `question` |
| EXPLAIN calibration, abstention | `question` with gold states, `concept_example` recognition tasks |
| Composer policy, Selector | `task` (with verifier and reference solution), `skill_demo` |
| Compiled skills | `skill_demo` + verified Composer traces |
| Concept Formation Engine priors | `concept_definition` hierarchies, few-shot `concept_example` episodes |
| Mouth renderer and articulators | `expression_pair` |
| Library contents | `entity`, `fact`, `concept_definition`, `procedure`, `causal`, `contradiction`, `known_unknown` |

## 13. Data category taxonomy (used in the Manifest)

A hierarchical path, `<domain>/<subdomain>/<source-kind>`. Top-level domains (extensible):
`synthetic/*`, `code/*`, `math/*`, `science/*`, `encyclopedic/*`, `procedural/*`, `dialog/*`, `web/*`, `tool/*`, `user/*`, `eval/*`.

The Manifest records, for each component version, the **proportion of training signal by category**, never raw data (`09` §4).

## 14. F0 synthetic generators (specification)

### 14.1 ObjectWorld
- **Concept hierarchy:** configurable depth (F0: 4) and branching (F0: 3–6). Total concepts configurable (F0: 300).
- Each concept has:
  - **defining properties** (part structure, functions, key geometric/dynamic properties) with narrow variability;
  - **typical properties** with wider variability;
  - **parts** (other concepts), relations to other concepts, invariances.
- **Property Basis ground truth:** F0 uses 200 properties of all kinds.
- **Instances** are sampled from concept distributions (values drawn within variability profiles).
- **Context scenes** place instances in environments (e.g. an airport, a road, a workshop) with relations.
- **Held-out splits:**
  - (a) *held-out compositions*: property and part combinations never seen in training concepts;
  - (b) *held-out concepts*: entire leaf concepts withheld, shown later with 1–5 examples;
  - (c) *wheel-style* out-of-range instances: a known concept with typical properties far outside the training range plus discriminating context.
- **Rendering:** each scene and definition is rendered into controlled English from multiple paraphrase templates, giving (text, gold SEF) pairs.

### 14.2 DSLWorld
- A typed list/integer DSL with about 40 primitives (map, filter, fold, sort, reverse, take, drop, zip, arithmetic, comparisons, conditionals, recursion via fold).
- **Tasks** are generated by sampling typed programs (depth-limited) and producing I/O examples; verifiers are unit tests plus property tests.
- **Splits** hold out program *compositions* (primitive pairs never adjacent in training).
- A Python-subset rendering of the same programs is used for code-generation tasks.
- **F0 Python subset:**
  - **Allowed:** function definitions with type annotations; `int`, `bool`, `str`, `list`, `dict`, `tuple`; arithmetic and comparison; `if`/`elif`/`else`; `for`/`while`; list/dict comprehensions; `return`; calls to whitelisted builtins (`len`, `range`, `sorted`, `min`, `max`, `sum`, `abs`, `enumerate`, `zip`, `set`, `reversed`, `any`, `all`); recursion.
  - **Disallowed:** imports (except a whitelist of pure modules, empty in F0), I/O, classes, exceptions other than `ValueError`, global state.
  - Enforced by an AST validator before execution and by the sandbox.

### 14.3 FactStream
- Entities with attributes and relations, changing over time.
- Facts are emitted as a time-ordered stream from sources with varied trust, including injected contradictions, hedged and reported statements, and declared unknowns.
- **Withheld entities** are never emitted (for UNKNOWN-ABSENT tests).
- **Partially described entities** have some relations never emitted (for UNKNOWN-NO-BASIS / INSUFFICIENT tests).

### 14.4 MathWorld-lite
- Arithmetic word problems and algebra identities from templates with symbolic verification.
- Perturbed variants (number and name changes, distractor clauses) for robustness tests.

### 14.5 Generator outputs
Every generator emits:
- gold SEF records (`method: generator`);
- raw renderings (text/code);
- `task` and `question` records with verifiers and gold states;
- `expression_pair` records for the Mouth;
- a dataset manifest with split statistics.

Generators are **deterministic given a seed** and versioned (`ow-x.y`, `dsl-x.y`, …).

### 14.6 F0 dataset sizes (defaults; Build Configuration `learning.bootstrap.data`)

| Dataset | Train | Dev | Test / held-out |
|---|---|---|---|
| ObjectWorld concepts / instances | 300 concepts (≈ 250 train-visible) / 1e6 instances | 5e4 instances | 50 held-out concepts; 5e4 held-out-composition instances; 1e4 wheel-style instances |
| ObjectWorld rendered texts (parser pairs) | 5e5 | 2e4 | 2e4 |
| DSLWorld tasks | 2e5 | 1e4 | 1e4 held-out compositions |
| FactStream | 1e6 facts over 1e5 entities | 5e4 questions | 5e4 questions (incl. 1e4 withheld-entity, 5e3 declared-unknown, 5e3 contested) |
| MathWorld-lite | 1e5 problems | 5e3 | 5e3 + 5e3 perturbed |
| Expression pairs | 3e5 | 1e4 | 1e4 |

Library-size sweeps (1e4 → 1e7 records) are produced by scaling FactStream entities and ObjectWorld instances with the same generators.
