# 13 — Answers to the 50 Architecture Questions (Final Research Phase)

Each answer is short and points to the normative section.

1. **How does SRM's prediction system change?**
   - Seven typed levels: L0–L5 World Prediction, L6 Expression in the Mouth only.
   - Prediction records keep hypotheses, evidence, assumptions and generator.
   - Taint until promotion; validity envelopes; Support ranking instead of generator probability.
   - (`06` §1–4)

2. **Principles taken from predictive processing:**
   - hierarchical prediction with upward errors;
   - precision weighting (Gate, ACh);
   - error-driven local learning;
   - explaining away (EXPLAIN);
   - narrow active inference (discriminating queries/tests).
   - **Not copied:** "perception is only prediction"; predictions here are hypotheses checked against evidence. (`06`, `07` §2–3)

3. **Recognizing something never seen:** by structure, not appearance.
   - Parse it into parts, properties, functions and relations.
   - Two-stage matching against schemas.
   - Accept the concept whose defining structure matches and whose variability profile tolerates the differences.
   - If nothing fits, the residual hypothesis wins and the Concept Formation Engine runs. (`05` §6, §9)

4. **The wheel example:** `11` §6.3, with full detail in the final research report:
   - parse → DERIVED "wheel" via defining structure;
   - reject "car wheel" (size and rating out of profile);
   - causal hypotheses (aircraft / truck / residual) as PREDICTED;
   - Support → aircraft;
   - promoted only through a REMEMBERED part-of chain;
   - otherwise rendered as a prediction;
   - recurring cases lead to a new concept.

5. **Concept from very few examples:** the Concept Formation Engine — parse → align and abstract → parent → overhypothesis widening → size principle → inheritance as INHERITED → contrast → store NEW. (`05` §9)

6. **Generalizable properties stored:** part structure, functions/affordances, relations, causal roles, invariances, variability profiles, discriminative contrasts, context priors. Instance specifics stay episodic. (`04` §2.2)

7. **Similarity vs. causality:** separate link types with separate admission rules.
   - `causes` needs mechanism or intervention evidence, or invariance across contexts plus a test.
   - Co-occurrence only creates `associated_with`.
   - L3 predictions use only causal links. (`04` §6.1, `08` §5.2 step 6)

8. **Reused capabilities:** primitives (always); schemas and their variability priors; causal schemas; procedures (rebound); skills within their envelopes; analogous cases; articulators. Example: one `loop` schema serves code, math, plans and music. (`05`)

9. **Structural generalization:** relational graphs; schemas with typed variables learned by anti-unification; alignment-based instantiation; type-directed composition fills the gaps. (`05` §5–6)

10. **Analogy:** signature retrieval (cross-domain) → ALIGN with systematicity → candidate inferences as ANALOGICAL → tests → abstraction over both domains on success. (`05` §6)

11. **World vs. token prediction:** L0–L5 in Core/Library/Simulator produce typed records; L6 lives only in the Mouth, whose probabilities are never evidence. The taint check is enforced at the Mouth input. (`06` §1, `07` §5)

12. **Retracting a wrong prediction:** error evidence → Support drops → retraction → TMS cascade over dependents → nogood → generator calibration update → envelope shrinks. (`05` §3, `06` §3)

13. **UNKNOWN vs. low-confidence prediction:**
    - UNKNOWN-ABSENT / -DECLARED / -NO-BASIS = no basis.
    - PREDICTION-class / INSUFFICIENT = a weak or unpromoted basis.
    - Each has its own decision path and rendering. (`06` §5–6, `07` §5.4)

14. **Why it answered:** the Answer Record holds the justification subgraph, sources, executed circuit, rejected hypotheses, taint map and envelope flags. "Why?" renders this record. (`06` §7)

15. **Reasoning machinery in small models:** a machinery floor enforced by Build Configuration validation; scale changes knowledge, skills, `K/R/H`, budgets and fluency only. (`01` §8, `09` §3.2)

16. **Small vs. large:** same machinery; differences in knowledge breadth, number of skills (amortization), workspace/depth (problem size), analogy sources and fluency. (`01` §8)

17. **Total vs. active capacity:** a dormant, content-addressed Library and Skills; a resident Core; Heart caps; active-fraction accounting. (`01` §8, `07` §1, `09` §6)

18. **Why not MoE:**
    - units are domain-general primitives plus addressable knowledge, not fixed experts;
    - selection by structural matching per reasoning step, not a token-level gate;
    - explicit typed composition into a program;
    - specialization assembled per task, then compiled;
    - the library grows and changes;
    - per-node verification. (`05` §4–5)

19. **Creating a temporary circuit:** schema retrieval → analogical transfer → type-directed synthesis with policy, nogoods and budgets → execution with local repair. (`05` §5.2)

20. **After the task:** the graph is dissolved; the trace is stored as an EPISODE; failures are stored as nogoods; compilation candidates are queued. (`05` §5.4)

21. **Temporary circuit → persistent skill:** sleep selects frequent, successful, costly subgraphs → procedure schema → distilled skill accepted at ≥ 98% agreement and ≥ 2× speedup → new component with lineage → monitored; split/merge/promotion. (`08` §6)

22. **Package format:** `09` §5 (directory `srmpkg-1`: header, Manifest, interface, components, regions, derived, validation, optional traces/episodic/history).

23. **Build Configuration role:** pre-training intent and limits; validated; immutable; its hash is referenced by Manifests. (`09` §3)

24. **Trained Model Manifest role:** the generated, Merkle-hashed map of the realized model; the authority for loading, training, surgery, compatibility and audit. (`09` §4)

25. **Component ID system:** logical ID = hash of identity metadata (family, interface major, type, interface signature, structural signature, capacity class, dependency interface signature, lineage root, generation); version ID = weight / dependency / training-record hash. (`09` §2)

26. **ID stability:** IDs are independent of weights and file locations; semantic-versioning rules decide between a new version and a new ID; lineage edges and an alias table handle splits and merges. (`09` §2.3)

27. **Dependency graph:** typed, versioned edges in `manifest/graph.json`, inside the Merkle tree. (`09` §4.2)

28. **Retraining one component:** select → closure → stage → train with anchor → local tests → integration → gates → version → commit → monitor. (`10`)

29. **Minimum neighbouring dependencies:** the closure algorithm — upstream producers replaced by traces, objective dependencies loaded frozen, consumers used only for tests. (`10` §2.2)

30. **Without loading the rest:** sharded, memory-mapped package; traces replace upstream; integration tests page in only what they recruit; no gradients outside `T`. (`10` §2.3, §3)

31. **Integration testing:** coverage-map-selected system tests, regression and safety subsets, A/B, shadow run. (`10` §5)

32. **Rollback:** immutable versions; an atomic pointer change in a new Manifest root; automatic triggers from monitors. (`10` §7)

33. **Third-party skills:** develop against the published interface and contracts; package with suite, fingerprint and signature; staged compatibility and validation; enters as `third_party` with SUGGESTED outputs until its envelope accrues. (`10` §8)

34. **Unsafe without the Manifest:** `10` §11.

35. **Old manifest vs. new model compatibility:** compare schema, package format, interface MAJOR/MINOR and ABI hash, primitive set, interface signatures, version ranges → compatible / migratable / incompatible. (`10` §9.1)

36. **Safe surgery:** a staging manifest (copy-on-write), operations on staged copies, structural invariants, validation, atomic root swap, previous root retained. (`10` §6)

37. **Guaranteed independently trainable:** `independent` class — skills, articulators, regions, modality parsers above the encoder, tool adapters, primitives with local objectives. "Guaranteed" means the mechanism and gates exist; each update must still pass validation. (`10` §10)

38. **Not safely separable:** `frozen` (interface layer), `derived` (indices, sketch); `with_closure` components (Composer, Selector, Gate, Heart estimators, verifier, renderer base) are trainable but need full integration. (`10` §10)

39. **Recording specialization:** Manifest emergence fields — created_by, source trace families, parents/lineage, usage profile, activation cluster, functional label, envelope summary. (`09` §4.2)

40. **Difference from adapters, LoRA and MoE:**
    - adapters and LoRA nudge a monolith without contracts, epistemics or addressable knowledge;
    - MoE: see Q18;
    - Final SRM's modularity is part of the computation itself: a frozen typed interface, addressable records, assembled-then-compiled specialization, identity/contract/lineage/trust.
    - Low-rank modulation is only how skills are *implemented*. (`05`, `09`)

41. **1B Final SRM vs. a conventional 1B model (hypotheses):**
    - **Better:** calibrated abstention and unsupported-assertion rate, few-shot concept/rule learning, compositional generalization, continual updates, verified program synthesis, component-level updates.
    - **Worse or equal:** open-ended fluency, implicit stylistic knowledge, simple-chat latency.
    - Both configurations are tested: 1B total and 1B active. (`12` §3 H7)

42. **Where a 1B SRM might compete with much larger models:** machinery-bound tasks — compositional splits, ARC-style abstraction, novel-object reasoning, unseen-API programming with tests, verifiable math, generated planning, knowledge editing, calibrated QA, continual streams. (`12` §2)

43. **Benchmark:** the Machinery-vs-Scale suite (`12` §2). The hypothesis is supported if SRM-1B ≥ dense 7B+ on at least 4 of 9 categories at matched inference compute (threshold fixed before running).

44. **Parameter efficiency:** score per total and active parameter; reuse factor; machinery fraction; Library-vs-Core ablation. (`12` §4.1)

45. **Sample efficiency:** learning curves, examples-to-X%, AULC by type, against baselines at matched compute; total bootstrap data and compute. (`12` §4.2)

46. **Active-compute efficiency:** FLOPs, bytes, wall-clock, joules per query and per correct answer; active fraction; exponent α over the library sweep. (`12` §4.3)

47. **Modular retraining cost:** peak GPU memory, GPU-hours, bytes loaded, components touched, integration time, collateral vs. target gain, against full fine-tuning and LoRA; gate catch and rollback rates. (`12` §4.4)

48. **Forgetting:** backward/forward transfer, edit locality and ripple accuracy, drift across sleep cycles, fingerprint stability. (`12` §4.5)

49. **Unsuccessful if:** `12` §5.2 (nine criteria, including "RAG + adapters + agents matches it").

50. **Complete system diagram:** Diagram A (`01` §9), with Diagrams B (`06` §9), C (`09` §7) and D (`10` §13).
