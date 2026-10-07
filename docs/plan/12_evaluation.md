# 12 — Evaluation, Falsification, Risks, Novelty

## 1. SRM-F0 (smallest decisive prototype)

**Domains:** ObjectWorld, DSLWorld (+ Python subset), FactStream, MathWorld-lite (`02` §14).

**Components:** full machinery floor at F0 sizes.
- Interface layer: ~100–512 properties.
- About 20 primitives.
- Composer, Selector, Error Monitor, Concept Formation Engine, Simulator.
- Knowledge Sketch.
- Library swept from 1e4 to 1e7 records.
- Mouth with template renderer, a small neural renderer and 3+ articulators.
- Heart, Gate, Modulators, Subconscious.
- Sleep.
- Manifest and surgery pipeline.

**Baselines** (same data, matched where stated):
1. **Dense Transformer LM**, matched active FLOPs per query.
2. Dense Transformer LM, matched **total** parameters.
3. **Transformer + RAG** over the *same* knowledge content (this is the critical baseline).
4. Transformer + RAG + adapters + an agent loop with tool access (the "RAG + LoRA + agents" control).
5. Continually fine-tuned Transformer (for forgetting tests).
6. PEER/memory-layer model (for capacity vs. active compute).
7. A Tiny Recursive Model-style recursive reasoner (for puzzle/composition tasks).

## 2. Benchmark suite ("Machinery-vs-Scale")

All splits are procedurally generated with contamination controls, plus public benchmarks where noted.

| # | Category | Content | Primary metric |
|---|---|---|---|
| 1 | Compositional generalization | SCAN/COGS/CFQ-style splits + new generated grammars + DSL held-out compositions | Exact-match accuracy |
| 2 | Few-shot abstraction | ARC-AGI-1/2 (public), ConceptARC, ObjectWorld held-out concepts at 1/2/4/8 shots | Accuracy vs. shots |
| 3 | Novel concept composition | ObjectWorld held-out property/part combinations; wheel-style out-of-range instances | Classification + function/parent inference accuracy |
| 4 | Unseen-API programming | New DSL/library documented only in context; tasks checked by tests | Pass@1 with tests |
| 5 | Robust math | MathWorld-lite perturbations (GSM-Symbolic style) | Accuracy; drop under perturbation |
| 6 | Planning | Generated domains with verifiable plans | Success rate; plan cost |
| 7 | Knowledge updates | FactStream edits with ripple effects (RippleEdits-style) | Edit success, locality, ripple accuracy |
| 8 | Calibrated QA | FactStream + withheld entities + declared unknowns + contested facts | Selective accuracy, coverage–risk AUC, ECE, UNKNOWN AUROC, taint-leak rate |
| 9 | Continual stream | Facts and skills arriving over time | Backward/forward transfer, retention |

## 3. Hypotheses and how each is tested

| Hypothesis | Test |
|---|---|
| H1: Few-shot concept learning beats in-context and fine-tuning baselines | Category 2–3 learning curves at equal compute |
| H2: Compositional/structural generalization beats size-matched baselines | Category 1, 3, 4 |
| H3: Active cost grows sublinearly with library size | Library sweep 1e4 → 1e7 (≥ 100× growth): FLOPs, bytes, wall-clock, joules per query |
| H4: The knowledge/prediction separation reduces unsupported answers | Category 8: taint-leak rate, selective accuracy, UNKNOWN AUROC vs. token entropy, verbalized confidence, semantic entropy, RAG score |
| H5: Modular retraining is cheap and local | `10` pipeline on articulator, skill, verifier: cost and collateral metrics (§4.4) |
| H6: Intelligence lives in the machinery | Library-vs-Core ablation: remove 90% of the Library vs. degrade the Core (fewer primitives, no Composer search); compare drops on categories 1–6 |
| H7: Small SRM competes with larger conventional models on machinery-bound tasks | SRM-1B-total and SRM-1B-active vs. dense 1B/7B/70B on categories 1–9 at matched inference compute (post-F0 scale-up) |

## 4. Metric definitions

### 4.1 Parameter efficiency
- Score per total parameter and per active parameter.
- **Reuse factor** = mean number of distinct task families that recruit each component (from usage logs).
- **Machinery fraction** = Core size / total size.

### 4.2 Sample efficiency
- Accuracy vs. number of examples (1, 2, 4, 8, 16, …).
- **Examples-to-X%** and area under the learning curve.
- Computed per type (concept, rule, procedure, API) at matched compute, against in-context learning, fine-tuning and RAG baselines.
- Plus the total bootstrap data and compute used to reach baseline-equivalent capability.

### 4.3 Active-compute efficiency
- FLOPs, **bytes moved**, wall-clock and joules per query **and per correct answer**.
- Active fraction (`09` §6).
- **Exponent** α in `active_cost ∝ N_lib^α`, fitted over the sweep.
- Prefetch hit rate; habit hit rate.

### 4.4 Modular retraining cost
- Peak GPU memory, GPU-hours, bytes loaded, number of components touched.
- Time to integrate (including validation).
- **Collateral change** (regression on untouched suites) and **target gain**.
- Comparison with full fine-tuning and LoRA at equal target gain.
- Gate catch rate; rollback success rate.

### 4.5 Forgetting
- Retention on all prior suites after k updates (backward transfer); forward transfer.
- Edit locality and ripple accuracy.
- Old-knowledge accuracy across sleep cycles (drift).
- Fingerprint stability of untouched components.

### 4.6 Epistemic quality
- **Taint-leak rate:** answers rendered as KNOWN-class whose supporting claim was tainted or unsupported. Must be 0 by construction; tested as a regression invariant.
- **Unsupported-assertion rate:** human/automatic check of factual statements against ground truth.
- ECE; coverage–risk; UNKNOWN AUROC.

### 4.7 F0 safety suite (used by retraining gates, `10` §5.3)
- **Epistemic invariants:** taint-leak = 0; fact-placement probes (skills asked factual questions must yield SUGGESTED only); abstention on withheld entities.
- **Sandbox isolation:** generated code that attempts I/O, imports, network or resource exhaustion is blocked and recorded.
- **Privacy scope:** private records never surface in another user's session or in shared regions.
- **Attribution:** rendered content spans map to plan items (no unattributed content).
- **Contradiction handling:** CONTESTED facts are never rendered as plain assertions.

## 5. Continue / stop criteria

### 5.1 Minimum results to continue after SRM-F0
These thresholds are fixed before running and may be tightened only with a decision-log entry.
1. Over ≥ 100× library growth, active bytes per query grow < 2× and accuracy does not fall.
2. One-shot facts are retained ≥ 90% after ≥ 1e4 subsequent writes, with less collateral change than fine-tuning.
3. UNKNOWN AUROC beats token-entropy and RAG-score baselines.
4. Multi-hop and DSL accuracy ≥ the Transformer with matched active FLOPs.
5. SRM beats Transformer + RAG on at least two of: post-sleep abstraction transfer, habit amortization, correction propagation, UNKNOWN detection, few-shot concept learning.

### 5.2 Failure criteria (declare the architecture unsuccessful if any persist after reasonable iteration)
1. **Few-shot:** no meaningful advantage in examples needed over matched-compute in-context learning or fine-tuning on new concepts and rules.
2. **Generalization:** below a size-matched conventional model on compositional or novel-concept splits.
3. **Capacity:** active cost scales roughly linearly with library size, or wall-clock/energy isn't lower despite lower FLOPs.
4. **Epistemics:** tainted predictions leak into factual answers at baseline-like rates, or UNKNOWN detection is no better than token-entropy baselines.
5. **Modularity:** single-component retraining causes collateral regressions comparable to full fine-tuning, or closures routinely grow to most of the model.
6. **Machinery:** Library ablation hurts reasoning categories as much as Core ablation.
7. **Integration:** a well-tuned Transformer + RAG + adapters + agent stack matches Final SRM on criteria 1–5.
8. **Recruitment:** partial-cue recall degrades steeply with library size and larger codes don't fix it.
9. **Consolidation:** old-knowledge accuracy declines steadily across sleep cycles.

## 6. Risk register

| # | Risk | Type | Mitigation | Detection |
|---|---|---|---|---|
| R1 | A factored Property Basis is hard to learn (disentanglement) | Scientific (largest) | Synthetic grounding, weak supervision, extensible basis | S1 acceptance metrics; category 3 |
| R2 | Structural parsing is weak outside formal domains | Scientific | Formal domains first; teacher extraction for NL; opaque-segment fallback | Parser F1 by domain |
| R3 | Composer search explosion | Scientific/engineering | Type-directed pruning, schemas, analogy, nogoods, budgets | Expansions per solved task |
| R4 | Frozen interface limits later improvement | Engineering | Planned MAJOR migrations with adapters | Migration cost tracking |
| R5 | Interface traces don't cover the deployment distribution | Engineering | Coverage tracking; widen-closure path; shadow mode | Gate reachability failures |
| R6 | Manifest and tooling complexity | Engineering | Automated generation; schema validation; tests | Tooling defects |
| R7 | Bootstrap depends on teacher models | Scientific | Synthetic worlds first; teacher use recorded in training records | Ablate teacher data |
| R8 | GPU memory-gather overhead hides gains | Engineering | Batching per beat, page layout, prefetch; measure bytes and wall-clock | H3 metrics |
| R9 | Fact leakage into skills/Core weakens UNKNOWN guarantees | Scientific | Fact-dropout; SUGGESTED tagging; probes | Leakage probes (§4.6) |
| R10 | Local learning plateaus below global training | Scientific | Periodic with-closure retraining; sleep-time distillation | Learning curves |
| R11 | Consolidation drift | Scientific | Interleaved replay; anchoring; region regression checks | Drift metric (§4.5) |
| R12 | Complexity slows iteration | Process | Vertical-slice roadmap (`14`) | Milestone tracking |

## 7. Novelty and precedents

### 7.1 Precedents (acknowledged)
- Truth maintenance (Doyle JTMS 1979; de Kleer ATMS 1986); blackboard / global workspace (Hearsay-II, Baars, Dehaene); Neural Production Systems; slot-based models.
- Complementary learning systems; hippocampal indexing theory; modern Hopfield networks; sparse distributed memory; FlyHash expand-and-sparsify; SLIDE; product-key memories; PEER; DeepSeek Engram.
- Active dendrites; Thousand Brains / Monty; BDH (Dragon Hatchling).
- Doya's neuromodulation-as-metalearning; agoric computing; rational metareasoning (value of computation).
- Vector-symbolic architectures / sparse block codes; structure-mapping theory and MAC/FAC; Bayesian Program Learning; Schema Networks; recognition-by-components; Bayesian concept learning and overhypotheses; anti-unification; DreamCoder; meta-learning for compositionality (Lake & Baroni 2023); Neural Module Networks; invariant causal prediction.
- Universal Transformer / ACT / PonderNet / Huginn / HRM / TRM; Coconut; Large Concept Models.
- Verifiers and PRMs; semantic entropy; evidential deep learning (and its critiques); subjective logic.
- Taint tracking (information-flow security); Git-Theta / MGit model versioning; Merkle DAG content addressing; design-by-contract; semantic versioning.

### 7.2 Hypotheses we did not find combined elsewhere
1. Epistemic taint tracking + validity envelopes for every generator.
2. Generator-as-proposal with Support-based selection and an explicit residual "novel" hypothesis.
3. A frozen interface code space enabling component-level retraining on recorded interface traces.
4. Component-level trust for third-party skills inside an evidence algebra.
5. Variability-profile (overhypothesis) priors driving few-shot concept formation inside a sparse recruitment system.
6. Per-task circuits compiled into versioned components with lineage in a manifest.
7. A machinery floor across scales.
8. Recognition-by-recall input gating.
9. The Heart compute economy as the single allocator.
10. An exact UNKNOWN-ABSENT state from the Knowledge Sketch combined with the fact-placement rule.

## 8. References (from the research phases)
- Neural Production Systems — https://arxiv.org/pdf/2103.01937v2
- Tiny Recursive Models — https://arxiv.org/html/2510.04871v1
- DeepSeek Engram — https://introl.com/fr/blog/deepseek-engram-conditional-memory-architecture-january-2026
- Callaway et al. 2018, Learning to Select Computations — https://is.mpg.de/publications/callaway2018learning
- Is Epistemic Uncertainty Faithfully Represented by Evidential DL? — https://mlanthology.org/icml/2024/juergens2024icml-epistemic
- Propositional Probes — https://arxiv.org/html/2406.19501v2
- Causal Concept Graphs — https://arxiv.org/html/2603.10377v1
- Partially Correlated Verifier Cascades — https://arxiv.org/pdf/2607.13918
- DeCoste 1991, CATMS label explosions — https://www.qrg.northwestern.edu/papers/Files/QRG_Dist_Files/QRG_1991/2_124_DeCoste_1991_CATMS_Label_Explosions.pdf
- Levy & Calvert, cortical computation energy — https://www.biorxiv.org/content/10.1101/2020.04.23.057927.full.pdf
- Lennie 2003, Cost of Cortical Computation — https://www2.bcs.rochester.edu/sites/plennie/pdfs/Lennie03a.pdf
- BDH (Dragon Hatchling) — https://arxiv.org/abs/2509.26507
- Nested Learning / Hope — https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning
- HiCL — https://arxiv.org/html/2508.16651v3
- Active Dendrites — https://ar5iv.labs.arxiv.org/html/2201.00042
- Doya, Metalearning and neuromodulation — https://en.wikipedia.org/wiki/Metalearning_(neuroscience)
- DreamCoder — https://simons.berkeley.edu/talks/dreamcoder-bootstrapping-inductive-program-synthesis-wake-sleep-library-learning
- SLIDE — https://proceedings.mlsys.org/paper_files/paper/2020/file/ca3480d82599b9b9b7040655483825c1-Paper.pdf
- FlyHash (Dasgupta et al. 2017) — https://www.its.caltech.edu/~jkenny/nb250c/papers/Dasgupta-2017.pdf
- Thousand Brains Project — https://arxiv.org/html/2412.18354v1
- Agoric Open Systems Papers — https://papers.agoric.com/papers/
- Language Models Need Sleep — https://arxiv.org/html/2606.03979v1
- SCM: Sleep-Consolidated Memory — https://arxiv.org/html/2604.20943v1
- HDC/VSA survey — https://arxiv.org/pdf/2111.06077
- Latent Relation Mapping Engine (structure mapping) — https://jair.org/index.php/jair/article/view/10583
- Lake et al. 2015, BPL — https://cims.nyu.edu/~brenden/papers/LakeEtAl2015Science.pdf
- Lake & Baroni 2023, MLC — https://pmc.ncbi.nlm.nih.gov/articles/PMC10620072
- Schema Networks — https://arxiv.org/pdf/1706.04317
- How modular should NMNs be — https://mlanthology.org/neurips/2021/damario2021neurips-modular
- ARC Prize 2025 Technical Report — https://arxiv.org/pdf/2601.10904
- Git-Theta — https://arxiv.org/pdf/2306.04529
- Merkle DAGs — https://docs-ipfs-tech.ipns.ipfs.hypha.coop/concepts/merkle-dag/
