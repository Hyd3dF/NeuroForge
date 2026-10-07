"""Build Configuration (09 §3).

Written before training, validated before any build, immutable afterwards.  Every
size, capacity, threshold and budget used anywhere in ``srm`` is read from here
(14 §3 rule 1).  Defaults are the SRM-F0 profile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from srm.util.canonical import canonical_hash

# Component types that must exist at every scale (the machinery floor, 01 §8).
MACHINERY_FLOOR_TYPES: tuple[str, ...] = (
    "interface.codec",
    "interface.codespace",
    "interface.property_basis",
    "core.composer",
    "core.selector",
    "core.error_monitor.verifier",
    "core.concept_formation",
    "core.simulator",
    "core.skill_core",
    "control.heart_policy",
    "control.gate",
    "control.modulators",
    "mouth.planner",
    "mouth.renderer_base",
)


class _Section(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MetaConfig(_Section):
    model_family_id: str = "neuroforge-srm"
    model_name: str = "srm-f0"
    model_generation: int = 1
    build_config_version: str = "1.0"
    target_hardware_profile: str = "single_gpu_24gb"
    seed: int = 1234


class CapacityConfig(_Section):
    total_capacity_budget: dict[str, int] = Field(
        default_factory=lambda: {
            "interface": 50_000_000,
            "core": 200_000_000,
            "library_records": 10_000_000,
            "skills": 100_000_000,
            "mouth": 50_000_000,
        }
    )
    active_capacity_ceiling: int = 300_000_000
    max_active_fraction: float = 0.05
    machinery_floor: tuple[str, ...] = MACHINERY_FLOOR_TYPES


class InterfaceConfig(_Section):
    B: int = 64
    L: int = 64
    r: int = 3
    n_b: int = 21
    d: int = 256
    d_k: int = 128
    P_0: int = 512
    N_role: int = 64
    scalar_buckets: int = 32
    version: str = "1.0"


class CodecConfig(_Section):
    input_normalization: str = "NFC"
    output_vocab_size: int = 8192
    special_tokens: tuple[str, ...] = (
        "<bos>", "<eos>", "<pad>", "<cite>", "<hedge>", "<abstain>", "<code>", "</code>",
    )


class PerceptionConfig(_Section):
    chunk_max_bytes: int = 128
    n_layers_perc: int = 4
    recurrent_state_dim: int = 256
    theta_known: float = 0.80
    theta_cos: float = 0.90
    theta_err: float = 0.30
    theta_parse: float = 0.50


class MemoryConfig(_Section):
    N_lib: int = 1_000_000
    N_hip: int = 100_000
    page_records: int = 4096
    tiers_enabled: tuple[str, ...] = ("T1", "T2")
    S_context: int = 4
    cand_max: int = 4096
    k_ret: int = 64
    index_backend: str = "banded_sparse"
    alpha_score: float = 0.5
    hopfield_beta: float = 8.0
    t_clean: int = 3
    sketch_fpr: float = 1e-3
    sketch_expected_keys: int = 1_000_000
    h_spread: int = 2
    k_spread: int = 256
    spread_gamma: float = 0.5
    eta_H: float = 0.05
    eta_decay: float = 0.001
    w_max: float = 1.0
    edge_gains: dict[str, float] = Field(
        default_factory=lambda: {
            "is_a": 0.8, "part_of": 0.7, "has_property": 0.6, "causes": 0.6, "prevents": 0.4,
            "enables": 0.5, "associated_with": 0.5, "analogous_to": 0.5, "used_with": 0.6,
            "contradicts": 0.0, "derived_from": 0.0, "trained_from": 0.0, "co_activated": 0.4,
        }
    )
    theta_link: float = 0.85
    theta_rel: float = 0.80
    n_use: int = 2
    n_sleep_stable: int = 3
    n_stable_use: int = 5
    theta_decay: float = 0.05
    T_decay: int = 5
    region_max_records: int = 1_000_000
    theta_new_region: float = 0.2
    lambda_cat: float = 0.3
    plasticity_by_stage: dict[str, float] = Field(
        default_factory=lambda: {
            "NEW": 1.0, "CORROBORATED": 0.7, "USED": 0.5, "CONSOLIDATED": 0.2,
            "STABLE": 0.05, "CONTESTED": 0.7, "DEPRECATED": 0.0, "DECAYED": 0.0,
        }
    )


class WorkspaceConfig(_Section):
    K: int = 64
    R: int = 32
    H: int = 8
    A: int = 3
    G_max: int = 8
    HS_max: int = 16
    NG_max: int = 1024


class EpistemicsConfig(_Section):
    # D-021: the evidence unit and prior weight are calibrated so that one fully
    # trusted source reaches θ_commit (W=2 with κ=1 made θ_commit unreachable).
    W: float = 0.15
    kappa: float = 1.0
    kappa_v: float = 1.0
    kappa_t: float = 2.0
    e_max: float = 50.0
    theta_commit: float = 0.80
    theta_retract: float = 0.60
    theta_abstain: float = 0.70
    theta_max: float = 0.95
    theta_predict: float = 0.50
    theta_dormant: float = 0.20
    delta_margin: float = 1.0
    pi_cap: float = 1.0
    r_extrap: float = 0.3
    t_norm: str = "min"
    trust_priors: dict[str, float] = Field(
        default_factory=lambda: {
            "generator_truth": 1.00, "task_given": 1.00, "tool_execution": 0.99, "curated_kb": 0.90,
            "reference_doc": 0.85, "textbook": 0.85, "api_docs": 0.85,
            "code_repository": 0.75, "user_statement": 0.70, "web_document": 0.50,
            "forum": 0.40, "third_party_component": 0.30, "model_output": 0.20,
        }
    )
    modality_factors: dict[str, float] = Field(
        default_factory=lambda: {
            "asserted": 1.0, "hedged": 1.0, "reported": 0.0, "hypothetical": 0.0,
            "conditional": 1.0, "negated": 1.0, "unknown_declared": 0.0, "fictional": 0.0,
        }
    )
    hedge_slope: float = 0.7
    theta_trust_gap: float = 0.3
    theta_conflict_mass: float = 0.5
    theta_noise: float = 0.3
    remembered_single_source_min_trust: float = 0.85
    lambda_c: float = 2.0
    lambda_v: float = 0.5
    lambda_mdl: float = 0.1
    lambda_p: float = 1.0
    llr_eps: float = 1e-4
    allow_verifier_promotion_low_stakes: bool = False
    default_stakes: float = 0.5


class CoreConfig(_Section):
    D_max: int = 8
    E_max: int = 512
    k_align: int = 8
    t_align: int = 5
    theta_schema: float = 0.6
    theta_analogy: float = 0.6
    theta_parent: float = 0.5
    theta_pair: float = 0.3
    lambda_s: float = 0.5
    verifier_width: int = 256
    verifier_layers: int = 3
    composer_width: int = 256
    composer_layers: int = 3
    lambda_cost: float = 0.01
    lambda_depth: float = 0.1
    z_match: float = 2.5
    lambda_miss: float = 0.5
    synth_max_size: int = 7
    synth_evals_per_beat: int = 20_000
    synth_max_evals: int = 400_000
    simulate_probes: int = 24
    procedure_min_cues: int = 2


class PredictionConfig(_Section):
    k_env: int = 16
    n_min: int = 20
    theta_env: float = 0.8
    lambda_res: float = 1.0
    lambda_res0: float = 1.0


class ControlConfig(_Section):
    T_max: int = 64
    base_budget: float = 1e9
    max_budget_per_query: float = 1e11
    max_flops_per_beat: float = 5e9
    max_bytes_per_beat: float = 2e8
    max_active_records: int = 4096
    floor_subconscious: float = 0.10
    floor_error_monitor: float = 0.15
    eta_c: float = 0.1
    c_min: float = 0.1
    c_max: float = 2.0
    stakes_gain: float = 1.0
    ne_gain: float = 0.5
    op_flops: dict[str, float] = Field(
        default_factory=lambda: {
            "LINK": 1e5, "JOIN_HOP": 5e5, "ANSWER": 1e5, "SYNTH_DSL_EVAL": 2e3, "TEST": 1e5,
            "SIMULATE": 5e5, "MATCH_PROCEDURE": 5e5, "BIND_QUANTITIES": 1e5, "EVAL_FORMULA": 2e5,
            "CHECK": 2e5, "ALGEBRA": 5e5, "PY_CHECK": 1e6, "GATE": 5e4, "SUBCONSCIOUS": 2e5,
        }
    )
    type_prior: dict[str, float] = Field(
        default_factory=lambda: {
            "test": 1.0, "answer": 0.95, "retrieve": 0.9, "transform": 0.85, "synthesis": 0.7,
            "simulate": 0.6, "background": 0.3,
        }
    )
    prices: dict[str, float] = Field(
        default_factory=lambda: {
            "flops": 1e-9, "bytes": 1e-8, "slots": 0.01, "verifier_calls": 0.05,
            "tool_calls": 0.2, "tokens": 0.001, "wall_ms": 0.001,
        }
    )


class SkillsConfig(_Section):
    d_s: int = 256
    n_layers_skill: int = 4
    r_skill: int = 8
    theta_accept: float = 0.98
    speedup_min: float = 2.0
    n_compile: int = 5
    c_compile: float = 1.0
    n_domains_promote: int = 3
    theta_recompile: float = 0.9


class MouthConfig(_Section):
    n_layers_mouth: int = 4
    d_mouth: int = 256
    V_out: int = 8192
    theta_attr: float = 1.5
    articulators: tuple[str, ...] = ("prose", "markdown", "json", "python", "dsl", "latex")


class BodyConfig(_Section):
    tools: tuple[str, ...] = ("dsl", "python_sandbox", "tests", "sympy", "units")
    sandbox_timeout_s: float = 2.0
    sandbox_memory_mb: int = 256
    tool_costs: dict[str, float] = Field(
        default_factory=lambda: {"dsl": 0.01, "python_sandbox": 0.5, "tests": 0.5, "sympy": 0.1, "units": 0.01}
    )


class BootstrapConfig(_Section):
    stages: tuple[str, ...] = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7")
    data: dict[str, int] = Field(
        default_factory=lambda: {
            "objectworld_concepts": 300,
            "objectworld_heldout_concepts": 50,
            "objectworld_instances": 1_000_000,
            "objectworld_texts": 500_000,
            "dsl_tasks": 200_000,
            "factstream_entities": 100_000,
            "factstream_facts": 1_000_000,
            "factstream_questions": 50_000,
            "math_problems": 100_000,
            "expression_pairs": 300_000,
        }
    )
    acceptance: dict[str, float] = Field(
        default_factory=lambda: {
            "s1_spearman_ov_cos": 0.8,
            "s1_block_entropy_frac": 0.9,
            "s1_banded_recall_at_64": 0.95,
            "s2_parser_fact_f1": 0.9,
            "s3_verifier_auroc": 0.9,
            "s4_heldout_composition_success": 0.5,
            "s5_attribution_precision": 0.95,
            "s7_ece": 0.05,
            "s7_unknown_auroc": 0.9,
        }
    )


class OnlineLearningConfig(_Section):
    eta_local: float = 1e-4
    n_local_steps: int = 8
    lambda_anchor: float = 1.0
    buffer_size: int = 512


class LearningConfig(_Section):
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)
    online: OnlineLearningConfig = Field(default_factory=OnlineLearningConfig)


class SleepConfig(_Section):
    t_idle_s: float = 30.0
    micro_sleep_budget: float = 1e9
    theta_hip_fill: float = 0.8
    sleep_every_queries: int = 10_000
    replay_mix: float = 1.0
    theta_mdl: float = 8.0
    downscale: float = 0.95
    w_min: float = 0.01
    theta_merge: float = 0.95
    n_ctx_causal: int = 3


class ComponentsConfig(_Section):
    fingerprint_probe_size: int = 256
    eps_reg: float = 0.005
    delta_min: float = 0.0
    eps_shadow: float = 0.05
    d_patch: float = 0.01
    d_minor: float = 0.10


class PackagingConfig(_Section):
    package_format: str = "srmpkg-1"
    shard_target_mb: int = 256
    compression: str = "zstd"
    signing_key_id: str | None = None


class RuntimeConfig(_Section):
    batch_size: int = 16
    precision: str = "bf16"
    device: str = "cpu"
    log_level: str = "INFO"
    trace_recording: str = "off"


class EvaluationConfig(_Section):
    suites: tuple[str, ...] = (
        "compositional", "fewshot", "novel_concept", "unseen_api", "robust_math",
        "planning", "knowledge_updates", "calibrated_qa", "continual",
    )
    baselines: tuple[str, ...] = (
        "dense_transformer_active", "dense_transformer_total", "transformer_rag",
        "transformer_rag_adapters_agent", "continual_finetune", "memory_layer", "trm",
    )
    seeds: tuple[int, ...] = (1, 2, 3)


class BuildConfig(_Section):
    meta: MetaConfig = Field(default_factory=MetaConfig)
    capacity: CapacityConfig = Field(default_factory=CapacityConfig)
    interface: InterfaceConfig = Field(default_factory=InterfaceConfig)
    codec: CodecConfig = Field(default_factory=CodecConfig)
    perception: PerceptionConfig = Field(default_factory=PerceptionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    epistemics: EpistemicsConfig = Field(default_factory=EpistemicsConfig)
    core: CoreConfig = Field(default_factory=CoreConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    mouth: MouthConfig = Field(default_factory=MouthConfig)
    body: BodyConfig = Field(default_factory=BodyConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    sleep: SleepConfig = Field(default_factory=SleepConfig)
    components: ComponentsConfig = Field(default_factory=ComponentsConfig)
    packaging: PackagingConfig = Field(default_factory=PackagingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @model_validator(mode="after")
    def _validate_rules(self) -> "BuildConfig":  # 09 §3.2
        errors: list[str] = []
        i, w, e, c, cap = self.interface, self.workspace, self.epistemics, self.control, self.capacity
        if i.r * i.n_b > i.B:
            errors.append(f"interface: r*n_b ({i.r * i.n_b}) must be <= B ({i.B})")
        if not 2 <= i.L <= 256:
            errors.append("interface: L must be in [2, 256] (one byte per block)")
        if i.r < 1 or i.n_b < 1:
            errors.append("interface: r and n_b must be >= 1")
        missing = [t for t in MACHINERY_FLOOR_TYPES if t not in cap.machinery_floor]
        if missing:
            errors.append(f"capacity.machinery_floor is missing required types: {missing}")
        if w.K < 2 * w.A + w.G_max:
            errors.append(f"workspace: K ({w.K}) must be >= 2*A + G_max ({2 * w.A + w.G_max})")
        if c.floor_subconscious + c.floor_error_monitor >= 1.0:
            errors.append("control: the sum of floors must be < 1")
        for name in (
            "theta_commit", "theta_retract", "theta_abstain", "theta_max", "theta_predict",
            "theta_dormant", "theta_trust_gap", "theta_noise", "r_extrap",
            "remembered_single_source_min_trust", "default_stakes",
        ):
            v = getattr(e, name)
            if not 0.0 <= v <= 1.0:
                errors.append(f"epistemics.{name} must be in [0, 1], got {v}")
        if not e.theta_predict <= e.theta_abstain <= e.theta_max:
            errors.append("epistemics: require theta_predict <= theta_abstain <= theta_max")
        if e.W <= 0 or e.kappa <= 0:
            errors.append("epistemics: W and kappa must be > 0")
        if e.t_norm not in ("min", "product"):
            errors.append("epistemics.t_norm must be 'min' or 'product'")
        resident = cap.total_capacity_budget.get("core", 0) + cap.total_capacity_budget.get("mouth", 0)
        if cap.active_capacity_ceiling < resident:
            errors.append(
                f"capacity.active_capacity_ceiling ({cap.active_capacity_ceiling}) must be >= resident "
                f"core+mouth estimate ({resident})"
            )
        if not 0 < cap.max_active_fraction <= 1:
            errors.append("capacity.max_active_fraction must be in (0, 1]")
        if self.memory.index_backend not in ("banded_sparse", "dense_ann"):
            errors.append("memory.index_backend must be 'banded_sparse' or 'dense_ann'")
        if errors:
            raise ValueError("invalid Build Configuration:\n  - " + "\n  - ".join(errors))
        return self

    # --- serialization ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON; referenced by every Manifest (09 §4.4)."""
        return canonical_hash(self.to_dict())

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BuildConfig":
        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_overrides(self, overrides: dict[str, Any]) -> "BuildConfig":
        """Return a new validated config with nested ``{"section": {"field": value}}`` overrides."""
        data = self.to_dict()
        for section, fields in overrides.items():
            if isinstance(fields, dict) and isinstance(data.get(section), dict):
                _deep_update(data[section], fields)
            else:
                data[section] = fields
        return BuildConfig.model_validate(data)


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
