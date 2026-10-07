"""SRM Experience Format (SEF) record models and validation (02 §3–4).

Every record carries the common header (02 §3.1).  Records are validated with
pydantic; :func:`parse_record` dispatches on ``kind``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from srm.util.canonical import canonical_hash

SEF_VERSION = "1.0"
SEF_MAJOR = 1

RECORD_KINDS = (
    "source", "segment", "entity", "property_def", "relation_def", "fact", "concept_definition",
    "concept_example", "procedure", "causal", "contradiction", "known_unknown", "skill_demo",
    "task", "question", "expression_pair",
)

SOURCE_TYPES = (
    "generator_truth", "tool_execution", "curated_kb", "reference_doc", "textbook", "code_repository",
    "api_docs", "web_document", "forum", "user_statement", "model_output", "third_party_component",
)

MODALITIES = (
    "asserted", "hedged", "reported", "hypothetical", "conditional", "negated", "unknown_declared", "fictional",
)

GOLD_STATES = (
    "KNOWN", "PREDICTION", "UNKNOWN-ABSENT", "UNKNOWN-NO-BASIS", "UNKNOWN-DECLARED", "CONTESTED", "INSUFFICIENT",
)


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- shared sub-structures -------------------------------------------------------------------
class Span(_M):
    segment_id: str
    start: int = 0
    end: int = 0


class Extraction(_M):
    method: Literal["deterministic", "teacher", "human", "generator"]
    method_version: str = ""
    confidence: float = Field(1.0, ge=0.0, le=1.0)


class Epistemic(_M):
    modality: Literal[
        "asserted", "hedged", "reported", "hypothetical", "conditional", "negated", "unknown_declared", "fictional"
    ] = "asserted"
    polarity: Literal["positive", "negative"] = "positive"
    hedge: float = Field(0.0, ge=0.0, le=1.0)
    attribution: str | None = None
    declared_confidence: float | None = Field(None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _consistent(self) -> "Epistemic":
        if self.modality == "reported" and not self.attribution:
            raise ValueError("reported modality requires an attribution")
        if self.modality == "hedged" and self.hedge <= 0.0:
            raise ValueError("hedged modality requires hedge > 0")
        return self


class Context(_M):
    time: tuple[float, float] | None = None
    version: str | None = None
    domain: str | None = None
    locale: str | None = None
    world_id: str | None = None

    @field_validator("time")
    @classmethod
    def _interval(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None and not v[0] < v[1]:
            raise ValueError("time interval must satisfy start < end")
        return v


class Link(_M):
    rel: Literal["supports", "contradicts", "example_of", "part_of", "derived_from", "answers"]
    target: str


class TypedValue(_M):
    type: Literal["int", "float", "bool", "enum", "text", "date", "range", "entity_ref", "code_ref", "expr_ref"]
    value: Any
    unit: str | None = None
    tolerance: float | None = Field(None, ge=0.0)


class PatternAtom(_M):
    subject: str
    relation: str
    object: str | TypedValue


class ConstraintAtom(_M):
    op: Literal["<", ">", "<=", ">=", "==", "!="]
    left: str
    right: str | TypedValue


class FactPattern(_M):
    atoms: list[PatternAtom] = Field(default_factory=list)
    constraints: list[ConstraintAtom] = Field(default_factory=list)


# --- header ----------------------------------------------------------------------------------
class RecordBase(_M):
    sef_version: str = SEF_VERSION
    record_id: str
    kind: str
    source_id: str | None = None
    span: Span | None = None
    data_category: str = "uncategorized"
    extraction: Extraction = Field(default_factory=lambda: Extraction(method="generator"))
    epistemic: Epistemic = Field(default_factory=Epistemic)
    context: Context = Field(default_factory=Context)
    links: list[Link] = Field(default_factory=list)
    content_hash: str | None = None

    @field_validator("sef_version")
    @classmethod
    def _major(cls, v: str) -> str:
        if int(v.split(".")[0]) != SEF_MAJOR:
            raise ValueError(f"unsupported SEF major version {v}")
        return v

    @model_validator(mode="after")
    def _source_required(self) -> "RecordBase":
        if self.kind != "source" and not self.source_id:
            raise ValueError(f"{self.kind} record {self.record_id} requires source_id")
        return self

    def compute_content_hash(self) -> str:
        data = self.model_dump(mode="json", exclude={"record_id", "content_hash"})
        return canonical_hash(data)

    def with_hash(self) -> "RecordBase":
        self.content_hash = self.compute_content_hash()
        return self


# --- record kinds ----------------------------------------------------------------------------
class SourceRecord(RecordBase):
    kind: Literal["source"] = "source"
    source_type: Literal[
        "generator_truth", "tool_execution", "curated_kb", "reference_doc", "textbook", "code_repository",
        "api_docs", "web_document", "forum", "user_statement", "model_output", "third_party_component",
    ]
    trust_class: str
    uri: str | None = None
    title: str | None = None
    author: str | None = None
    published_at: str | None = None
    retrieved_at: str | None = None
    license: str = "synthetic"
    root_source_id: str | None = None
    privacy_scope: Literal["public", "user_private", "org_private"] = "public"


class SegmentRecord(RecordBase):
    kind: Literal["segment"] = "segment"
    segment_id: str
    modality: Literal["text", "code", "math", "table", "dialog", "tool_output"] = "text"
    language: str = "en"
    text: str | None = None
    blob_ref: str | None = None
    parent_segment_id: str | None = None
    position: int = 0


class EntityRecord(RecordBase):
    kind: Literal["entity"] = "entity"
    entity_id: str
    canonical_name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    disambiguation: str = ""
    external_ids: dict[str, str] = Field(default_factory=dict)
    provisional: bool = False


class PropertyDefRecord(RecordBase):
    kind: Literal["property_def"] = "property_def"
    property_id: str
    name: str
    property_kind: Literal["geometric", "dynamic", "functional", "relational", "computational", "contextual", "other"]
    value_type: Literal["bool", "int", "float", "range", "enum", "entity_ref", "text"]
    unit: str | None = None
    enum_values: list[str] | None = None
    range: tuple[float, float] | None = None
    log_scale: bool = False
    description: str = ""
    provisional: bool = False


class RelationDefRecord(RecordBase):
    kind: Literal["relation_def"] = "relation_def"
    relation_id: str
    name: str
    arity: int = 2
    arg_types: list[str] = Field(default_factory=list)
    cardinality: Literal["functional", "multi"] = "multi"
    temporal: bool = False
    symmetric: bool = False
    inverse_of: str | None = None
    description: str = ""
    provisional: bool = False


class FactRecord(RecordBase):
    kind: Literal["fact"] = "fact"
    subject: str
    relation: str
    object: TypedValue
    qualifiers: dict[str, Any] = Field(default_factory=dict)


class PropertyConstraint(_M):
    property_id: str
    constraint: TypedValue  # value | range | enum set (value: list) | bool
    weight: float | None = None


class PartSpec(_M):
    role: str
    concept_id: str
    count: tuple[int, int] = (1, 1)


class Invariance(_M):
    transform: str
    preserves_identity: bool


class ConceptDefinitionRecord(RecordBase):
    kind: Literal["concept_definition"] = "concept_definition"
    concept_id: str
    name: str
    parent_ids: list[str] = Field(default_factory=list)
    defining_properties: list[PropertyConstraint] = Field(default_factory=list)
    typical_properties: list[PropertyConstraint] = Field(default_factory=list)
    parts: list[PartSpec] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    relations: list[FactPattern] = Field(default_factory=list)
    invariances: list[Invariance] = Field(default_factory=list)
    contrast_with: list[str] = Field(default_factory=list)


class SceneNode(_M):
    ref: str
    type: str | None = None
    properties: dict[str, TypedValue] = Field(default_factory=dict)


class SceneRelation(_M):
    relation: str
    args: list[str]


class Scene(_M):
    nodes: list[SceneNode] = Field(default_factory=list)
    relations: list[SceneRelation] = Field(default_factory=list)
    root: str | None = None


class ConceptExampleRecord(RecordBase):
    kind: Literal["concept_example"] = "concept_example"
    example_id: str
    label: str | None = None
    polarity: Literal["positive", "negative"] = "positive"
    scene: Scene
    salient_features: list[str] = Field(default_factory=list)


class ProcedureStep(_M):
    step_id: str
    action: str
    op: dict[str, Any] | None = None
    pre: list[FactPattern] = Field(default_factory=list)
    post: list[FactPattern] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)


class TypedPort(_M):
    name: str
    type: str


class ProcedureRecord(RecordBase):
    kind: Literal["procedure"] = "procedure"
    procedure_id: str
    goal: FactPattern | str
    inputs: list[TypedPort] = Field(default_factory=list)
    outputs: list[TypedPort] = Field(default_factory=list)
    steps: list[ProcedureStep]
    invariants: list[FactPattern] = Field(default_factory=list)
    domain: str = ""


class CausalRecord(RecordBase):
    kind: Literal["causal"] = "causal"
    cause: FactPattern
    effect: FactPattern
    conditions: list[FactPattern] = Field(default_factory=list)
    causal_polarity: Literal["causes", "prevents", "enables"] = "causes"
    strength: float | None = None
    mechanism: str | None = None
    evidence_type: Literal["mechanism", "intervention", "observational", "statement"]
    scope: dict[str, Any] = Field(default_factory=dict)


class Resolution(_M):
    preferred: str
    reason: str


class ContradictionRecord(RecordBase):
    kind: Literal["contradiction"] = "contradiction"
    claims: list[str | FactPattern]
    nature: Literal["value_conflict", "polarity_conflict", "temporal_conflict", "definitional_conflict", "disputed"]
    resolution: Resolution | None = None
    declared_by: str | None = None


class KnownUnknownRecord(RecordBase):
    kind: Literal["known_unknown"] = "known_unknown"
    pattern: FactPattern
    scope: Literal["global", "source", "domain"] = "global"
    as_of: str | None = None
    note: str = ""


class Verifier(_M):
    type: Literal[
        "exact", "tolerance", "unit_tests", "property_tests", "proof_check", "constraint_set", "reference_compare",
    ]
    spec: dict[str, Any] = Field(default_factory=dict)


class SkillDemoRecord(RecordBase):
    kind: Literal["skill_demo"] = "skill_demo"
    skill_hint: str | None = None
    input: Any
    output: Any
    trace: list[dict[str, Any]] | None = None
    checker: Verifier | None = None
    domain: str = ""


class TaskRecord(RecordBase):
    kind: Literal["task"] = "task"
    task_id: str
    family: str
    goal: dict[str, Any]
    inputs: dict[str, Any] = Field(default_factory=dict)
    verifier: Verifier
    reference_solution: Any = None
    split: str = "train"
    difficulty: float | None = None


class QuestionRecord(RecordBase):
    kind: Literal["question"] = "question"
    query: str
    pattern: FactPattern | None = None
    gold_state: Literal[
        "KNOWN", "PREDICTION", "UNKNOWN-ABSENT", "UNKNOWN-NO-BASIS", "UNKNOWN-DECLARED", "CONTESTED", "INSUFFICIENT",
    ]
    gold_answer: TypedValue | None = None
    acceptable_hedges: list[str] = Field(default_factory=list)
    split: str = "dev"


class ExpressionPairRecord(RecordBase):
    kind: Literal["expression_pair"] = "expression_pair"
    plan: dict[str, Any]
    target_text: str
    target_format: str = "prose"
    style: str | None = None


SEFRecord = Annotated[
    Union[
        SourceRecord, SegmentRecord, EntityRecord, PropertyDefRecord, RelationDefRecord, FactRecord,
        ConceptDefinitionRecord, ConceptExampleRecord, ProcedureRecord, CausalRecord, ContradictionRecord,
        KnownUnknownRecord, SkillDemoRecord, TaskRecord, QuestionRecord, ExpressionPairRecord,
    ],
    Field(discriminator="kind"),
]

_ADAPTER: TypeAdapter[Any] = TypeAdapter(SEFRecord)


def parse_record(data: dict[str, Any]) -> RecordBase:
    """Validate a raw dict as an SEF record (02 §8 schema validation)."""
    return _ADAPTER.validate_python(data)


def record_to_dict(record: RecordBase) -> dict[str, Any]:
    return record.model_dump(mode="json", exclude_none=True)
