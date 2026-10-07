"""Kind-specific record payloads (04 §2.2–2.8)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from srm.interface.messages import Qualifiers, RecordKind
from srm.interface.values import Value


@dataclass
class EntityPayload:
    entity_id: str
    name: str
    entity_type: str
    aliases: list[str] = field(default_factory=list)
    provisional: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EntityPayload":
        return EntityPayload(**d)


@dataclass
class EngramPayload:
    """A fact ``(subject, relation, object)`` with qualifiers, polarity and modality (04 §2.3)."""

    subject: str
    relation: str
    object: Value
    qualifiers: Qualifiers = field(default_factory=Qualifiers)
    polarity: str = "positive"
    modality: str = "asserted"
    attribution: str | None = None
    attributed_evidence: float = 0.0  # evidence for "attribution asserts P" (02 §10.2, D-013)
    hedge: float = 0.0
    derived_from: list[int] = field(default_factory=list)
    sef_record_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["object"] = self.object.to_dict()
        d["qualifiers"] = self.qualifiers.to_dict()
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EngramPayload":
        d = dict(d)
        d["object"] = Value.from_dict(d["object"])
        d["qualifiers"] = Qualifiers.from_dict(d.get("qualifiers"))
        return EngramPayload(**d)


@dataclass
class OpenQuestionPayload:
    """A declared unknown (04 §2.8, from SEF ``known_unknown``)."""

    subject: str
    relation: str
    scope: str = "global"
    as_of: str | None = None
    note: str = ""
    source_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "OpenQuestionPayload":
        return OpenQuestionPayload(**d)


@dataclass
class ProcedurePayload:
    """An explicit, inspectable recipe (04 §2.4): a circuit template, DSL program or formula."""

    procedure_id: str
    domain: str
    signature: str  # e.g. "dsl:list", "math:number"
    goal: str = ""
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    program: Any = None  # DSL program JSON when the procedure is a compiled solution
    cues: list[str] = field(default_factory=list)
    uses: int = 0
    successes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ProcedurePayload":
        return ProcedurePayload(**d)


@dataclass
class EpisodePayload:
    """A task/query episode with its circuit trace and outcome (04 §2.6)."""

    episode_id: str
    goal: str
    kind: str
    outcome: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    record_ids: list[int] = field(default_factory=list)
    replay_priority: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EpisodePayload":
        return EpisodePayload(**d)


@dataclass
class GenericPayload:
    """Structured payload for kinds whose full schema is implemented in later milestones."""

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"data": self.data}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GenericPayload":
        return GenericPayload(dict(d.get("data", {})))


PAYLOAD_TYPES: dict[RecordKind, type] = {
    RecordKind.ENTITY: EntityPayload,
    RecordKind.ENGRAM: EngramPayload,
    RecordKind.OPEN_QUESTION: OpenQuestionPayload,
    RecordKind.PROCEDURE: ProcedurePayload,
    RecordKind.EPISODE: EpisodePayload,
}


def payload_from_dict(kind: RecordKind, d: dict[str, Any]) -> Any:
    cls = PAYLOAD_TYPES.get(kind, GenericPayload)
    return cls.from_dict(d)  # type: ignore[attr-defined]


def register_payload_type(kind: RecordKind, cls: type) -> None:
    """Later milestones register full payload schemas (concepts, procedures, episodes, skills)."""
    PAYLOAD_TYPES[kind] = cls
