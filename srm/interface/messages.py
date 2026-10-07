"""Message types exchanged between components (03 §5) and shared enumerations.

Components exchange only these types (14 §3 rule 2).  The workspace ``Claim``
(message fields plus bookkeeping) lives in :mod:`srm.core.workspace`.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from srm.interface.values import Value

MESSAGE_SCHEMA_VERSION = "1.0"


class Modality(str, enum.Enum):
    ASSERTED = "asserted"
    HEDGED = "hedged"
    REPORTED = "reported"
    HYPOTHETICAL = "hypothetical"
    CONDITIONAL = "conditional"
    NEGATED = "negated"
    UNKNOWN_DECLARED = "unknown_declared"
    FICTIONAL = "fictional"


class Polarity(str, enum.Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class EpistemicState(str, enum.Enum):
    """The epistemic state lattice (06 §5)."""

    OBSERVED = "OBSERVED"
    REMEMBERED = "REMEMBERED"
    DERIVED = "DERIVED"
    TESTED = "TESTED"
    PREDICTED = "PREDICTED"
    INHERITED = "INHERITED"
    ANALOGICAL = "ANALOGICAL"
    EXTRAPOLATED = "EXTRAPOLATED"
    SUGGESTED = "SUGGESTED"
    CONJECTURED = "CONJECTURED"
    CONTESTED = "CONTESTED"
    INSUFFICIENT = "INSUFFICIENT"
    UNRESOLVED = "UNRESOLVED"
    UNKNOWN_ABSENT = "UNKNOWN-ABSENT"
    UNKNOWN_DECLARED = "UNKNOWN-DECLARED"
    UNKNOWN_NO_BASIS = "UNKNOWN-NO-BASIS"


KNOWLEDGE_STATES = frozenset(
    {EpistemicState.OBSERVED, EpistemicState.REMEMBERED, EpistemicState.DERIVED, EpistemicState.TESTED}
)
PREDICTION_STATES = frozenset(
    {
        EpistemicState.PREDICTED, EpistemicState.INHERITED, EpistemicState.ANALOGICAL,
        EpistemicState.EXTRAPOLATED, EpistemicState.SUGGESTED,
    }
)


class Lifecycle(str, enum.Enum):
    NEW = "NEW"
    CORROBORATED = "CORROBORATED"
    USED = "USED"
    CONSOLIDATED = "CONSOLIDATED"
    STABLE = "STABLE"
    CONTESTED = "CONTESTED"
    DEPRECATED = "DEPRECATED"
    DECAYED = "DECAYED"


LIFECYCLE_ORDER = {
    Lifecycle.NEW: 0, Lifecycle.CORROBORATED: 1, Lifecycle.USED: 2,
    Lifecycle.CONSOLIDATED: 3, Lifecycle.STABLE: 4,
}


class ClaimStatus(str, enum.Enum):
    OPEN = "OPEN"
    CHECKED = "CHECKED"
    COMMITTED = "COMMITTED"
    RETRACTED = "RETRACTED"
    DORMANT = "DORMANT"


class PredictionLevel(str, enum.Enum):
    L0 = "L0"
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    L6 = "L6"


class EvidenceKind(str, enum.Enum):
    OBSERVATION = "observation"
    MEMORY = "memory"
    VERIFICATION = "verification"
    TEST = "test"
    DERIVATION = "derivation"
    CORROBORATION = "corroboration"
    CONTRADICTION = "contradiction"
    PREDICTION_ERROR = "prediction_error"


class RecordKind(str, enum.Enum):
    ENTITY = "ENTITY"
    CONCEPT = "CONCEPT"
    ENGRAM = "ENGRAM"
    PROCEDURE = "PROCEDURE"
    SKILL_REF = "SKILL_REF"
    EPISODE = "EPISODE"
    OPEN_QUESTION = "OPEN_QUESTION"
    PROPERTY_REF = "PROPERTY_REF"


class DecisionClass(str, enum.Enum):
    """Outcome classes of the goal decision procedure (06 §6)."""

    KNOWN = "KNOWN"
    PREDICTION = "PREDICTION"
    AMBIGUOUS = "AMBIGUOUS"
    CONTESTED = "CONTESTED"
    INSUFFICIENT = "INSUFFICIENT"
    UNRESOLVED = "UNRESOLVED"
    UNKNOWN_ABSENT = "UNKNOWN-ABSENT"
    UNKNOWN_DECLARED = "UNKNOWN-DECLARED"
    UNKNOWN_NO_BASIS = "UNKNOWN-NO-BASIS"


# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Qualifiers:
    """Claim qualifiers: time interval ``[start, end)``, version, condition, location, world."""

    time: tuple[float, float] | None = None
    version: str | None = None
    condition: str | None = None
    location: str | None = None
    world_id: str | None = None

    def compatible(self, other: "Qualifiers") -> bool:
        """Overlapping time, matching version/world/condition when both specify them (02 §9.1)."""
        if self.time is not None and other.time is not None:
            if not (self.time[0] < other.time[1] and other.time[0] < self.time[1]):
                return False
        for name in ("version", "world_id", "condition", "location"):
            a, b = getattr(self, name), getattr(other, name)
            if a is not None and b is not None and a != b:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items() if v is not None}

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "Qualifiers":
        if not d:
            return Qualifiers()
        t = d.get("time")
        return Qualifiers(
            time=(float(t[0]), float(t[1])) if t is not None else None,
            version=d.get("version"), condition=d.get("condition"),
            location=d.get("location"), world_id=d.get("world_id"),
        )


def is_var(term: Any) -> bool:
    return isinstance(term, str) and term.startswith("?")


@dataclass(frozen=True)
class PatternAtom:
    """``(subject, relation, object)`` where any of subject/object may be a ``?variable``."""

    subject: str
    relation: str
    object: str | Value

    def variables(self) -> list[str]:
        return [t for t in (self.subject, self.object) if is_var(t)]


@dataclass
class Goal:
    goal_id: str
    atoms: list[PatternAtom]
    answer_vars: list[tuple[str, str]] = field(default_factory=list)  # (var, type)
    qualifiers: Qualifiers = field(default_factory=Qualifiers)
    stakes: float = 0.5
    deadline_beats: int | None = None
    surface_forms: dict[str, str] = field(default_factory=dict)  # unlinked "@name" → text


@dataclass(frozen=True)
class EvidenceEvent:
    event_id: str
    target_id: str
    delta_plus: float
    delta_minus: float
    kind: EvidenceKind
    source_root_id: str
    producer_id: str = ""


@dataclass
class Referent:
    ref_id: str
    type_id: str
    dense: np.ndarray | None = None
    content_code: np.ndarray | None = None
    identity_code: np.ndarray | None = None
    bound_record: int | None = None


@dataclass
class Bid:
    process_id: str
    resources: dict[str, float]
    predicted_value: float
    confidence: float = 1.0
