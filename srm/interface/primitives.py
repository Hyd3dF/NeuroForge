"""Primitive Basis signatures (05 §4).  Part of the interface layer (03 §1.2).

Adding a primitive is a MINOR interface change; changing a signature is MAJOR.
Implementations live in ``srm.core.primitives`` (later batches); the signatures
are fixed here so that the ABI hash covers them.
"""

from __future__ import annotations

from dataclasses import dataclass

from srm.util.canonical import canonical_hash


@dataclass(frozen=True)
class PrimitiveSignature:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    impl: str  # "A" exact algorithm, "L" learned, "H" hybrid

    def describe(self) -> dict[str, object]:
        return {"name": self.name, "inputs": list(self.inputs), "outputs": list(self.outputs), "impl": self.impl}


PRIMITIVE_SIGNATURES: tuple[PrimitiveSignature, ...] = (
    PrimitiveSignature("BIND", ("Code", "Code"), ("Code",), "A"),
    PrimitiveSignature("UNBIND", ("Code", "Code"), ("Code",), "A"),
    PrimitiveSignature("BUNDLE", ("Code[]",), ("Bundle",), "A"),
    PrimitiveSignature("PERMUTE", ("Code", "int"), ("Code",), "A"),
    PrimitiveSignature("SPARSIFY", ("Bundle",), ("Code",), "A"),
    PrimitiveSignature("SIM", ("Code|Dense", "Code|Dense"), ("float",), "A"),
    PrimitiveSignature("RETRIEVE", ("Code", "Dense", "Filters"), ("Record[]",), "A"),
    PrimitiveSignature("CLEANUP", ("Dense", "Record[]"), ("Dense",), "A"),
    PrimitiveSignature("COMPARE", ("SceneGraph|Schema", "SceneGraph|Schema"), ("DiffRecord",), "H"),
    PrimitiveSignature("ALIGN", ("SceneGraph", "SceneGraph"), ("Correspondence", "float", "Claim[]"), "H"),
    PrimitiveSignature("ABSTRACT", ("SceneGraph[]",), ("Schema",), "A"),
    PrimitiveSignature("SPECIALIZE", ("Schema", "Bindings"), ("SceneGraph",), "A"),
    PrimitiveSignature("TRANSFORM", ("Operator", "Value[]"), ("Value|SceneGraph",), "A"),
    PrimitiveSignature("ORDER", ("Value[]",), ("Value[]",), "A"),
    PrimitiveSignature("COUNT", ("Value[]",), ("Value",), "A"),
    PrimitiveSignature("AGGREGATE", ("Value[]",), ("Value",), "A"),
    PrimitiveSignature("PREDICT_STEP", ("SceneGraph", "Action"), ("PredictionRecord[]",), "H"),
    PrimitiveSignature("REGRESS", ("Goal",), ("Plan[]",), "H"),
    PrimitiveSignature("DECOMPOSE", ("Goal",), ("Goal[]",), "H"),
    PrimitiveSignature("EXPLAIN", ("EvidenceEvent[]", "Hypothesis[]"), ("float[]",), "H"),
    PrimitiveSignature("TEST", ("Claim|Constraint",), ("EvidenceEvent",), "A"),
    PrimitiveSignature("SEARCH", ("Generator", "Scorer", "Budget"), ("Candidate[]",), "A"),
    PrimitiveSignature("ITERATE", ("CircuitGraph", "Value", "Test", "int"), ("Value",), "A"),
    PrimitiveSignature("BRANCH", ("bool", "CircuitGraph", "CircuitGraph"), ("Value",), "A"),
    PrimitiveSignature("UNIFY", ("Pattern", "Claim"), ("Bindings|None",), "A"),
    PrimitiveSignature("QUERY_JOIN", ("Pattern[]",), ("Bindings[]",), "A"),
)

PRIMITIVE_SET_VERSION = "1.0"


def primitive_signature_digest() -> str:
    return canonical_hash({"version": PRIMITIVE_SET_VERSION, "primitives": [p.describe() for p in PRIMITIVE_SIGNATURES]})


def primitive(name: str) -> PrimitiveSignature:
    for p in PRIMITIVE_SIGNATURES:
        if p.name == name:
            return p
    raise KeyError(name)
