"""Typed values and value encoders (03 §3 ``value_encoder``, 03 §5 ``Value``).

Scalars use thermometer-style bucket codes so that nearby values share blocks;
enums and booleans use identity codes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from srm.interface import codes as C
from srm.interface.codespace import CodeSpace, normalize, seeded_dense

NUMERIC_TYPES = ("int", "float")


@dataclass(frozen=True)
class Value:
    """A typed value: int, float, bool, enum, text, date, range, entity_ref, code_ref, expr_ref."""

    type: str
    value: Any
    unit: str | None = None
    tolerance: float = 0.0

    def is_numeric(self) -> bool:
        return self.type in NUMERIC_TYPES

    def matches(self, other: "Value") -> bool:
        """Equality within tolerance (02 §11 value-conflict rule)."""
        if self.is_numeric() and other.is_numeric():
            if self.unit != other.unit and self.unit is not None and other.unit is not None:
                return False
            a, b = float(self.value), float(other.value)
            tol = max(self.tolerance, other.tolerance, 1e-9 * max(abs(a), abs(b), 1.0))
            return abs(a - b) <= tol
        if self.type == "range" and other.type == "range":
            return tuple(self.value) == tuple(other.value)
        return self.type == other.type and self.value == other.value

    def key(self) -> str:
        """Canonical string used in the structured triple index (04 §4.3)."""
        if self.type == "entity_ref":
            return str(self.value)
        if self.is_numeric():
            return f"{self.type}:{float(self.value):.12g}:{self.unit or ''}"
        return f"{self.type}:{self.value!r}"

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "value": self.value}
        if self.unit is not None:
            d["unit"] = self.unit
        if self.tolerance:
            d["tolerance"] = self.tolerance
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Value":
        v = d["value"]
        if isinstance(v, list):
            v = tuple(v)
        return Value(type=d["type"], value=v, unit=d.get("unit"), tolerance=float(d.get("tolerance") or 0.0))


def entity(entity_id: str) -> Value:
    return Value("entity_ref", entity_id)


class ValueEncoder:
    """Encodes values as fillers: codes plus dense vectors."""

    def __init__(self, codespace: CodeSpace, buckets: int, salt: str = "value") -> None:
        self.cs = codespace
        self.buckets = buckets
        self.salt = salt

    def bucket(self, x: float, lo: float, hi: float, log: bool = False) -> int:
        if log:
            x, lo, hi = (math.log(max(v, 1e-12)) for v in (x, lo, hi))
        if hi <= lo:
            return 0
        t = (x - lo) / (hi - lo)
        return int(min(self.buckets - 1, max(0, round(t * (self.buckets - 1)))))

    def scalar_code(self, prop_key: str, x: float, lo: float, hi: float, log: bool = False) -> np.ndarray:
        """Thermometer code: the first ``m`` blocks come from the HI code, the rest from LO."""
        B, L = self.cs.B, self.cs.L
        lo_code = C.code_from_key(prop_key + "#lo", B, L, self.salt)
        hi_code = C.code_from_key(prop_key + "#hi", B, L, self.salt)
        t = self.bucket(x, lo, hi, log)
        m = round(t / max(1, self.buckets - 1) * B)
        out = lo_code.copy()
        out[:m] = hi_code[:m]
        return out

    def scalar_dense(self, prop_key: str, x: float, lo: float, hi: float, log: bool = False) -> np.ndarray:
        d = self.cs.d
        t = self.bucket(x, lo, hi, log) / max(1, self.buckets - 1)
        lo_v = seeded_dense(prop_key + "#lo", d, self.salt)
        hi_v = seeded_dense(prop_key + "#hi", d, self.salt)
        return normalize((1.0 - t) * lo_v + t * hi_v)

    def symbol_code(self, prop_key: str, value: Any) -> np.ndarray:
        return C.code_from_key(f"{prop_key}={value!r}", self.cs.B, self.cs.L, self.salt)

    def symbol_dense(self, prop_key: str, value: Any) -> np.ndarray:
        return seeded_dense(f"{prop_key}={value!r}", self.cs.d, self.salt)

    def encode(self, prop_key: str, value: Value, lo: float = 0.0, hi: float = 1.0, log: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(code, dense)`` for a value used as a filler of property ``prop_key``."""
        if value.is_numeric():
            x = float(value.value)
            return self.scalar_code(prop_key, x, lo, hi, log), self.scalar_dense(prop_key, x, lo, hi, log)
        return self.symbol_code(prop_key, value.value), self.symbol_dense(prop_key, value.value)
