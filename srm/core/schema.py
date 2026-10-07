"""Factored schemas and variability profiles (04 §2.2).

A ``PropertyProfile`` summarizes one property within a concept: scalar (log-space
Welford), enum (Dirichlet counts) or bool (Beta).  Normalized variability gives the
defining weight ``w_def = clip(1 − v, 0, 1) × presence_rate``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from srm.interface.values import Value


def _log(x: float) -> float:
    return math.log10(max(abs(float(x)), 1e-12))


@dataclass
class PropertyProfile:
    vtype: str  # scalar | enum | bool
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    lo: float = math.inf
    hi: float = -math.inf
    counts: dict[str, float] = field(default_factory=dict)
    unit: str | None = None
    prior_std: float | None = None  # from an explicit definition range or an overhypothesis

    @staticmethod
    def for_value(v: Value) -> "PropertyProfile":
        if v.is_numeric():
            return PropertyProfile("scalar", unit=v.unit)
        if v.type == "bool":
            return PropertyProfile("bool")
        return PropertyProfile("enum")

    @staticmethod
    def from_constraint(c: Value, defining: bool) -> "PropertyProfile":
        """Initialize from a definition constraint (range / enum set / bool) (02 §9.2)."""
        if c.type == "range":
            lo, hi = float(c.value[0]), float(c.value[1])
            p = PropertyProfile("scalar", unit=c.unit)
            mid = (_log(lo) + _log(hi)) / 2.0
            p.mean, p.lo, p.hi = mid, _log(lo), _log(hi)
            p.prior_std = max((_log(hi) - _log(lo)) / (4.0 if defining else 3.0), 1e-3)
            return p
        if c.type == "bool":
            p = PropertyProfile("bool")
            p.counts = {"true": 9.0, "false": 1.0} if c.value else {"true": 1.0, "false": 9.0}
            return p
        values = c.value if isinstance(c.value, (list, tuple)) else [c.value]
        p = PropertyProfile("enum")
        p.counts = {str(v): 9.0 / len(values) for v in values}
        return p

    # --- updates --------------------------------------------------------------------------------
    def update(self, v: Value) -> None:
        self.n += 1
        if self.vtype == "scalar":
            x = _log(v.value)
            delta = x - self.mean
            self.mean += delta / self.n
            self.m2 += delta * (x - self.mean)
            self.lo, self.hi = min(self.lo, x), max(self.hi, x)
        else:
            key = str(v.value).lower() if self.vtype == "bool" else str(v.value)
            self.counts[key] = self.counts.get(key, 0.0) + 1.0

    @property
    def std(self) -> float:
        empirical = math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else None
        if empirical is not None and self.prior_std is not None:
            w = self.n / (self.n + 3.0)
            return max(w * empirical + (1 - w) * self.prior_std, 1e-3)
        if empirical is not None:
            return max(empirical, 1e-3)
        return self.prior_std if self.prior_std is not None else 0.5

    def prob(self, v: Value) -> float:
        total = sum(self.counts.values())
        if total <= 0:
            return 0.5
        key = str(v.value).lower() if self.vtype == "bool" else str(v.value)
        return (self.counts.get(key, 0.0) + 0.1) / (total + 0.1 * max(2, len(self.counts) + 1))

    def z(self, v: Value) -> float:
        return abs(_log(v.value) - self.mean) / self.std

    def variability(self, global_std: float = 1.0) -> float:
        if self.vtype == "scalar":
            return min(1.0, self.std / max(global_std, 1e-6))
        total = sum(self.counts.values())
        if total <= 0 or len(self.counts) <= 1:
            return 0.0
        ps = [c / total for c in self.counts.values()]
        h = -sum(p * math.log(p) for p in ps if p > 0)
        return h / math.log(max(2, len(self.counts)))

    def typical(self) -> Value:
        if self.vtype == "scalar":
            return Value("float", round(10 ** self.mean, 4), self.unit)
        best = max(self.counts.items(), key=lambda t: t[1])[0] if self.counts else "?"
        if self.vtype == "bool":
            return Value("bool", best == "true")
        return Value("enum", best)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["lo"] = None if math.isinf(self.lo) else self.lo
        d["hi"] = None if math.isinf(self.hi) else self.hi
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PropertyProfile":
        d = dict(d)
        d["lo"] = math.inf if d.get("lo") is None else d["lo"]
        d["hi"] = -math.inf if d.get("hi") is None else d["hi"]
        return PropertyProfile(**d)


@dataclass
class Schema:
    """A factored concept description: property profiles, parts and context priors."""

    schema_id: str
    name: str = ""
    parents: list[str] = field(default_factory=list)
    profiles: dict[str, PropertyProfile] = field(default_factory=dict)
    defining: dict[str, bool] = field(default_factory=dict)  # explicit definitional flags, when known
    presence: dict[str, float] = field(default_factory=dict)  # fraction of instances showing the property
    parts: dict[str, float] = field(default_factory=dict)  # part concept → mean count
    contexts: dict[str, float] = field(default_factory=dict)
    n_instances: int = 0

    def w_def(self, pid: str, global_std: float = 1.0) -> float:
        prof = self.profiles[pid]
        if pid in self.defining:
            return 1.0 if self.defining[pid] else 0.3
        return max(0.0, 1.0 - prof.variability(global_std)) * self.presence.get(pid, 1.0)

    def observe(self, props: dict[str, Value], parts: list[str] | None = None, contexts: list[str] | None = None) -> None:
        self.n_instances += 1
        for pid, v in props.items():
            prof = self.profiles.get(pid)
            if prof is None:
                prof = self.profiles[pid] = PropertyProfile.for_value(v)
            prof.update(v)
        for pid in self.profiles:
            seen = 1.0 if pid in props else 0.0
            old = self.presence.get(pid, seen)
            self.presence[pid] = old + (seen - old) / self.n_instances
        for p in parts or []:
            self.parts[p] = self.parts.get(p, 0.0) + 1.0
        for c in contexts or []:
            self.contexts[c] = self.contexts.get(c, 0.0) + 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id, "name": self.name, "parents": list(self.parents),
            "profiles": {k: v.to_dict() for k, v in self.profiles.items()}, "defining": dict(self.defining),
            "presence": dict(self.presence), "parts": dict(self.parts), "contexts": dict(self.contexts),
            "n_instances": self.n_instances,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Schema":
        s = Schema(d["schema_id"], d.get("name", ""), list(d.get("parents", [])))
        s.profiles = {k: PropertyProfile.from_dict(v) for k, v in d.get("profiles", {}).items()}
        s.defining = dict(d.get("defining", {}))
        s.presence = dict(d.get("presence", {}))
        s.parts = dict(d.get("parts", {}))
        s.contexts = dict(d.get("contexts", {}))
        s.n_instances = int(d.get("n_instances", 0))
        return s
