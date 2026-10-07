"""Simulator (05 §10): rollouts and counterexample generation.

F0 domains: DSL programs (property-based random inputs, behaviour grouping, distinguishing
inputs) and symbolic identities (random numeric substitution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from srm.body import dsl


@dataclass
class ProgramProbe:
    program: dsl.Node
    errors: int
    behaviour: tuple[Any, ...]


def probe_inputs(rng: np.random.Generator, n: int, examples: list[dict[str, Any]] | None = None) -> list[tuple[list[int], int]]:
    """Random typed inputs plus boundary cases (empty list, singleton, k outside list length)."""
    out: list[tuple[list[int], int]] = [([], 0), ([0], 1), ([5, -3, 5], 7), ([9, 8, 7, 6], -1)]
    if examples:
        ks = sorted({int(e["k"]) for e in examples})
        out += [([1, 2, 3], k) for k in ks]
    while len(out) < n:
        out.append(dsl.random_input(rng))
    return out[:n]


def probe_program(program: dsl.Node, inputs: list[tuple[list[int], int]]) -> ProgramProbe:
    errors, behaviour = 0, []
    for xs, k in inputs:
        try:
            v = dsl.run(program, xs, k)
            behaviour.append(tuple(v) if isinstance(v, list) else v)
        except dsl.DSLError:
            errors += 1
            behaviour.append("<error>")
    return ProgramProbe(program, errors, tuple(behaviour))


def group_by_behaviour(programs: list[dsl.Node], inputs: list[tuple[list[int], int]]) -> list[list[ProgramProbe]]:
    """Programs that agree on every probe input are (probably) equivalent and are pooled."""
    groups: dict[tuple[Any, ...], list[ProgramProbe]] = {}
    for p in programs:
        pr = probe_program(p, inputs)
        groups.setdefault(pr.behaviour, []).append(pr)
    return list(groups.values())


def distinguishing_input(a: dsl.Node, b: dsl.Node, inputs: list[tuple[list[int], int]]) -> tuple[list[int], int] | None:
    for xs, k in inputs:
        ra, rb = probe_program(a, [(xs, k)]), probe_program(b, [(xs, k)])
        if ra.behaviour != rb.behaviour:
            return xs, k
    return None


def identity_counterexample(lhs: str, rhs: str, symbol: str = "x", trials: int = 12, seed: int = 0) -> float | None:
    """Numeric substitution search for a value where two expressions differ."""
    x = sp.Symbol(symbol)
    f, g = sp.sympify(lhs.replace("^", "**")), sp.sympify(rhs.replace("^", "**"))
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        v = float(rng.uniform(-10, 10))
        if abs(float(f.subs(x, v)) - float(g.subs(x, v))) > 1e-6 * max(1.0, abs(float(f.subs(x, v)))):
            return v
    return None
