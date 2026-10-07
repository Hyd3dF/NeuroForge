"""Type-directed bottom-up synthesis over the DSL (05 §5.2 step 4).

Programs are enumerated by size with observational-equivalence pruning: two programs
of the same type with the same values on every example (and, for functions, on a
probe domain) are interchangeable, so only the smallest is kept.  Library programs
(previously solved procedures, 08 §6) enter at size 1, which is where reuse pays
off.  ``step(budget)`` runs incrementally so the Heart can fund it per beat.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterator

from srm.body import dsl

PROBE = tuple(range(-9, 10))
MAX_LIST = 64


@dataclass
class Entry:
    node: dsl.Node
    type: str
    values: tuple[Any, ...]  # per example: list | int | callable
    key: tuple[Any, ...]


def _freeze(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v


def _key(t: str, values: tuple[Any, ...]) -> tuple[Any, ...]:
    if t in ("fn", "pred"):
        out = []
        for f in values:
            try:
                out.append(tuple(f(p) for p in PROBE))
            except Exception:  # noqa: BLE001
                out.append(None)
        return (t, tuple(out))
    return (t, tuple(_freeze(v) for v in values))


@dataclass
class Synthesizer:
    examples: list[dict[str, Any]]  # [{xs, k, out}]
    ret_type: str  # "list" | "int"
    library: list[dsl.Node] = field(default_factory=list)
    max_size: int = 7
    literals: tuple[int, ...] = (0, 1, 2, 3)
    max_solutions: int = 4
    alt_max_size: int = 3  # below this size, one extra level is searched for alternatives (Occam needs rivals)
    evaluated: int = 0
    solutions: list[dsl.Node] = field(default_factory=list)
    solution_size: int | None = None
    exhausted: bool = False

    def __post_init__(self) -> None:
        self.n = len(self.examples)
        self.target = tuple(_freeze(e["out"]) for e in self.examples)
        self.bank: dict[tuple[str, int], list[Entry]] = {}
        self.seen: set[tuple[Any, ...]] = set()
        self._gen = self._enumerate()

    # --- bank -----------------------------------------------------------------------------------
    def _add(self, node: dsl.Node, t: str, values: tuple[Any, ...], size: int) -> None:
        # solutions are collected before observational-equivalence pruning: programs that agree on
        # the examples are exactly the rival hypotheses the Simulator and Support must separate
        if t == self.ret_type and tuple(_freeze(v) for v in values) == self.target:
            if self.solution_size is None:
                self.solution_size = size
            if size <= self.solution_limit() and len(self.solutions) < self.max_solutions \
                    and all(str(node) != str(s) for s in self.solutions):
                self.solutions.append(node)
        key = _key(t, values)
        if key in self.seen:
            return
        self.seen.add(key)
        self.bank.setdefault((t, size), []).append(Entry(node, t, values, key))

    def solution_limit(self) -> int:
        """Collect solutions up to the first solution size (+1 when that size is small)."""
        if self.solution_size is None:
            return self.max_size
        return self.solution_size + (1 if self.solution_size <= self.alt_max_size else 0)

    def _terminals(self) -> None:
        self._add(dsl.Node("xs"), "list", tuple(list(e["xs"]) for e in self.examples), 1)
        self._add(dsl.Node("k"), "int", tuple(e["k"] for e in self.examples), 1)
        for c in self.literals:
            self._add(dsl.Node("lit", (), c), "int", tuple(c for _ in self.examples), 1)
        for name, (args, ret, impl) in dsl.PRIMITIVES.items():
            if not args:
                f = impl()
                self._add(dsl.Node(name), ret, tuple(f for _ in self.examples), 1)
        for prog in self.library:  # stored procedures count as single components
            try:
                t = dsl.type_of(prog)
                vals = tuple(dsl.run(prog, e["xs"], e["k"]) for e in self.examples)
            except dsl.DSLError:
                continue
            self._add(prog, t, vals, 1)

    def _enumerate(self) -> Iterator[None]:
        self._terminals()
        yield
        ops = [(name, args, impl) for name, (args, ret, impl) in dsl.PRIMITIVES.items() if args]
        for size in range(2, self.max_size + 1):
            if self.solution_size is not None and size > self.solution_limit():
                break
            for name, arg_types, impl in ops:
                ret = dsl.PRIMITIVES[name][1]
                n_args = len(arg_types)
                for split in _compositions(size - 1, n_args):
                    pools = [self.bank.get((t, s), []) for t, s in zip(arg_types, split)]
                    if any(not p for p in pools):
                        continue
                    for combo in itertools.product(*pools):
                        self.evaluated += 1
                        vals = []
                        ok = True
                        for e in range(self.n):
                            try:
                                v = impl(*[c.values[e] for c in combo])
                            except (dsl.DSLError, TypeError, ValueError, ZeroDivisionError, OverflowError, RecursionError):
                                ok = False
                                break
                            if isinstance(v, list) and len(v) > MAX_LIST:
                                ok = False
                                break
                            vals.append(v)
                        if ok:
                            self._add(dsl.Node(name, tuple(c.node for c in combo)), ret, tuple(vals), size)
                        yield
        self.exhausted = True

    # --- public -----------------------------------------------------------------------------------
    def step(self, budget: int) -> bool:
        """Advance by up to ``budget`` evaluations; returns True when finished (found or exhausted)."""
        start = self.evaluated
        while self.evaluated - start < budget:
            try:
                next(self._gen)
            except StopIteration:
                self.exhausted = True
                return True
            if self.solutions and self._level_done():
                return True
        return self.exhausted

    def _level_done(self) -> bool:
        # stop as soon as the minimal-size solutions are collected (Occam)
        return len(self.solutions) >= self.max_solutions or self.exhausted

    def run(self, budget: int = 200_000) -> list[dsl.Node]:
        while not self.step(min(budget, 5_000)) and self.evaluated < budget:
            pass
        return self.solutions


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)
