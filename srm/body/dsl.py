"""Typed list/integer DSL: types, primitives, interpreter, sampler, Python-subset printer (02 §14.2).

Used by the DSLWorld generator (training/eval tasks), the Body's DSL tool (07 §6.2)
and, later, by the Composer as a source of symbolic TRANSFORM operators.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

# Types: "int", "bool", "list", "fn" (int→int), "pred" (int→bool)


class DSLError(Exception):
    """Raised when a program is undefined on an input (e.g. head of an empty list)."""


@dataclass(frozen=True)
class Node:
    op: str
    args: tuple["Node", ...] = ()
    value: int | None = None  # literal payload for op == "lit"

    def to_json(self) -> Any:
        if self.op == "lit":
            return {"lit": self.value}
        if not self.args:
            return self.op
        return [self.op, *[a.to_json() for a in self.args]]

    @staticmethod
    def from_json(obj: Any) -> "Node":
        if isinstance(obj, dict):
            return Node("lit", (), int(obj["lit"]))
        if isinstance(obj, str):
            return Node(obj)
        return Node(obj[0], tuple(Node.from_json(a) for a in obj[1:]))

    def ops(self) -> list[str]:
        out = [self.op]
        for a in self.args:
            out.extend(a.ops())
        return out

    def edges(self) -> list[tuple[str, str]]:
        """Parent→child operator adjacencies (used for held-out composition splits)."""
        out = [(self.op, a.op) for a in self.args if a.op not in ("xs", "k", "lit")]
        for a in self.args:
            out.extend(a.edges())
        return out

    def depth(self) -> int:
        return 1 + max((a.depth() for a in self.args), default=0)

    def __str__(self) -> str:
        if self.op == "lit":
            return str(self.value)
        if not self.args:
            return self.op
        return f"({self.op} {' '.join(str(a) for a in self.args)})"


def _head(xs: list[int]) -> int:
    if not xs:
        raise DSLError("head of empty list")
    return xs[0]


def _last(xs: list[int]) -> int:
    if not xs:
        raise DSLError("last of empty list")
    return xs[-1]


def _max(xs: list[int]) -> int:
    if not xs:
        raise DSLError("max of empty list")
    return max(xs)


def _min(xs: list[int]) -> int:
    if not xs:
        raise DSLError("min of empty list")
    return min(xs)


def _mod(a: int, b: int) -> int:
    if b == 0:
        raise DSLError("mod by zero")
    return a % b


def _prod_mod(xs: list[int]) -> int:
    p = 1
    for v in xs:
        p = (p * v) % 1000
    return p


def _take_while(p: Callable[[int], bool], xs: list[int]) -> list[int]:
    out = []
    for v in xs:
        if not p(v):
            break
        out.append(v)
    return out


def _drop_while(p: Callable[[int], bool], xs: list[int]) -> list[int]:
    i = 0
    while i < len(xs) and p(xs[i]):
        i += 1
    return xs[i:]


def _check_int(v: int) -> int:
    if abs(v) > 10**9:
        raise DSLError("integer overflow bound exceeded")
    return v


# name → (arg types, return type, implementation)
PRIMITIVES: dict[str, tuple[tuple[str, ...], str, Callable[..., Any]]] = {
    # list → list
    "reverse": (("list",), "list", lambda xs: xs[::-1]),
    "sort": (("list",), "list", sorted),
    "sort_desc": (("list",), "list", lambda xs: sorted(xs, reverse=True)),
    "unique": (("list",), "list", lambda xs: list(dict.fromkeys(xs))),
    "tail": (("list",), "list", lambda xs: xs[1:]),
    "init": (("list",), "list", lambda xs: xs[:-1]),
    "scan_sum": (("list",), "list", lambda xs: [sum(xs[: i + 1]) for i in range(len(xs))]),
    "evens": (("list",), "list", lambda xs: xs[::2]),
    "odds": (("list",), "list", lambda xs: xs[1::2]),
    "rotate_left": (("list",), "list", lambda xs: xs[1:] + xs[:1]),
    "rotate_right": (("list",), "list", lambda xs: xs[-1:] + xs[:-1]),
    # higher order
    "map": (("fn", "list"), "list", lambda f, xs: [_check_int(f(v)) for v in xs]),
    "filter": (("pred", "list"), "list", lambda p, xs: [v for v in xs if p(v)]),
    "take_while": (("pred", "list"), "list", _take_while),
    "drop_while": (("pred", "list"), "list", _drop_while),
    "count": (("pred", "list"), "int", lambda p, xs: sum(1 for v in xs if p(v))),
    # int × list → list
    "take": (("int", "list"), "list", lambda n, xs: xs[: max(n, 0)]),
    "drop": (("int", "list"), "list", lambda n, xs: xs[max(n, 0) :]),
    # list × list → list
    "concat": (("list", "list"), "list", lambda a, b: a + b),
    "zip_add": (("list", "list"), "list", lambda a, b: [x + y for x, y in zip(a, b)]),
    "zip_max": (("list", "list"), "list", lambda a, b: [max(x, y) for x, y in zip(a, b)]),
    # list → int
    "length": (("list",), "int", len),
    "sum": (("list",), "int", sum),
    "max": (("list",), "int", _max),
    "min": (("list",), "int", _min),
    "head": (("list",), "int", _head),
    "last": (("list",), "int", _last),
    "prod_mod": (("list",), "int", _prod_mod),
    "count_distinct": (("list",), "int", lambda xs: len(set(xs))),
    # int × int → int
    "add": (("int", "int"), "int", lambda a, b: _check_int(a + b)),
    "sub": (("int", "int"), "int", lambda a, b: _check_int(a - b)),
    "mul": (("int", "int"), "int", lambda a, b: _check_int(a * b)),
    "max2": (("int", "int"), "int", max),
    "min2": (("int", "int"), "int", min),
    "mod": (("int", "int"), "int", _mod),
    # functions int → int
    "inc": ((), "fn", lambda: (lambda v: v + 1)),
    "dec": ((), "fn", lambda: (lambda v: v - 1)),
    "double": ((), "fn", lambda: (lambda v: v * 2)),
    "square": ((), "fn", lambda: (lambda v: v * v)),
    "negate": ((), "fn", lambda: (lambda v: -v)),
    "abs": ((), "fn", lambda: abs),
    "halve": ((), "fn", lambda: (lambda v: v // 2)),
    "mod3": ((), "fn", lambda: (lambda v: v % 3)),
    "add_n": (("int",), "fn", lambda n: (lambda v: v + n)),
    "mul_n": (("int",), "fn", lambda n: (lambda v: v * n)),
    # predicates int → bool
    "is_even": ((), "pred", lambda: (lambda v: v % 2 == 0)),
    "is_odd": ((), "pred", lambda: (lambda v: v % 2 == 1)),
    "is_pos": ((), "pred", lambda: (lambda v: v > 0)),
    "is_neg": ((), "pred", lambda: (lambda v: v < 0)),
    "gt_n": (("int",), "pred", lambda n: (lambda v: v > n)),
    "lt_n": (("int",), "pred", lambda n: (lambda v: v < n)),
}

VARIABLES = {"xs": "list", "k": "int"}


def type_of(node: Node) -> str:
    if node.op == "lit":
        return "int"
    if node.op in VARIABLES:
        return VARIABLES[node.op]
    if node.op not in PRIMITIVES:
        raise DSLError(f"unknown op {node.op}")
    arg_types, ret, _ = PRIMITIVES[node.op]
    if len(arg_types) != len(node.args):
        raise DSLError(f"{node.op} expects {len(arg_types)} args")
    for t, a in zip(arg_types, node.args):
        if type_of(a) != t:
            raise DSLError(f"type error in {node.op}: expected {t}, got {type_of(a)}")
    return ret


def evaluate(node: Node, env: dict[str, Any]) -> Any:
    if node.op == "lit":
        return node.value
    if node.op in VARIABLES:
        return env[node.op]
    _, _, impl = PRIMITIVES[node.op]
    return impl(*[evaluate(a, env) for a in node.args])


def run(program: Node, xs: list[int], k: int) -> Any:
    return evaluate(program, {"xs": list(xs), "k": k})


# --- Python-subset printer (02 §14.2 F0 Python subset) -------------------------------------------
class _Printer:
    def __init__(self) -> None:
        self.helpers: dict[str, str] = {}
        self.depth = 0

    def fresh(self) -> str:
        self.depth += 1
        return f"v{self.depth}"

    def fn(self, node: Node, var: str) -> str:
        n = lambda i: self.expr(node.args[i])  # noqa: E731
        table = {
            "inc": f"({var} + 1)", "dec": f"({var} - 1)", "double": f"({var} * 2)", "square": f"({var} * {var})",
            "negate": f"(-{var})", "abs": f"abs({var})", "halve": f"({var} // 2)", "mod3": f"({var} % 3)",
            "is_even": f"({var} % 2 == 0)", "is_odd": f"({var} % 2 == 1)", "is_pos": f"({var} > 0)",
            "is_neg": f"({var} < 0)",
        }
        if node.op in table:
            return table[node.op]
        if node.op == "add_n":
            return f"({var} + {n(0)})"
        if node.op == "mul_n":
            return f"({var} * {n(0)})"
        if node.op == "gt_n":
            return f"({var} > {n(0)})"
        if node.op == "lt_n":
            return f"({var} < {n(0)})"
        raise DSLError(f"not a function node: {node.op}")

    def helper(self, name: str, body: str) -> str:
        self.helpers.setdefault(name, body)
        return name

    def expr(self, node: Node) -> str:
        op, a = node.op, node.args
        if op == "lit":
            return str(node.value)
        if op in VARIABLES:
            return op
        e = lambda i: self.expr(a[i])  # noqa: E731
        simple = {
            "reverse": lambda: f"{e(0)}[::-1]", "sort": lambda: f"sorted({e(0)})",
            "sort_desc": lambda: f"sorted({e(0)}, reverse=True)", "unique": lambda: f"list(dict.fromkeys({e(0)}))",
            "tail": lambda: f"{e(0)}[1:]", "init": lambda: f"{e(0)}[:-1]", "evens": lambda: f"{e(0)}[::2]",
            "odds": lambda: f"{e(0)}[1::2]", "length": lambda: f"len({e(0)})", "sum": lambda: f"sum({e(0)})",
            "max": lambda: f"max({e(0)})", "min": lambda: f"min({e(0)})", "head": lambda: f"{e(0)}[0]",
            "last": lambda: f"{e(0)}[-1]", "count_distinct": lambda: f"len(set({e(0)}))",
            "add": lambda: f"({e(0)} + {e(1)})", "sub": lambda: f"({e(0)} - {e(1)})",
            "mul": lambda: f"({e(0)} * {e(1)})", "max2": lambda: f"max({e(0)}, {e(1)})",
            "min2": lambda: f"min({e(0)}, {e(1)})", "mod": lambda: f"({e(0)} % {e(1)})",
            "take": lambda: f"{e(1)}[:max({e(0)}, 0)]", "drop": lambda: f"{e(1)}[max({e(0)}, 0):]",
            "concat": lambda: f"({e(0)} + {e(1)})",
        }
        if op in simple:
            return simple[op]()
        if op == "rotate_left":
            x = e(0)
            return f"({x}[1:] + {x}[:1])"
        if op == "rotate_right":
            x = e(0)
            return f"({x}[-1:] + {x}[:-1])"
        if op == "scan_sum":
            x = e(0)
            i = self.fresh()
            return f"[sum({x}[:{i} + 1]) for {i} in range(len({x}))]"
        if op == "zip_add":
            p, q = self.fresh(), self.fresh()
            return f"[{p} + {q} for {p}, {q} in zip({e(0)}, {e(1)})]"
        if op == "zip_max":
            p, q = self.fresh(), self.fresh()
            return f"[max({p}, {q}) for {p}, {q} in zip({e(0)}, {e(1)})]"
        if op == "map":
            v = self.fresh()
            return f"[{self.fn(a[0], v)} for {v} in {e(1)}]"
        if op == "filter":
            v = self.fresh()
            return f"[{v} for {v} in {e(1)} if {self.fn(a[0], v)}]"
        if op == "count":
            v = self.fresh()
            return f"len([{v} for {v} in {e(1)} if {self.fn(a[0], v)}])"
        if op == "prod_mod":
            self.helper("_prod_mod", "def _prod_mod(values: list[int]) -> int:\n    p = 1\n    for v in values:\n        p = (p * v) % 1000\n    return p\n")
            return f"_prod_mod({e(0)})"
        if op in ("take_while", "drop_while"):
            v = self.fresh()
            cond = self.fn(a[0], v)
            name = f"_{op}_{hashlib.sha256((op + cond).encode()).hexdigest()[:8]}"
            if op == "take_while":
                body = (f"def {name}(values: list[int], k: int) -> list[int]:\n    out = []\n    for {v} in values:\n"
                        f"        if not {cond}:\n            break\n        out.append({v})\n    return out\n")
            else:
                body = (f"def {name}(values: list[int], k: int) -> list[int]:\n    i = 0\n"
                        f"    while i < len(values):\n        {v} = values[i]\n        if not {cond}:\n            break\n"
                        f"        i += 1\n    return values[i:]\n")
            self.helper(name, body)
            return f"{name}({e(1)}, k)"
        raise DSLError(f"cannot print op {op}")


def to_python(program: Node, fn_name: str = "solve") -> str:
    """Render a program as an F0 Python-subset module defining ``solve(xs, k)``."""
    p = _Printer()
    body = p.expr(program)
    ret = "int" if type_of(program) == "int" else "list[int]"
    helpers = "".join(h + "\n" for _, h in sorted(p.helpers.items()))
    return f"{helpers}def {fn_name}(xs: list[int], k: int) -> {ret}:\n    return {body}\n"


# --- sampler -------------------------------------------------------------------------------------
_BY_RETURN: dict[str, list[str]] = {}
for _name, (_args, _ret, _impl) in PRIMITIVES.items():
    _BY_RETURN.setdefault(_ret, []).append(_name)


def sample_program(rng: np.random.Generator, ret: str, max_depth: int) -> Node:
    """Random well-typed program of the given return type."""
    if max_depth <= 1 or (ret in ("list", "int") and rng.random() < 0.25):
        if ret == "list":
            return Node("xs")
        if ret == "int":
            return Node("k") if rng.random() < 0.5 else Node("lit", (), int(rng.integers(0, 6)))
    candidates = _BY_RETURN[ret]
    if max_depth <= 2:
        candidates = [c for c in candidates if all(t in ("list", "int") for t in PRIMITIVES[c][0]) or not PRIMITIVES[c][0]]
        if not candidates:
            candidates = _BY_RETURN[ret]
    op = candidates[int(rng.integers(len(candidates)))]
    arg_types = PRIMITIVES[op][0]
    return Node(op, tuple(sample_program(rng, t, max_depth - 1) for t in arg_types))


def random_input(rng: np.random.Generator, max_len: int = 8) -> tuple[list[int], int]:
    n = int(rng.integers(0, max_len + 1))
    return [int(v) for v in rng.integers(-9, 10, size=n)], int(rng.integers(-3, 6))


def behavior(program: Node, inputs: list[tuple[list[int], int]]) -> tuple[Any, ...] | None:
    outs = []
    for xs, k in inputs:
        try:
            outs.append(_freeze(run(program, xs, k)))
        except DSLError:
            return None
    return tuple(outs)


def _freeze(v: Any) -> Any:
    return tuple(v) if isinstance(v, list) else v
