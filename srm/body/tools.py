"""Body tool interface (07 §6.1) and the F0 tools (07 §6.2).

Tool results are OBSERVED evidence (trust class ``tool_execution``) and act as
promotion events (06 §2.3).  A tool error is itself an observation.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from srm.body import dsl


@dataclass
class ToolResult:
    tool: str
    ok: bool
    value: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str
    input_schema: str
    output_schema: str
    sandbox: str
    cost: float
    deterministic: bool = True
    trust_class: str = "tool_execution"


class Tool:
    spec: ToolSpec

    def __call__(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


# --- DSL interpreter ---------------------------------------------------------------------------------
class DSLTool(Tool):
    spec = ToolSpec("dsl", "1.0", "{program, xs, k}", "value", "none", 0.01)

    def __call__(self, program: Any, xs: list[int], k: int) -> ToolResult:
        node = program if isinstance(program, dsl.Node) else dsl.Node.from_json(program)
        try:
            dsl.type_of(node)
            return ToolResult("dsl", True, dsl.run(node, xs, k))
        except dsl.DSLError as exc:
            return ToolResult("dsl", False, error=f"DSLError: {exc}")


# --- Python subset validator + sandbox (02 §14.2, 07 §6.2) ---------------------------------------------
ALLOWED_BUILTINS = frozenset(
    {"len", "range", "sorted", "min", "max", "sum", "abs", "enumerate", "zip", "set", "reversed", "any", "all",
     "list", "dict", "tuple", "int", "bool", "str"}
)
_ALLOWED_NODES = (
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.For, ast.While, ast.If, ast.Break, ast.Continue, ast.Pass, ast.Expr, ast.Name, ast.Load, ast.Store,
    ast.Constant, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Call, ast.Subscript, ast.Slice, ast.List,
    ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp, ast.comprehension,
    ast.IfExp, ast.keyword, ast.Attribute, ast.Raise,
    ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Pow, ast.Div, ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
)
_ALLOWED_METHODS = frozenset({"append", "extend", "pop", "insert", "index", "count", "keys", "values", "items",
                              "get", "sort", "reverse", "copy", "fromkeys"})


class SubsetViolation(ValueError):
    pass


def validate_python_subset(source: str) -> ast.Module:
    """Reject anything outside the F0 Python subset before execution."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SubsetViolation(f"syntax error: {exc}") from exc
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise SubsetViolation(f"construct not allowed: {type(node).__name__}")
        if isinstance(node, ast.Attribute) and (node.attr not in _ALLOWED_METHODS or node.attr.startswith("_")):
            raise SubsetViolation(f"attribute not allowed: {node.attr}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SubsetViolation(f"name not allowed: {node.id}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in ALLOWED_BUILTINS and node.func.id not in defined:
                local_names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                if node.func.id not in local_names:
                    raise SubsetViolation(f"call not allowed: {node.func.id}")
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if not (isinstance(exc, ast.Name) and exc.id == "ValueError"):
                raise SubsetViolation("only ValueError may be raised")
    return tree


_RUNNER = textwrap.dedent(
    """
    import json, resource, sys
    limit = int(sys.argv[1]) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        pass
    payload = json.loads(sys.stdin.read())
    allowed = payload["allowed"]
    import builtins
    safe = {name: getattr(builtins, name) for name in allowed}
    safe["ValueError"] = ValueError
    ns = {"__builtins__": safe}
    out = []
    try:
        exec(compile(payload["source"], "<candidate>", "exec"), ns)
        fn = ns[payload["entry"]]
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"fatal": f"{type(exc).__name__}: {exc}"}))
        sys.exit(0)
    for case in payload["cases"]:
        try:
            out.append({"ok": True, "value": fn(*case)})
        except Exception as exc:  # noqa: BLE001
            out.append({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    print(json.dumps({"results": out}))
    """
)


class PythonSandbox(Tool):
    """Runs validated subset code in a separate interpreter with time and memory limits."""

    spec = ToolSpec("python_sandbox", "1.0", "{source, entry, cases}", "results", "process", 0.5)

    def __init__(self, timeout_s: float = 2.0, memory_mb: int = 256) -> None:
        self.timeout_s, self.memory_mb = timeout_s, memory_mb

    def __call__(self, source: str, entry: str, cases: list[list[Any]]) -> ToolResult:
        try:
            validate_python_subset(source)
        except SubsetViolation as exc:
            return ToolResult("python_sandbox", False, error=f"SubsetViolation: {exc}")
        payload = json.dumps({"source": source, "entry": entry, "cases": cases, "allowed": sorted(ALLOWED_BUILTINS)})
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _RUNNER, str(self.memory_mb)], input=payload, capture_output=True,
                text=True, timeout=self.timeout_s, env=env,
            )
        except subprocess.TimeoutExpired:
            return ToolResult("python_sandbox", False, error="Timeout")
        try:
            data = json.loads(proc.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            return ToolResult("python_sandbox", False, error=f"SandboxError: {proc.stderr.strip()[-300:]}")
        if "fatal" in data:
            return ToolResult("python_sandbox", False, error=data["fatal"])
        return ToolResult("python_sandbox", True, data["results"])


class TestRunner(Tool):
    """Runs unit tests ``[{xs, k, out}]`` against a DSL program or Python source (07 §6.2)."""

    spec = ToolSpec("tests", "1.0", "{artifact, tests}", "{passed, failures}", "process", 0.5)

    def __init__(self, sandbox: PythonSandbox) -> None:
        self.sandbox = sandbox

    def __call__(self, tests: list[dict[str, Any]], program: Any = None, source: str | None = None,
                 entry: str = "solve") -> ToolResult:
        failures: list[dict[str, Any]] = []
        if program is not None:
            node = program if isinstance(program, dsl.Node) else dsl.Node.from_json(program)
            for t in tests:
                try:
                    got = dsl.run(node, t["xs"], t["k"])
                except dsl.DSLError as exc:
                    failures.append({"case": t, "error": str(exc)})
                    continue
                if got != t["out"]:
                    failures.append({"case": t, "got": got})
        else:
            res = self.sandbox(source or "", entry, [[t["xs"], t["k"]] for t in tests])
            if not res.ok:
                return ToolResult("tests", False, {"passed": 0, "failures": [{"error": res.error}]}, error=res.error)
            for t, r in zip(tests, res.value):
                if not r["ok"]:
                    failures.append({"case": t, "error": r["error"]})
                elif r["value"] != t["out"]:
                    failures.append({"case": t, "got": r["value"]})
        passed = len(tests) - len(failures)
        return ToolResult("tests", not failures, {"passed": passed, "total": len(tests), "failures": failures})


# --- SymPy and units -----------------------------------------------------------------------------------
class SympyTool(Tool):
    spec = ToolSpec("sympy", "1.0", "{op, expr, ...}", "value", "none", 0.1)

    def __call__(self, op: str, expr: str, other: str | None = None, subs: dict[str, float] | None = None) -> ToolResult:
        try:
            e = sp.sympify(expr.replace("^", "**"))
            if op == "evaluate":
                return ToolResult("sympy", True, float(e.subs(subs or {})))
            if op == "expand":
                return ToolResult("sympy", True, sp.sstr(sp.expand(e)))
            if op == "factor":
                return ToolResult("sympy", True, sp.sstr(sp.factor(e)))
            if op == "equivalent":
                o = sp.sympify((other or "0").replace("^", "**"))
                return ToolResult("sympy", True, bool(sp.simplify(e - o) == 0))
            return ToolResult("sympy", False, error=f"unknown op {op}")
        except (sp.SympifyError, TypeError, ValueError) as exc:
            return ToolResult("sympy", False, error=f"{type(exc).__name__}: {exc}")


_UNIT_FACTORS = {
    "m": ("length", 1.0), "km": ("length", 1000.0), "cm": ("length", 0.01), "mm": ("length", 0.001),
    "s": ("time", 1.0), "min": ("time", 60.0), "h": ("time", 3600.0), "hour": ("time", 3600.0),
    "kg": ("mass", 1.0), "g": ("mass", 0.001), "t": ("mass", 1000.0),
    "m/s": ("speed", 1.0), "km/h": ("speed", 1000.0 / 3600.0),
    "km2": ("area", 1e6), "m2": ("area", 1.0),
}


class UnitsTool(Tool):
    spec = ToolSpec("units", "1.0", "{value, from, to}", "value", "none", 0.01)

    def __call__(self, value: float, from_unit: str, to_unit: str) -> ToolResult:
        a, b = _UNIT_FACTORS.get(from_unit), _UNIT_FACTORS.get(to_unit)
        if a is None or b is None:
            return ToolResult("units", False, error=f"unknown unit {from_unit if a is None else to_unit}")
        if a[0] != b[0]:
            return ToolResult("units", False, error=f"incompatible dimensions {a[0]} vs {b[0]}")
        return ToolResult("units", True, value * a[1] / b[1])


class Body:
    """The set of enabled tools (Build Configuration ``body.tools``)."""

    def __init__(self, tools: tuple[str, ...] = ("dsl", "python_sandbox", "tests", "sympy", "units"),
                 timeout_s: float = 2.0, memory_mb: int = 256) -> None:
        sandbox = PythonSandbox(timeout_s, memory_mb)
        available: dict[str, Tool] = {
            "dsl": DSLTool(), "python_sandbox": sandbox, "tests": TestRunner(sandbox),
            "sympy": SympyTool(), "units": UnitsTool(),
        }
        self.tools = {name: available[name] for name in tools}
        self.calls: list[dict[str, Any]] = []

    def call(self, name: str, **kwargs: Any) -> ToolResult:
        if name not in self.tools:
            return ToolResult(name, False, error=f"tool {name} not enabled")
        result = self.tools[name](**kwargs)
        self.calls.append({"tool": name, "ok": result.ok, "error": result.error})
        return result

    def cost(self, name: str) -> float:
        return self.tools[name].spec.cost
