"""Structural parsers (03 §8) and request intake (03 §9).

* controlled-language questions (FactStream style, including nested "the R of the R2 of X")
* math word problems and algebra requests (routed to stored procedures / SymPy)
* DSL s-expressions and Python-subset source (deterministic; ``ast``)
* structured task requests (examples)

The learned text parser (S2) will replace the rule-based question parser for open text;
these deterministic parsers remain for formal domains (D-009).
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from srm.body import dsl
from srm.data.sef import QuestionRecord, TaskRecord
from srm.interface.codec import normalize_surface
from srm.interface.messages import PatternAtom, Qualifiers
from srm.interface.registries import RelationRegistry


@dataclass
class Request:
    kind: str  # factual | dsl_task | code_task | math_word | algebra | unknown
    text: str = ""
    atoms: list[PatternAtom] = field(default_factory=list)
    qualifiers: Qualifiers = field(default_factory=Qualifiers)
    examples: list[dict[str, Any]] = field(default_factory=list)
    ret_type: str = "list"
    output_format: str = "prose"
    algebra_op: str = ""
    expr: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


# --- controlled-language questions ----------------------------------------------------------------
_Q = re.compile(r"^\s*(?:what|who|which)\s+is\s+the\s+(?P<body>.+?)\s*\?\s*$", re.IGNORECASE)
_TIME = re.compile(r"\s+in\s+(?P<year>\d{3,4})$", re.IGNORECASE)


def relation_lexicon(relations: RelationRegistry) -> dict[str, str]:
    """Surface phrase → relation id (names with underscores read as spaces)."""
    lex = {}
    for entry in relations.entries():
        for phrase in (entry.name, entry.id.split(":", 1)[-1], *entry.aliases):
            lex[normalize_surface(phrase.replace("_", " "))] = entry.id
    return lex


def parse_question(text: str, lexicon: dict[str, str]) -> tuple[list[PatternAtom], Qualifiers] | None:
    """'What is the R1 of the R2 of X [in YEAR]?' → atoms (X R2 ?h0)(?h0 R1 ?x)."""
    m = _Q.match(text)
    if not m:
        return None
    body = m.group("body").strip()
    quals = Qualifiers()
    t = _TIME.search(body)
    if t:
        year = float(t.group("year"))
        quals = Qualifiers(time=(year, year + 1.0))
        body = body[: t.start()]
    parts = re.split(r"\s+of\s+(?:the\s+)?", body)
    if len(parts) < 2:
        return None
    entity = parts[-1].strip()
    rel_phrases = parts[:-1]
    rels = []
    for phrase in rel_phrases:
        rid = lexicon.get(normalize_surface(phrase))
        if rid is None:
            return None
        rels.append(rid)
    rels = list(reversed(rels))  # innermost relation applies first
    atoms, var = [], f"@{entity}"
    for i, rid in enumerate(rels):
        nxt = "?x" if i == len(rels) - 1 else f"?h{i}"
        atoms.append(PatternAtom(var, rid, nxt))
        var = nxt
    return atoms, quals


# --- math and algebra ------------------------------------------------------------------------------
_ALGEBRA = re.compile(r"^\s*(?P<op>expand|factor)\s+(?P<expr>.+?)\.?\s*$", re.IGNORECASE)


def parse_math(text: str) -> Request | None:
    m = _ALGEBRA.match(text)
    if m:
        return Request("algebra", text, algebra_op=m.group("op").lower(), expr=m.group("expr").strip())
    if re.search(r"\d", text) and text.strip().endswith("?") and re.search(r"\b(how|what)\b", text, re.IGNORECASE):
        return Request("math_word", text)
    return None


# --- DSL and Python --------------------------------------------------------------------------------
def parse_dsl(source: str) -> dsl.Node:
    """S-expression → DSL node, e.g. ``(map inc (filter is_even xs))``."""
    tokens = source.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def read() -> dsl.Node:
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            op = tokens[pos]
            pos += 1
            args = []
            while tokens[pos] != ")":
                args.append(read())
            pos += 1
            return dsl.Node(op, tuple(args))
        if re.fullmatch(r"-?\d+", tok):
            return dsl.Node("lit", (), int(tok))
        return dsl.Node(tok)

    node = read()
    if pos != len(tokens):
        raise ValueError("trailing tokens in DSL source")
    dsl.type_of(node)
    return node


@dataclass
class CodeSummary:
    functions: list[str]
    calls: list[tuple[str, str]]  # (caller, callee)
    defines: dict[str, list[str]]  # function → parameter names


def parse_python(source: str) -> CodeSummary:
    """Deterministic code parser (03 §8.2): functions, parameters and call edges."""
    tree = ast.parse(source)
    funcs, calls, params = [], [], {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
            params[node.name] = [a.arg for a in node.args.args]
            for sub in ast.walk(node):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                    calls.append((node.name, sub.func.id))
    return CodeSummary(funcs, calls, params)


# --- intake (03 §9) --------------------------------------------------------------------------------
def intake(request: Any, lexicon: dict[str, str]) -> Request:
    """Turn a user request (text, SEF question/task record, or dict) into a typed Request."""
    if isinstance(request, Request):
        return request
    if isinstance(request, QuestionRecord):
        if request.pattern is not None and request.pattern.atoms:
            atoms = [PatternAtom(a.subject, a.relation, a.object if isinstance(a.object, str) else a.object.value)
                     for a in request.pattern.atoms]
            quals = Qualifiers(time=request.context.time, version=request.context.version,
                               world_id=request.context.world_id)
            return Request("factual", request.query, atoms, quals)
        return intake(request.query, lexicon)
    if isinstance(request, TaskRecord):
        if request.family.startswith("dsl/"):
            ret = "list" if request.goal.get("output_type") == "list[int]" else "int"
            return Request("dsl_task", examples=list(request.inputs["examples"]), ret_type=ret, output_format="dsl",
                           meta={"task_id": request.task_id})
        if request.family.startswith("math/"):
            return intake(request.inputs["text"], lexicon)
        return Request("unknown", meta={"task_id": request.task_id})
    if isinstance(request, dict):
        if "examples" in request:
            out = request.get("output_type", "list")
            fmt = request.get("format", "python")
            return Request("code_task" if fmt == "python" else "dsl_task", examples=list(request["examples"]),
                           ret_type="int" if out in ("int", "number") else "list", output_format=fmt)
        if "text" in request:
            return intake(request["text"], lexicon)
    if isinstance(request, str):
        parsed = parse_question(request, lexicon)
        if parsed is not None:
            atoms, quals = parsed
            return Request("factual", request, atoms, quals)
        m = parse_math(request)
        if m is not None:
            return m
        return Request("unknown", request)
    raise TypeError(f"unsupported request type {type(request).__name__}")
