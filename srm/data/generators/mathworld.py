"""MathWorld-lite generator (02 §14.4): templated arithmetic word problems with symbolic
verification, algebra identities, and perturbed variants (number/name changes, distractors)."""

from __future__ import annotations

from typing import Callable

import sympy as sp

from srm.data.generators.common import GenContext
from srm.data.io import Dataset
from srm.data.sef import TaskRecord, Verifier

VERSION = "0.1"
CATEGORY = "synthetic/mathworld"
ITEMS = ("apple", "book", "brick", "ticket", "lamp", "coin", "plank", "jar")

Template = Callable[[GenContext, str, str, bool], tuple[str, float, dict[str, float], str]]


def _rate(ctx: GenContext, name: str, item: str, distract: bool) -> tuple[str, float, dict[str, float], str]:
    r, t = int(ctx.rng.integers(10, 120)), int(ctx.rng.integers(1, 9))
    text = f"{name} travels at {r} km per hour for {t} hours."
    if distract:
        text += f" {name} also owns {int(ctx.rng.integers(2, 9))} {item}s."
    return text + f" How far does {name} travel?", float(r * t), {"r": r, "t": t}, "r*t"


def _cost(ctx: GenContext, name: str, item: str, distract: bool) -> tuple[str, float, dict[str, float], str]:
    n, p = int(ctx.rng.integers(2, 20)), int(ctx.rng.integers(1, 30))
    text = f"{name} buys {n} {item}s at {p} dollars each."
    if distract:
        text += f" The shop opened {int(ctx.rng.integers(2, 30))} years ago."
    return text + f" How much does {name} pay?", float(n * p), {"n": n, "p": p}, "n*p"


def _share(ctx: GenContext, name: str, item: str, distract: bool) -> tuple[str, float, dict[str, float], str]:
    k = int(ctx.rng.integers(2, 9))
    n = k * int(ctx.rng.integers(2, 15))
    text = f"{n} {item}s are shared equally among {k} friends of {name}."
    if distract:
        text += f" {name} is {int(ctx.rng.integers(8, 70))} years old."
    return text + " How many does each friend get?", float(n // k), {"n": n, "k": k}, "n/k"


def _work(ctx: GenContext, name: str, item: str, distract: bool) -> tuple[str, float, dict[str, float], str]:
    b = int(ctx.rng.integers(2, 8))
    d = b * int(ctx.rng.integers(1, 6))
    a = int(ctx.rng.integers(2, 9))
    text = f"{a} workers build a wall in {d} days."
    if distract:
        text += f" Each worker carries {int(ctx.rng.integers(2, 9))} {item}s."
    return text + f" How many days do {a * b} workers need?", float(d / b), {"a": a, "d": d, "w": a * b}, "a*d/w"


def _discount(ctx: GenContext, name: str, item: str, distract: bool) -> tuple[str, float, dict[str, float], str]:
    p = 20 * int(ctx.rng.integers(1, 20))
    q = 5 * int(ctx.rng.integers(1, 10))
    text = f"A {item} costs {p} dollars and is discounted by {q} percent."
    if distract:
        text += f" {name} has {int(ctx.rng.integers(1, 9))} friends."
    return text + " What is the new price?", p * (100 - q) / 100.0, {"p": p, "q": q}, "p*(100-q)/100"


def _two_step(ctx: GenContext, name: str, item: str, distract: bool) -> tuple[str, float, dict[str, float], str]:
    a, b = int(ctx.rng.integers(5, 40)), int(ctx.rng.integers(1, 20))
    c = int(ctx.rng.integers(1, a + b))
    text = f"{name} has {a} {item}s, buys {b} more, then gives away {c}."
    if distract:
        text += f" The {item}s are kept in {int(ctx.rng.integers(2, 6))} boxes."
    return text + f" How many {item}s are left?", float(a + b - c), {"a": a, "b": b, "c": c}, "a+b-c"


TEMPLATES: dict[str, Template] = {
    "rate": _rate, "cost": _cost, "share": _share, "work": _work, "discount": _discount, "two_step": _two_step,
}


def generate(dataset_id: str = "math", seed: int = 0, n_problems: int = 1_000, perturbed_fraction: float = 0.1,
             algebra_fraction: float = 0.2) -> Dataset:
    ctx = GenContext(dataset_id, seed, "mathworld", VERSION)
    rng = ctx.rng
    src = ctx.source("gen", "generator_truth", category=CATEGORY)
    x = sp.Symbol("x")
    names = list(TEMPLATES)
    for i in range(n_problems):
        if rng.random() < algebra_fraction:
            a, b = int(rng.integers(-9, 10)), int(rng.integers(-9, 10))
            if rng.random() < 0.5:
                expr = (x + a) * (x + b)
                question, answer, family = f"Expand ({sp.sstr(x + a)})*({sp.sstr(x + b)}).", sp.sstr(sp.expand(expr)), "math/algebra/expand"
            else:
                expr = sp.expand((x + a) * (x + b))
                question, answer, family = f"Factor {sp.sstr(expr)}.", sp.sstr(sp.factor(expr)), "math/algebra/factor"
            split = "train" if rng.random() < 0.8 else "dev"
            ctx.add(TaskRecord(
                record_id=ctx.rid("task"), source_id=src, data_category=CATEGORY, extraction=ctx.extraction(),
                task_id=ctx.rid("task_id"), family=family, goal={"output_type": "expr"}, inputs={"text": question},
                verifier=Verifier(type="reference_compare", spec={"answer": answer, "symbol": "x"}), split=split,
            ))
            continue
        tname = names[int(rng.integers(len(names)))]
        name = ctx.names.make((2, 2))
        item = ITEMS[int(rng.integers(len(ITEMS)))]
        perturbed = rng.random() < perturbed_fraction
        text, answer, quantities, formula = TEMPLATES[tname](ctx, name, item, perturbed)
        split = "test_perturbed" if perturbed else ("train" if rng.random() < 0.8 else ("dev" if rng.random() < 0.5 else "test"))
        ctx.add(TaskRecord(
            record_id=ctx.rid("task"), source_id=src, data_category=CATEGORY, extraction=ctx.extraction(),
            task_id=ctx.rid("task_id"), family=f"math/word/{tname}", goal={"output_type": "number"},
            inputs={"text": text},
            verifier=Verifier(type="tolerance", spec={"answer": answer, "tolerance": 1e-6}),
            reference_solution={"formula": formula, "quantities": quantities}, split=split,
        ))
    return ctx.dataset
