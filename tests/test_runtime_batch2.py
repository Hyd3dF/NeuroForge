"""Batch 2 integration: the full beat loop (11 §2) across factual (incl. multi-hop and text),
DSL/Python programming, math word problems, algebra, budgets and online library learning."""

from __future__ import annotations

from collections import Counter

import pytest
import sympy as sp

from srm.body import dsl
from srm.config import tiny
from srm.data.generators import dslworld, factstream, mathworld
from srm.interface.messages import DecisionClass
from srm.interface.values import Value
from srm.mouth.mouth import _exact_number
from srm.runtime import SRM


@pytest.fixture(scope="module")
def srm():
    m = SRM(tiny())
    m.ingest(factstream.generate("fs", 21, n_countries=30))
    m.ingest(mathworld.generate("math", 4, n_problems=80))
    return m


def gold_value(g) -> Value:
    return Value(g.type, g.value, g.unit, float(g.tolerance or 0.0))


def test_factual_questions_including_multihop(srm) -> None:
    ds = factstream.generate("fs", 21, n_countries=30)
    seen = Counter()
    for q in ds.by_kind("question"):
        r = srm.ask(q)
        assert r.cls.value == q.gold_state, (q.query, r.decision.reason)
        if q.gold_state == "KNOWN":
            assert r.answer.matches(gold_value(q.gold_answer)), q.query
        seen[q.split] += 1
        assert r.accounting["active_record_fraction"] < 0.05  # sparse recruitment
    assert seen["dev_multihop"] > 0


def test_text_questions_parse_to_the_same_answers(srm) -> None:
    ds = factstream.generate("fs", 21, n_countries=30)
    for q in [q for q in ds.by_kind("question") if q.gold_state == "KNOWN"][:60]:
        if q.context.time is not None:
            continue
        r = srm.ask(q.query)
        assert r.cls == DecisionClass.KNOWN, q.query
        assert r.answer.matches(gold_value(q.gold_answer))
        shown = srm.name_of(str(r.answer.value)) if r.answer.type == "entity_ref" else _exact_number(r.answer.value)
        assert r.text.endswith(".") and shown in r.text  # literal values are copied exactly


def test_unknown_and_unsupported_requests(srm) -> None:
    r = srm.ask("What is the capital of Nowhereland?")
    assert r.cls == DecisionClass.UNKNOWN_ABSENT and "don't know" in r.text
    r = srm.ask("Sing me a song")
    assert r.cls == DecisionClass.UNKNOWN_NO_BASIS


def test_math_word_problems_and_algebra(srm) -> None:
    ds = mathworld.generate("math", 4, n_problems=80)
    for t in ds.by_kind("task"):
        r = srm.ask(t)
        assert r.cls == DecisionClass.KNOWN, t.inputs["text"]
        if t.verifier.type == "tolerance":
            assert abs(float(r.answer.value) - t.verifier.spec["answer"]) < 1e-6, t.inputs["text"]
        else:
            assert sp.simplify(sp.sympify(str(r.answer.value)) - sp.sympify(t.verifier.spec["answer"])) == 0
        assert r.answer_record.circuit_trace and r.accounting["tool_calls"] >= 1
    r = srm.ask("Zed rides a carpet to the moon. How many clouds are there?")
    assert r.cls == DecisionClass.UNKNOWN_NO_BASIS  # no stored procedure applies → no fabricated number


def test_dsl_tasks_and_generalization(srm) -> None:
    ds = dslworld.generate("dsl", 6, n_tasks=60)
    tasks = [t for t in ds.by_kind("task") if t.split == "train"][:15]
    solved = 0
    for t in tasks:
        r = srm.ask(t, max_beats=40)
        if r.artifacts.get("programs") and r.cls in (DecisionClass.KNOWN, DecisionClass.AMBIGUOUS):
            prog = dsl.Node.from_json(r.decision.best and __import__("json").loads(r.decision.best.binding.value))
            assert all(dsl.run(prog, e["xs"], e["k"]) == e["out"] for e in t.inputs["examples"])  # TESTED on examples
            solved += all(dsl.run(prog, e["xs"], e["k"]) == e["out"] for e in t.verifier.spec["tests"])
    assert solved >= 0.6 * len(tasks)


def test_python_code_task_renders_tested_code(srm) -> None:
    r = srm.ask({"examples": [{"xs": [3, 1, 2], "k": 0, "out": [1, 2, 3]}, {"xs": [5, 4], "k": 0, "out": [4, 5]},
                              {"xs": [], "k": 0, "out": []}, {"xs": [2, 2, 1], "k": 0, "out": [1, 2, 2]}],
                 "format": "python"})
    assert r.cls == DecisionClass.KNOWN
    assert r.text.startswith("def solve(") and "sorted(xs)" in r.text
    ns: dict = {}
    exec(r.text, ns)  # noqa: S102 — validated subset code produced and tested by the model
    assert ns["solve"]([9, 7, 8], 0) == [7, 8, 9]


def test_budget_exhaustion_is_unresolved(srm) -> None:
    hard = {"examples": [{"xs": [4, 1, 7, 2], "k": 2, "out": [9, 3]}, {"xs": [5, 5, 3], "k": 1, "out": [6, 6, 4]},
                         {"xs": [], "k": 0, "out": []}], "format": "dsl"}
    r = srm.ask(hard, max_beats=1)
    assert r.cls == DecisionClass.UNRESOLVED and "within the budget" in r.text


def test_online_library_learning_amortizes_search() -> None:
    m = SRM(tiny())
    task = {"examples": [{"xs": [3, -1, 4], "k": 0, "out": [8, 6]}, {"xs": [1, 2, 0], "k": 0, "out": [4, 2]},
                         {"xs": [-5, 7], "k": 0, "out": [14]}, {"xs": [], "k": 0, "out": []}], "format": "dsl"}
    first = m.ask(task, max_beats=60)
    assert first.cls in (DecisionClass.KNOWN, DecisionClass.AMBIGUOUS)
    evals_first = next(n for n in first.circuit if n["op"] == "SYNTH_DSL")["note"]
    second = m.ask(task, max_beats=60)
    evals_second = next(n for n in second.circuit if n["op"] == "SYNTH_DSL")["note"]
    n1 = int(evals_first.split("after ")[1].split(" ")[0])
    n2 = int(evals_second.split("after ")[1].split(" ")[0])
    assert n2 < n1  # the stored procedure is reused as a single component


def test_answer_record_and_episode_written(srm) -> None:
    n_before = srm.memory.n_records()
    r = srm.ask("Expand (x + 1)*(x + 4).")
    assert r.answer_record.decision_class == "KNOWN"
    assert any(t["op"] == "ALGEBRA" for t in r.answer_record.circuit_trace)
    assert srm.memory.n_records() == n_before + 1  # the episode
