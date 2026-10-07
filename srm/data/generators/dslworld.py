"""DSLWorld generator (02 §14.2): typed DSL tasks with I/O examples, unit-test verifiers,
reference programs (DSL + Python subset) and held-out compositions."""

from __future__ import annotations

from srm.body import dsl
from srm.data.generators.common import GenContext
from srm.data.io import Dataset
from srm.data.sef import SkillDemoRecord, TaskRecord, Verifier

VERSION = "0.1"
CATEGORY = "synthetic/dslworld"


def generate(dataset_id: str = "dsl", seed: int = 0, n_tasks: int = 2_000, n_examples: int = 5, n_tests: int = 5,
             max_depth: int = 4, heldout_pair_fraction: float = 0.1, heldout_fraction: float = 0.1) -> Dataset:
    ctx = GenContext(dataset_id, seed, "dslworld", VERSION)
    rng = ctx.rng
    src = ctx.source("gen", "generator_truth", category=CATEGORY)

    probe_inputs = [dsl.random_input(rng) for _ in range(12)]
    seen_behaviors: set[tuple] = set()
    programs: list[dsl.Node] = []
    attempts = 0
    while len(programs) < n_tasks and attempts < n_tasks * 200:
        attempts += 1
        prog = dsl.sample_program(rng, "list" if rng.random() < 0.6 else "int", max_depth)
        if "xs" not in prog.ops():
            continue
        beh = dsl.behavior(prog, probe_inputs)
        if beh is None or len(set(beh)) <= 1 or beh in seen_behaviors:
            continue
        seen_behaviors.add(beh)
        programs.append(prog)

    # held-out compositions: operator adjacencies never seen in training (02 §14.2)
    all_pairs = sorted({e for p in programs for e in p.edges()})
    n_hold = max(1, int(len(all_pairs) * heldout_pair_fraction))
    order = rng.permutation(len(all_pairs))
    heldout_pairs = {all_pairs[i] for i in order[:n_hold]}
    target_heldout = int(n_tasks * heldout_fraction)

    n_heldout_tasks = 0
    for prog in programs:
        has_heldout = any(e in heldout_pairs for e in prog.edges())
        if has_heldout:
            if n_heldout_tasks >= target_heldout:
                continue
            split = "heldout_composition"
            n_heldout_tasks += 1
        else:
            r = rng.random()
            split = "train" if r < 0.85 else ("dev" if r < 0.93 else "test")
        examples, tests = [], []
        while len(examples) + len(tests) < n_examples + n_tests:
            xs, k = dsl.random_input(rng)
            try:
                out = dsl.run(prog, xs, k)
            except dsl.DSLError:
                continue
            item = {"xs": xs, "k": k, "out": out}
            (examples if len(examples) < n_examples else tests).append(item)
        ret = dsl.type_of(prog)
        ctx.add(TaskRecord(
            record_id=ctx.rid("task"), source_id=src, data_category=CATEGORY, extraction=ctx.extraction(),
            task_id=ctx.rid("task_id"), family=f"dsl/{ret}",
            goal={"output_type": "list[int]" if ret == "list" else "int", "inputs": {"xs": "list[int]", "k": "int"}},
            inputs={"examples": examples},
            verifier=Verifier(type="unit_tests", spec={"tests": tests}),
            reference_solution={"program": prog.to_json(), "python": dsl.to_python(prog), "depth": prog.depth()},
            split=split, difficulty=float(prog.depth()),
        ))
        if split == "train" and rng.random() < 0.2:
            ctx.add(SkillDemoRecord(
                record_id=ctx.rid("skill_demo"), source_id=src, data_category=CATEGORY, extraction=ctx.extraction(),
                skill_hint=str(prog), input={"xs": examples[0]["xs"], "k": examples[0]["k"]},
                output=examples[0]["out"], domain="dsl",
            ))

    ctx.dataset.meta = {
        "n_programs": len(programs),
        "heldout_pairs": sorted([list(p) for p in heldout_pairs]),
        "n_heldout_tasks": n_heldout_tasks,
    }
    return ctx.dataset
