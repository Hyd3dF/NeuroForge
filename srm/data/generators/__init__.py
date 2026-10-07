"""Deterministic synthetic generators for SRM-F0 (02 §14)."""

from __future__ import annotations

from srm.config.build_config import BuildConfig
from srm.data.generators import dslworld, factstream, mathworld, objectworld
from srm.data.io import Dataset


def generate_all(config: BuildConfig, seed: int | None = None) -> dict[str, Dataset]:
    """Generate the four F0 datasets sized by ``learning.bootstrap.data`` (02 §14.6)."""
    s = config.meta.seed if seed is None else seed
    data = config.learning.bootstrap.data
    n_countries = max(4, data["factstream_entities"] // 6)
    return {
        "objectworld": objectworld.generate(
            "ow", s, n_props=min(200, config.interface.P_0), n_concepts=data["objectworld_concepts"],
            n_heldout=data["objectworld_heldout_concepts"], n_instances=data["objectworld_instances"],
            n_texts=data["objectworld_texts"],
        ),
        "factstream": factstream.generate("fs", s, n_countries=n_countries),
        "dslworld": dslworld.generate("dsl", s, n_tasks=data["dsl_tasks"]),
        "mathworld": mathworld.generate("math", s, n_problems=data["math_problems"]),
    }


__all__ = ["dslworld", "factstream", "mathworld", "objectworld", "generate_all"]
