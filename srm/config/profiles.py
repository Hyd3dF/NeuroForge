"""Named Build Configuration profiles.

``f0`` is the SRM-F0 prototype profile (12 §1).  ``tiny`` keeps every mechanism
(the machinery floor is unchanged) but shrinks capacities so tests run quickly.
"""

from __future__ import annotations

from srm.config.build_config import BuildConfig


def f0() -> BuildConfig:
    return BuildConfig()


def tiny() -> BuildConfig:
    return BuildConfig().with_overrides(
        {
            "meta": {"model_name": "srm-tiny"},
            "interface": {"d": 64, "d_k": 32, "P_0": 64},
            "memory": {
                "N_lib": 100_000,
                "N_hip": 20_000,
                "page_records": 512,
                "sketch_expected_keys": 200_000,
                "cand_max": 1024,
            },
            "learning": {
                "bootstrap": {
                    "data": {
                        "objectworld_concepts": 60,
                        "objectworld_heldout_concepts": 8,
                        "objectworld_instances": 2_000,
                        "objectworld_texts": 500,
                        "dsl_tasks": 300,
                        "factstream_entities": 300,
                        "factstream_facts": 3_000,
                        "factstream_questions": 400,
                        "math_problems": 200,
                        "expression_pairs": 200,
                    }
                }
            },
        }
    )
