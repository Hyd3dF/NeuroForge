"""M0: Build Configuration (09 §3) and canonical JSON / hashing (09 §2, §4.1)."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, strategies as st

from srm.config import BuildConfig, f0, tiny
from srm.config.build_config import MACHINERY_FLOOR_TYPES
from srm.util.canonical import b32_hash, canonical_hash, canonical_json, derive_seed


# --- canonical JSON (RFC 8785) ------------------------------------------------------------------
@pytest.mark.parametrize(
    "value, text",
    [
        (333333333.33333329, "333333333.3333333"),
        (1e30, "1e+30"),
        (4.50, "4.5"),
        (2e-3, "0.002"),
        (0.000001, "0.000001"),
        (1e-7, "1e-7"),
        (1e21, "1e+21"),
        (1e20, "100000000000000000000"),
        (-0.0, "0"),
        (-5e-324, "-5e-324"),
        (1.7976931348623157e308, "1.7976931348623157e+308"),
        (123.456, "123.456"),
    ],
)
def test_jcs_numbers(value: float, text: str) -> None:
    assert canonical_json(value) == text


def test_jcs_key_order_and_escaping() -> None:
    obj = {"€": "Euro", "\r": "CR", "1": "One", "\u0080": "ctrl", "a": [True, None, 1, "\u0007"]}
    out = canonical_json(obj)
    assert out.startswith('{"\\r":"CR","1":"One","a":[true,null,1,"\\u0007"]')
    assert json.loads(out) == obj


def test_jcs_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        canonical_json(float("nan"))


@given(st.recursive(
    st.none() | st.booleans() | st.integers(-10**12, 10**12) | st.floats(allow_nan=False, allow_infinity=False) | st.text(),
    lambda children: st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=5), children, max_size=4),
    max_leaves=12,
))
def test_jcs_roundtrip_and_determinism(obj: object) -> None:
    text = canonical_json(obj)
    assert canonical_json(json.loads(text)) == text


def test_hash_helpers_are_stable() -> None:
    assert canonical_hash({"b": 1, "a": 2}) == canonical_hash({"a": 2, "b": 1})
    h = b32_hash({"x": 1})
    assert len(h) == 26 and h == h.lower()
    assert derive_seed("a", 1) == derive_seed("a", 1) != derive_seed("a", 2)


# --- Build Configuration ------------------------------------------------------------------------
def test_f0_defaults_validate_and_hash_is_stable(tmp_path) -> None:
    cfg = f0()
    assert cfg.interface.B == 64 and cfg.workspace.K == 64
    path = tmp_path / "build.json"
    cfg.save(path)
    again = BuildConfig.load(path)
    assert again.config_hash() == cfg.config_hash()
    assert tiny().config_hash() != cfg.config_hash()


@pytest.mark.parametrize(
    "override, message",
    [
        ({"interface": {"r": 4, "n_b": 20}}, "r*n_b"),
        ({"interface": {"L": 300}}, "L must be"),
        ({"workspace": {"K": 8}}, "K (8)"),
        ({"control": {"floor_subconscious": 0.6, "floor_error_monitor": 0.5}}, "floors"),
        ({"epistemics": {"theta_commit": 1.5}}, "theta_commit"),
        ({"epistemics": {"theta_predict": 0.9}}, "theta_predict <= theta_abstain"),
        ({"capacity": {"active_capacity_ceiling": 10}}, "active_capacity_ceiling"),
        ({"capacity": {"machinery_floor": ["core.composer"]}}, "machinery_floor"),
    ],
)
def test_validation_rules_reject_invalid_configs(override: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        f0().with_overrides(override)


def test_config_is_immutable() -> None:
    cfg = f0()
    with pytest.raises(Exception):
        cfg.interface.B = 32  # type: ignore[misc]


def test_machinery_floor_is_part_of_every_profile() -> None:
    for cfg in (f0(), tiny()):
        assert set(MACHINERY_FLOOR_TYPES) <= set(cfg.capacity.machinery_floor)
