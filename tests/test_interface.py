"""M1: interface layer — code algebra (03 §2.2), projection (03 §2.4), registries (03 §3–4), ABI (03 §1.3)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from srm.interface import InterfaceLayer
from srm.interface import codes as C
from srm.interface.codec import normalize_surface, normalize_text
from srm.interface.values import Value

B, L = 64, 64
code_strategy = st.lists(st.integers(0, L - 1), min_size=B, max_size=B).map(lambda xs: np.array(xs, dtype=np.uint8))


@given(code_strategy, code_strategy)
def test_bind_unbind_are_exact_inverses(a: np.ndarray, b: np.ndarray) -> None:
    c = C.bind(a, b, L)
    assert np.array_equal(C.unbind(c, b, L), a)
    assert np.array_equal(C.bind(a, b, L), C.bind(b, a, L))  # commutative
    assert c.dtype == np.uint8 and int(c.max()) < L


@given(code_strategy, st.integers(0, B))
def test_permute_is_invertible(a: np.ndarray, k: int) -> None:
    assert np.array_equal(C.permute(C.permute(a, k), -k), a)


def test_random_overlap_matches_null_model() -> None:
    rng = np.random.default_rng(0)
    a, b = C.random_codes(rng, 5000, B, L), C.random_codes(rng, 5000, B, L)
    ov = C.overlap(a, b)
    mean, std = C.null_overlap_stats(B, L)
    assert abs(ov.mean() - mean) < 0.05
    assert abs(ov.std() - std) < 0.05
    assert C.familiarity_z(B, B, L) > 50


def test_bundle_membership_and_role_unbinding() -> None:
    rng = np.random.default_rng(1)
    roles = C.random_codes(rng, 4, B, L)
    fillers = C.random_codes(rng, 4, B, L)
    M = C.bundle(C.bind(roles, fillers, L), L)
    for r, f in zip(roles, fillers):
        assert C.match_bundle(C.bind(r, f, L), M) >= B  # every bound pair is a member
        recovered = C.unbind_bundle(M, r)
        # the true filler scores a full B in the role-unbound distribution
        assert C.match_bundle(f, recovered) >= B
    outsider = C.random_codes(rng, 1, B, L)[0]
    assert C.match_bundle(outsider, M) < B / 4
    sparse = C.sparsify(M, tie_seed=3)
    assert sparse.shape == (B,)


def test_code_from_key_is_deterministic_and_salted() -> None:
    a = C.code_from_key("rel:capital", B, L, "relation")
    assert np.array_equal(a, C.code_from_key("rel:capital", B, L, "relation"))
    assert C.overlap(a, C.code_from_key("rel:capital", B, L, "entity")) < 8


def test_projection_preserves_similarity_monotonically(interface: InterfaceLayer) -> None:
    cs = interface.codespace
    rng = np.random.default_rng(0)
    x = rng.standard_normal((800, cs.d)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    agree = []
    for c in (0.0, 0.5, 0.8, 0.95, 0.99):
        n = rng.standard_normal(x.shape).astype(np.float32)
        n -= (n * x).sum(1, keepdims=True) * x
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        y = c * x + np.sqrt(1 - c * c) * n
        agree.append(float((cs.project(x) == cs.project(y)).mean()))
    assert agree == sorted(agree)
    assert agree[0] < 0.05 and agree[-1] > 0.7


def test_embed_and_key_shapes(interface: InterfaceLayer) -> None:
    cs = interface.codespace
    code = cs.project(np.ones(cs.d, dtype=np.float32))
    v = cs.embed(code)
    assert v.shape == (cs.d,) and abs(np.linalg.norm(v) - 1.0) < 1e-5
    assert cs.key(v).shape == (cs.d_k,)


def test_value_thermometer_codes_preserve_order(interface: InterfaceLayer) -> None:
    ve = interface.values
    base = ve.scalar_code("p", 10.0, 0.0, 100.0)
    near = ve.scalar_code("p", 12.0, 0.0, 100.0)
    far = ve.scalar_code("p", 90.0, 0.0, 100.0)
    assert C.overlap(base, near) > C.overlap(base, far)


def test_value_matching_with_tolerance() -> None:
    assert Value("float", 1.0, "m", 0.1).matches(Value("float", 1.05, "m"))
    assert not Value("float", 1.0, "m").matches(Value("float", 1.0, "km"))
    assert Value("entity_ref", "x").matches(Value("entity_ref", "x"))
    assert Value.from_dict(Value("range", (1, 2)).to_dict()) == Value("range", (1, 2))


def test_registries_are_append_only_after_freeze(interface: InterfaceLayer) -> None:
    rel = interface.relations
    rel.add_relation("rel:capital", "capital", cardinality="functional")
    assert rel.is_functional("rel:capital")
    with pytest.raises(ValueError):
        rel.add_relation("rel:capital", "capital", cardinality="multi")  # immutable definition
    abi_before = interface.abi_hash()
    interface.freeze()
    assert interface.abi_hash() == abi_before  # freezing does not change the ABI
    entry = rel.add_relation("rel:new", "new")
    assert entry.provisional
    interface.note_minor_addition()
    assert str(interface.version) == "1.1"
    assert interface.abi_hash() == abi_before  # MINOR additions do not change the MAJOR baseline
    with pytest.raises(RuntimeError):
        interface.codespace.load_tensors(interface.codespace.tensors())


def test_abi_hash_changes_with_major_properties(cfg) -> None:
    a = InterfaceLayer(cfg).abi_hash()
    assert a == InterfaceLayer(cfg).abi_hash()  # deterministic
    other = InterfaceLayer(cfg.with_overrides({"interface": {"L": 32}}))
    assert other.abi_hash() != a


@settings(max_examples=50)
@given(st.text())
def test_codec_normalization_is_idempotent(text: str) -> None:
    once = normalize_text(text)
    assert normalize_text(once) == once
    assert "\r" not in once
    assert normalize_surface(normalize_surface(text)) == normalize_surface(text)
