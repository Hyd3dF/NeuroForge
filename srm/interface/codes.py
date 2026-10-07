"""Sparse block codes and their exact algebra (03 §2.1–2.2; D-002).

A code is a ``uint8`` array of ``B`` block indices in ``[0, L)``: one active unit
per block.  All operations are exact and keep sparsity.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

CODE_DTYPE = np.uint8


def _as_int(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.int32)


def bind(a: np.ndarray, b: np.ndarray, L: int) -> np.ndarray:
    """``(a ⊗ b)[i] = (a[i] + b[i]) mod L`` (broadcasts over leading axes)."""
    return ((_as_int(a) + _as_int(b)) % L).astype(CODE_DTYPE)


def unbind(c: np.ndarray, b: np.ndarray, L: int) -> np.ndarray:
    """``(c ⊘ b)[i] = (c[i] − b[i]) mod L``; exact inverse of :func:`bind`."""
    return ((_as_int(c) - _as_int(b)) % L).astype(CODE_DTYPE)


def permute(a: np.ndarray, k: int) -> np.ndarray:
    """``ρ_k(a)[i] = a[(i + k) mod B]`` — encodes order and position."""
    return np.roll(np.asarray(a), -k, axis=-1)


def overlap(a: np.ndarray, b: np.ndarray) -> np.ndarray | int:
    """Number of agreeing blocks; similarity in ``[0, B]``."""
    return (np.asarray(a) == np.asarray(b)).sum(axis=-1)


def overlap_many(query: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Overlap of one query code with each row of ``codes`` (shape ``(n, B)``)."""
    return (np.asarray(codes) == np.asarray(query)[None, :]).sum(axis=1)


def bundle(codes: np.ndarray, L: int, weights: np.ndarray | None = None) -> np.ndarray:
    """Soft bundle: count matrix ``M[i, a_j[i]] += w_j`` of shape ``(B, L)``."""
    codes = np.atleast_2d(np.asarray(codes, dtype=np.int64))
    n, B = codes.shape
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    M = np.zeros((B, L), dtype=np.float64)
    rows = np.broadcast_to(np.arange(B), (n, B))
    np.add.at(M, (rows.ravel(), codes.ravel()), np.repeat(w, B))
    return M


def sparsify(M: np.ndarray, tie_seed: int = 0) -> np.ndarray:
    """Per-block argmax of a bundle; ties are broken by a seeded jitter < 1e-6."""
    M = np.asarray(M, dtype=np.float64)
    rng = np.random.default_rng(tie_seed)
    jitter = rng.random(M.shape) * 1e-6
    return np.argmax(M + jitter, axis=-1).astype(CODE_DTYPE)


def match_bundle(query: np.ndarray, M: np.ndarray) -> float:
    """Membership score ``Σ_i M[i, q[i]]`` of a code in a bundle."""
    q = np.asarray(query, dtype=np.int64)
    return float(np.asarray(M)[np.arange(q.shape[0]), q].sum())


def unbind_bundle(M: np.ndarray, role: np.ndarray) -> np.ndarray:
    """Row-wise cyclic shift: ``out[i, f] = M[i, (f + role[i]) mod L]``.

    The result is a per-block distribution over fillers bound to ``role``; it is
    cleaned up against a codebook by the memory system (04 §5.4).
    """
    M = np.asarray(M)
    B, L = M.shape
    idx = (np.arange(L)[None, :] + np.asarray(role, dtype=np.int64)[:, None]) % L
    return np.take_along_axis(M, idx, axis=1)


def null_overlap_stats(B: int, L: int) -> tuple[float, float]:
    """Mean and standard deviation of the overlap of two random codes (Binomial(B, 1/L))."""
    p = 1.0 / L
    return B * p, math.sqrt(B * p * (1.0 - p))


def familiarity_z(ov: float, B: int, L: int) -> float:
    """z-score of an overlap against the random-code null model (03 §7.4)."""
    mean, std = null_overlap_stats(B, L)
    return (ov - mean) / std


def code_from_key(key: str, B: int, L: int, salt: str = "") -> np.ndarray:
    """Deterministic pseudo-random code for a symbol (identity codes, 03 §2.3)."""
    stream = hashlib.shake_256((salt + "\x1f" + key).encode("utf-8")).digest(4 * B)
    words = np.frombuffer(stream, dtype=">u4").astype(np.uint64)
    return (words % np.uint64(L)).astype(CODE_DTYPE)


def random_codes(rng: np.random.Generator, n: int, B: int, L: int) -> np.ndarray:
    return rng.integers(0, L, size=(n, B), dtype=np.int64).astype(CODE_DTYPE)


def validate_code(code: np.ndarray, B: int, L: int) -> None:
    c = np.asarray(code)
    if c.shape[-1] != B:
        raise ValueError(f"code has {c.shape[-1]} blocks, expected {B}")
    if c.size and int(c.max()) >= L:
        raise ValueError(f"code contains a unit index >= L ({L})")
