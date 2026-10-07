"""Code space: content-code projection, code embedding and dense keys (03 §2.3–2.5).

The projection ``code[i] = argmax_l (W_i x + β_i)_l`` maps dense vectors to sparse
block codes so that similar vectors share blocks.  Before bootstrap stage S1 the
tables are seeded random (a winner-take-all random projection, which already
preserves cosine similarity in expectation); S1 trains them and then freezes them.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from srm.interface import codes as C
from srm.util.canonical import array_digest


def normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


class CodeSpace:
    """Holds the projection ``W, β``, embedding tables ``E`` and the dense-key head."""

    def __init__(self, B: int, L: int, d: int, d_k: int, r: int, n_b: int, seed: int) -> None:
        if r * n_b > B:
            raise ValueError("band layout does not fit in B blocks")
        self.B, self.L, self.d, self.d_k, self.r, self.n_b = B, L, d, d_k, r, n_b
        rng = np.random.default_rng(seed)
        self.W = (rng.standard_normal((B * L, d)) / math.sqrt(d)).astype(np.float32)
        self.beta = np.zeros((B * L,), dtype=np.float32)
        self.E = (rng.standard_normal((B, L, d)) / math.sqrt(d)).astype(np.float32)
        self.key_head = (rng.standard_normal((d, d_k)) / math.sqrt(d)).astype(np.float32)
        self.band_layout: tuple[tuple[int, ...], ...] = tuple(
            tuple(range(j * r, j * r + r)) for j in range(n_b)
        )
        self.frozen = False

    # --- projection: dense → code -------------------------------------------------------
    def logits(self, x: np.ndarray) -> np.ndarray:
        x2 = np.atleast_2d(np.asarray(x, dtype=np.float32))
        return (x2 @ self.W.T + self.beta).reshape(x2.shape[0], self.B, self.L)

    def project(self, x: np.ndarray, batch: int = 4096) -> np.ndarray:
        """Dense ``(d,)`` or ``(n, d)`` → code ``(B,)`` or ``(n, B)``."""
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        x2 = np.atleast_2d(x)
        out = np.empty((x2.shape[0], self.B), dtype=C.CODE_DTYPE)
        for s in range(0, x2.shape[0], batch):
            out[s : s + batch] = np.argmax(self.logits(x2[s : s + batch]), axis=-1)
        return out[0] if single else out

    # --- embedding: code → dense --------------------------------------------------------
    def embed(self, code: np.ndarray) -> np.ndarray:
        """``emb(c) = normalize(Σ_i E_i[c[i]])``; accepts ``(B,)`` or ``(n, B)``."""
        c = np.asarray(code, dtype=np.int64)
        single = c.ndim == 1
        c2 = np.atleast_2d(c)
        vec = self.E[np.arange(self.B)[None, :], c2].sum(axis=1)
        vec = normalize(vec)
        return vec[0] if single else vec

    def embed_bundle(self, M: np.ndarray) -> np.ndarray:
        """``normalize(Σ_i Σ_l M[i,l] · E_i[l])``."""
        return normalize(np.einsum("bl,bld->d", np.asarray(M, dtype=np.float32), self.E))

    # --- dense key ----------------------------------------------------------------------
    def key(self, dense: np.ndarray) -> np.ndarray:
        return normalize(np.asarray(dense, dtype=np.float32) @ self.key_head)

    # --- identity -----------------------------------------------------------------------
    def digest(self) -> str:
        return array_digest(
            np.array([self.B, self.L, self.d, self.d_k, self.r, self.n_b], dtype=np.int64),
            self.W, self.beta, self.E, self.key_head,
        )

    def freeze(self) -> None:
        self.frozen = True
        for arr in (self.W, self.beta, self.E, self.key_head):
            arr.setflags(write=False)

    def tensors(self) -> dict[str, np.ndarray]:
        return {"W": self.W, "beta": self.beta, "E": self.E, "key_head": self.key_head}

    def load_tensors(self, tensors: dict[str, np.ndarray]) -> None:
        if self.frozen:
            raise RuntimeError("code space is frozen (03 §1.4)")
        for name in ("W", "beta", "E", "key_head"):
            arr = np.asarray(tensors[name], dtype=np.float32)
            if arr.shape != getattr(self, name).shape:
                raise ValueError(f"tensor {name} has shape {arr.shape}, expected {getattr(self, name).shape}")
            setattr(self, name, arr.copy())


def seeded_dense(key: str, d: int, salt: str = "") -> np.ndarray:
    """Deterministic unit vector for a symbol (pre-S1 initialization of symbol embeddings)."""
    seed = int.from_bytes(hashlib.sha256((salt + "\x1f" + key).encode("utf-8")).digest()[:8], "big")
    return normalize(np.random.default_rng(seed).standard_normal(d))
