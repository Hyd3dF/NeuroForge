"""Derived indices (04 §4): banded LSH (content and signature), structured triples, aliases, dense."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from srm.interface.codec import normalize_surface


class BandedIndex:
    """``n_b`` bands of ``r`` blocks; key = the band's block values (04 §4.1, D-003)."""

    def __init__(self, band_layout: tuple[tuple[int, ...], ...], L: int) -> None:
        self.layout = band_layout
        self.L = L
        self.tables: list[dict[int, list[int]]] = [defaultdict(list) for _ in band_layout]
        self._cols = [np.asarray(b, dtype=np.int64) for b in band_layout]
        self._mult = [np.asarray([L**i for i in range(len(b))], dtype=np.int64) for b in band_layout]
        self.size = 0

    def keys(self, code: np.ndarray) -> list[int]:
        c = np.asarray(code, dtype=np.int64)
        return [int((c[cols] * mult).sum()) for cols, mult in zip(self._cols, self._mult)]

    def add(self, local: int, code: np.ndarray) -> None:
        for table, key in zip(self.tables, self.keys(code)):
            table[key].append(local)
        self.size += 1

    def add_many(self, locals_: np.ndarray, codes: np.ndarray) -> None:
        codes = np.asarray(codes, dtype=np.int64)
        for j, (cols, mult) in enumerate(zip(self._cols, self._mult)):
            keys = (codes[:, cols] * mult).sum(axis=1)
            table = self.tables[j]
            for local, key in zip(locals_.tolist(), keys.tolist()):
                table[key].append(local)
        self.size += len(locals_)

    def candidates(self, code: np.ndarray, cand_max: int) -> tuple[np.ndarray, np.ndarray]:
        """Union of postings across bands → (locals, band votes), capped by votes."""
        hits: list[int] = []
        for table, key in zip(self.tables, self.keys(code)):
            posting = table.get(key)
            if posting:
                hits.extend(posting)
        if not hits:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        locals_, votes = np.unique(np.asarray(hits, dtype=np.int64), return_counts=True)
        if len(locals_) > cand_max:
            top = np.argsort(-votes, kind="stable")[:cand_max]
            locals_, votes = locals_[top], votes[top]
        return locals_, votes


class TripleIndex:
    """Exact structured lookup for engrams and taxonomy (04 §4.3)."""

    def __init__(self) -> None:
        self.by_sr: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_ro: dict[tuple[str, str], list[int]] = defaultdict(list)
        self.by_s: dict[str, list[int]] = defaultdict(list)
        self.instances: dict[str, list[int]] = defaultdict(list)
        self.children: dict[str, list[int]] = defaultdict(list)

    def add_fact(self, record_id: int, subject: str, relation: str, object_key: str) -> None:
        self.by_sr[(subject, relation)].append(record_id)
        self.by_ro[(relation, object_key)].append(record_id)
        self.by_s[subject].append(record_id)

    def facts(self, subject: str | None = None, relation: str | None = None, object_key: str | None = None) -> list[int]:
        if subject is not None and relation is not None:
            ids = self.by_sr.get((subject, relation), [])
            if object_key is not None:
                allowed = set(self.by_ro.get((relation, object_key), []))
                ids = [i for i in ids if i in allowed]
            return list(ids)
        if relation is not None and object_key is not None:
            return list(self.by_ro.get((relation, object_key), []))
        if subject is not None:
            return list(self.by_s.get(subject, []))
        raise ValueError("triple lookup needs at least a subject or a (relation, object) pair")


class AliasIndex:
    """Normalized surface form → symbol ids; trigram Jaccard for fuzzy matches (04 §4.4)."""

    def __init__(self) -> None:
        self.exact: dict[str, set[str]] = defaultdict(set)
        self.trigrams: dict[str, set[str]] = defaultdict(set)

    @staticmethod
    def _grams(s: str) -> set[str]:
        s = f"  {s} "
        return {s[i : i + 3] for i in range(len(s) - 2)}

    def add(self, surface: str, symbol_id: str) -> None:
        norm = normalize_surface(surface)
        if not norm:
            return
        if norm not in self.exact:
            for g in self._grams(norm):
                self.trigrams[g].add(norm)
        self.exact[norm].add(symbol_id)

    def lookup(self, surface: str) -> set[str]:
        return set(self.exact.get(normalize_surface(surface), set()))

    def fuzzy(self, surface: str, threshold: float = 0.6, k: int = 5) -> list[tuple[str, float]]:
        norm = normalize_surface(surface)
        grams = self._grams(norm)
        cands: set[str] = set()
        for g in grams:
            cands |= self.trigrams.get(g, set())
        scored = []
        for c in cands:
            cg = self._grams(c)
            j = len(grams & cg) / max(1, len(grams | cg))
            if j >= threshold:
                for sid in self.exact[c]:
                    scored.append((sid, j))
        scored.sort(key=lambda t: -t[1])
        return scored[:k]


class DenseIndex:
    """Brute-force cosine over dense keys (the ``dense_ann`` fallback backend, 04 §4.5)."""

    def __init__(self, d_k: int) -> None:
        self.d_k = d_k

    @staticmethod
    def search(query_key: np.ndarray, keys: np.ndarray, alive: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if len(keys) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        sims = keys.astype(np.float32) @ query_key.astype(np.float32)
        sims = np.where(alive, sims, -np.inf)
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top], kind="stable")]
        return top.astype(np.int64), sims[top]


def unique_preserving(ids: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out
