"""Knowledge Sketch: counting Bloom filter with 4-bit saturating counters (04 §7).

No false negatives: a counter that saturates is never decremented, so a stored key
can never read as absent.  A zero counter therefore proves the key was never
stored → UNKNOWN-ABSENT (06 §5).
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

SATURATED = 15


def sketch_size(n_keys: int, fpr: float) -> tuple[int, int]:
    n = max(1, n_keys)
    m = math.ceil(-n * math.log(fpr) / (math.log(2) ** 2))
    k = max(1, round((m / n) * math.log(2)))
    return m, k


class KnowledgeSketch:
    def __init__(self, expected_keys: int, fpr: float) -> None:
        self.m, self.k = sketch_size(expected_keys, fpr)
        self.counters = np.zeros(self.m, dtype=np.uint8)
        self.n_added = 0

    def _positions(self, key: str) -> np.ndarray:
        digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little") | 1
        return np.array([(h1 + i * h2) % self.m for i in range(self.k)], dtype=np.int64)

    def add(self, key: str) -> None:
        pos = self._positions(key)
        c = self.counters[pos]
        self.counters[pos] = np.where(c < SATURATED, c + 1, c)
        self.n_added += 1

    def remove(self, key: str) -> None:
        pos = self._positions(key)
        c = self.counters[pos]
        if np.any(c == 0):
            raise KeyError(f"key was never added to the sketch: {key}")
        self.counters[pos] = np.where((c > 0) & (c < SATURATED), c - 1, c)
        self.n_added -= 1

    def maybe_contains(self, key: str) -> bool:
        return bool(np.all(self.counters[self._positions(key)] > 0))

    def definitely_absent(self, key: str) -> bool:
        return not self.maybe_contains(key)

    def fill_ratio(self) -> float:
        return float(np.count_nonzero(self.counters)) / self.m


# --- canonical sketch keys (04 §7) -----------------------------------------------------------------
def key_entity(entity_id: str) -> str:
    return f"e:{entity_id}"


def key_entity_relation(entity_id: str, relation: str) -> str:
    return f"er:{entity_id}|{relation}"


def key_concept(concept_id: str) -> str:
    return f"c:{concept_id}"


def key_concept_property(concept_id: str, property_id: str) -> str:
    return f"cp:{concept_id}|{property_id}"


def key_alias(normalized_alias: str) -> str:
    return f"a:{normalized_alias}"
