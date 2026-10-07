"""Encode SEF symbols and facts into dense vectors with the frozen interface layer (02 §9).

Before S1 trains symbol embeddings, entity/relation/value vectors are deterministic
seeded vectors; the composition below is fixed so encodings stay stable.
"""

from __future__ import annotations

import numpy as np

from srm.interface.codespace import normalize, seeded_dense
from srm.interface.layer import InterfaceLayer
from srm.interface.values import Value


class SymbolEncoder:
    def __init__(self, interface: InterfaceLayer) -> None:
        self.il = interface
        self.d = interface.codespace.d

    def entity(self, entity_id: str) -> np.ndarray:
        return seeded_dense(entity_id, self.d, salt="entity")

    def relation(self, relation_id: str) -> np.ndarray:
        entry = self.il.relations.get(relation_id)
        return entry.dense if entry is not None else seeded_dense(relation_id, self.d, salt="relation")

    def value(self, relation_id: str, v: Value) -> np.ndarray:
        if v.type == "entity_ref":
            return self.entity(str(v.value))
        if v.is_numeric():
            x = float(v.value)
            lo, hi = (x / 10.0, x * 10.0) if x > 0 else (x - 10.0, x + 10.0)
            return self.il.values.scalar_dense(relation_id, x, min(lo, hi), max(lo, hi))
        return self.il.values.symbol_dense(relation_id, v.value)

    def fact(self, subject: str, relation: str, obj: Value) -> np.ndarray:
        """``normalize(s + ρ(r) + ½·o)`` with ρ a fixed cyclic shift (keeps roles distinguishable)."""
        s, r, o = self.entity(subject), self.relation(relation), self.value(relation, obj)
        return normalize(s + np.roll(r, 1) + 0.5 * o)
