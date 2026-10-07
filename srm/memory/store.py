"""Columnar record store (04 §2.1, §11).

One store per region (and one for the Hippocampal Index).  Numeric columns are
numpy arrays that grow geometrically; structured columns (payload, evidence
ledger, provenance) are Python lists.  Persistence: safetensors + Parquet (D-005).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import load_file, save_file

from srm.core.evidence import EvidenceLedger
from srm.interface.messages import EpistemicState, Lifecycle, RecordKind
from srm.memory.payloads import payload_from_dict

KINDS = list(RecordKind)
LIFECYCLES = list(Lifecycle)
STATES = [None, *list(EpistemicState)]
TIERS = ("T0", "T1", "T2")
LOCAL_BITS = 48
LOCAL_MASK = (1 << LOCAL_BITS) - 1


def make_record_id(store_id: int, local: int) -> int:
    return (store_id << LOCAL_BITS) | local


def split_record_id(record_id: int) -> tuple[int, int]:
    return record_id >> LOCAL_BITS, record_id & LOCAL_MASK


@dataclass
class RecordView:
    record_id: int
    kind: RecordKind
    identity_code: np.ndarray
    content_code: np.ndarray
    dense: np.ndarray
    payload: Any
    ledger: EvidenceLedger
    provenance: list[dict[str, Any]]
    lifecycle: Lifecycle
    state: EpistemicState | None
    usage_count: int
    alive: bool
    version: int
    supersedes: int
    meta: dict[str, Any]


class RecordStore:
    def __init__(self, store_id: int, name: str, B: int, d: int, d_k: int, S: int, page_records: int,
                 capacity: int = 1024) -> None:
        self.store_id, self.name = store_id, name
        self.B, self.d, self.d_k, self.S, self.page_records = B, d, d_k, S, page_records
        self.n = 0
        self._cap = 0
        self._arrays: dict[str, np.ndarray] = {}
        self.payloads: list[Any] = []
        self.ledgers: list[EvidenceLedger] = []
        self.provenance: list[list[dict[str, Any]]] = []
        self.meta: list[dict[str, Any]] = []
        self._grow(capacity)

    # --- storage ------------------------------------------------------------------------------
    def _specs(self) -> dict[str, tuple[tuple[int, ...], Any, Any]]:
        return {
            "kind": ((), np.uint8, 0),
            "identity": ((self.B,), np.uint8, 0),
            "content": ((self.B,), np.uint8, 0),
            "signature": ((self.B,), np.uint8, 0),
            "has_signature": ((), np.bool_, False),
            "context": ((self.S, self.B), np.uint8, 0),
            "n_context": ((), np.uint8, 0),
            "dense": ((self.d,), np.float16, 0),
            "key": ((self.d_k,), np.float16, 0),
            "state": ((), np.uint8, 0),
            "lifecycle": ((), np.uint8, 0),
            "usage": ((), np.int32, 0),
            "last_used": ((), np.float64, 0),
            "value_sum": ((), np.float32, 0),
            "cost_sum": ((), np.float32, 0),
            "plasticity": ((), np.float16, 1.0),
            "tier": ((), np.uint8, 1),
            "version": ((), np.uint32, 0),
            "supersedes": ((), np.int64, -1),
            "alive": ((), np.bool_, True),
        }

    def _grow(self, capacity: int) -> None:
        new_cap = max(capacity, 2 * self._cap, 16)
        for name, (shape, dtype, fill) in self._specs().items():
            arr = np.full((new_cap, *shape), fill, dtype=dtype)
            if name in self._arrays and self.n:
                arr[: self.n] = self._arrays[name][: self.n]
            self._arrays[name] = arr
        self._cap = new_cap

    def col(self, name: str) -> np.ndarray:
        """Live view of a column for the first ``n`` records."""
        return self._arrays[name][: self.n]

    # --- mutation -----------------------------------------------------------------------------
    def append(
        self, kind: RecordKind, identity_code: np.ndarray, content_code: np.ndarray, dense: np.ndarray,
        key: np.ndarray, payload: Any, ledger: EvidenceLedger, provenance: list[dict[str, Any]],
        lifecycle: Lifecycle = Lifecycle.NEW, signature: np.ndarray | None = None,
        context_codes: list[np.ndarray] | None = None, meta: dict[str, Any] | None = None,
        plasticity: float = 1.0,
    ) -> int:
        if self.n >= self._cap:
            self._grow(self._cap * 2)
        i = self.n
        a = self._arrays
        a["kind"][i] = KINDS.index(kind)
        a["identity"][i] = identity_code
        a["content"][i] = content_code
        if signature is not None:
            a["signature"][i] = signature
            a["has_signature"][i] = True
        for j, cc in enumerate((context_codes or [])[: self.S]):
            a["context"][i, j] = cc
        a["n_context"][i] = min(len(context_codes or []), self.S)
        a["dense"][i] = dense
        a["key"][i] = key
        a["lifecycle"][i] = LIFECYCLES.index(lifecycle)
        a["plasticity"][i] = plasticity
        self.payloads.append(payload)
        self.ledgers.append(ledger)
        self.provenance.append(list(provenance))
        self.meta.append(dict(meta or {}))
        self.n += 1
        return make_record_id(self.store_id, i)

    def set_lifecycle(self, local: int, lifecycle: Lifecycle, plasticity: float | None = None) -> None:
        self._arrays["lifecycle"][local] = LIFECYCLES.index(lifecycle)
        if plasticity is not None:
            self._arrays["plasticity"][local] = plasticity

    def set_state(self, local: int, state: EpistemicState | None) -> None:
        self._arrays["state"][local] = STATES.index(state)

    def touch(self, local: int, now: float, value: float = 0.0, cost: float = 0.0) -> None:
        a = self._arrays
        a["usage"][local] += 1
        a["last_used"][local] = now
        a["value_sum"][local] += value
        a["cost_sum"][local] += cost

    def kill(self, local: int) -> None:
        self._arrays["alive"][local] = False

    # --- access -------------------------------------------------------------------------------
    def kind_of(self, local: int) -> RecordKind:
        return KINDS[int(self._arrays["kind"][local])]

    def lifecycle_of(self, local: int) -> Lifecycle:
        return LIFECYCLES[int(self._arrays["lifecycle"][local])]

    def page_of(self, local: int) -> int:
        return local // self.page_records

    def view(self, local: int) -> RecordView:
        a = self._arrays
        return RecordView(
            record_id=make_record_id(self.store_id, local),
            kind=self.kind_of(local),
            identity_code=a["identity"][local],
            content_code=a["content"][local],
            dense=a["dense"][local].astype(np.float32),
            payload=self.payloads[local],
            ledger=self.ledgers[local],
            provenance=self.provenance[local],
            lifecycle=self.lifecycle_of(local),
            state=STATES[int(a["state"][local])],
            usage_count=int(a["usage"][local]),
            alive=bool(a["alive"][local]),
            version=int(a["version"][local]),
            supersedes=int(a["supersedes"][local]),
            meta=self.meta[local],
        )

    # --- persistence (04 §11, 09 §5) -----------------------------------------------------------
    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        save_file({k: np.ascontiguousarray(v[: self.n]) for k, v in self._arrays.items()},
                  str(directory / "columns.safetensors"))
        table = pa.table({
            "payload": [json.dumps(p.to_dict(), sort_keys=True) for p in self.payloads],
            "ledger": [json.dumps(l.to_dict(), sort_keys=True) for l in self.ledgers],
            "provenance": [json.dumps(p, sort_keys=True) for p in self.provenance],
            "meta": [json.dumps(m, sort_keys=True) for m in self.meta],
        })
        pq.write_table(table, directory / "records.parquet")
        (directory / "store.json").write_text(json.dumps({
            "store_id": self.store_id, "name": self.name, "n": self.n, "B": self.B, "d": self.d,
            "d_k": self.d_k, "S": self.S, "page_records": self.page_records,
        }), encoding="utf-8")

    @staticmethod
    def load(directory: str | Path) -> "RecordStore":
        directory = Path(directory)
        info = json.loads((directory / "store.json").read_text(encoding="utf-8"))
        st = RecordStore(info["store_id"], info["name"], info["B"], info["d"], info["d_k"], info["S"],
                         info["page_records"], capacity=max(16, info["n"]))
        cols = load_file(str(directory / "columns.safetensors"))
        n = info["n"]
        for k, v in cols.items():
            st._arrays[k][:n] = v
        st.n = n
        table = pq.read_table(directory / "records.parquet").to_pydict()
        kinds = st.col("kind")
        st.payloads = [payload_from_dict(KINDS[int(kinds[i])], json.loads(p)) for i, p in enumerate(table["payload"])]
        st.ledgers = [EvidenceLedger.from_dict(json.loads(x)) for x in table["ledger"]]
        st.provenance = [json.loads(x) for x in table["provenance"]]
        st.meta = [json.loads(x) for x in table["meta"]]
        return st
