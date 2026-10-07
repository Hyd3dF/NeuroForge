"""Perception pipeline (03 §7): chunker, encoder, L0 predictor and recognition-by-recall.

Before bootstrap stage S2 trains the neural perception encoder (component
``perception.encoder``), chunks are encoded by a deterministic hashed bag of word and
character-trigram vectors.  The interface to the rest of the system (``h_chunk``,
``ε0``, recognition outcome) is the same for both encoders.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

import numpy as np

from srm.config.build_config import BuildConfig
from srm.interface import codes as C
from srm.interface.codec import normalize_surface, normalize_text
from srm.interface.codespace import normalize, seeded_dense
from srm.interface.layer import InterfaceLayer
from srm.interface.messages import RecordKind
from srm.memory.system import MemorySystem

_SENT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, max_bytes: int) -> list[str]:
    """Sentence chunks, each capped at ``max_bytes`` (03 §7.1)."""
    out = []
    for sent in _SENT.split(normalize_text(text).strip()):
        sent = sent.strip()
        while len(sent.encode("utf-8")) > max_bytes:
            cut = sent.rfind(" ", 0, max_bytes) if " " in sent[:max_bytes] else max_bytes
            out.append(sent[:cut].strip())
            sent = sent[cut:].strip()
        if sent:
            out.append(sent)
    return out


class HashedEncoder:
    """Deterministic pre-S2 chunk encoder: words + character trigrams → seeded vectors."""

    def __init__(self, d: int) -> None:
        self.d = d
        self._cache: dict[str, np.ndarray] = {}

    def _vec(self, token: str) -> np.ndarray:
        v = self._cache.get(token)
        if v is None:
            v = self._cache[token] = seeded_dense(token, self.d, salt="perception")
        return v

    def encode(self, chunk: str) -> np.ndarray:
        norm = normalize_surface(chunk)
        words = norm.split()
        acc = np.zeros(self.d, dtype=np.float32)
        for w in words:
            acc += self._vec("w:" + w)
        padded = f" {norm} "
        for i in range(len(padded) - 2):
            acc += 0.3 * self._vec("t:" + padded[i : i + 3])
        return normalize(acc) if np.any(acc) else acc


@dataclass
class ChunkPercept:
    text: str
    h: np.ndarray
    code: np.ndarray
    epsilon0: float
    familiarity: float
    familiarity_z: float
    known: bool
    pointer: int | None = None


@dataclass
class Percept:
    chunks: list[ChunkPercept] = field(default_factory=list)

    @property
    def novelty(self) -> float:
        if not self.chunks:
            return 0.0
        return float(np.mean([0.0 if c.known else 1.0 for c in self.chunks]))

    @property
    def mean_error(self) -> float:
        return float(np.mean([c.epsilon0 for c in self.chunks])) if self.chunks else 0.0


class Perception:
    def __init__(self, config: BuildConfig, interface: InterfaceLayer, memory: MemorySystem) -> None:
        self.cfg = config
        self.cs = interface.codespace
        self.memory = memory
        self.encoder = HashedEncoder(config.interface.d)
        self.state = np.zeros(config.interface.d, dtype=np.float32)  # recurrent state across chunks
        self.decay = 0.5
        self.trained_l0 = False

    def l0_predict(self) -> np.ndarray:
        """Pre-S2 L0 predictor: the next chunk is predicted to resemble the running state."""
        return normalize(self.state) if np.any(self.state) else self.state

    def process(self, text: str, remember: bool = False) -> Percept:
        p = self.cfg.perception
        out = Percept()
        for chunk in chunk_text(text, p.chunk_max_bytes):
            h = self.encoder.encode(chunk)
            pred = self.l0_predict()
            eps = 1.0 - float(np.dot(pred, h)) if np.any(pred) else 1.0
            self.state = self.decay * self.state + h
            code = self.cs.project(h)
            hits = self.memory.retrieve(code, kind=RecordKind.EPISODE, k=1)
            fam, z, known, pointer = 0.0, 0.0, False, None
            if hits:
                rid = hits[0][0]
                rec = self.memory.record(rid)
                ov = int(C.overlap(code, rec.content_code))
                fam, z = ov / self.cs.B, C.familiarity_z(ov, self.cs.B, self.cs.L)
                cos = float(np.dot(rec.dense, h))
                # the pre-S2 L0 predictor is a running average, so its error does not gate recognition
                # until the trained predictor is installed (``trained_l0``)
                err_ok = eps <= p.theta_err if self.trained_l0 else True
                known = fam >= p.theta_known and cos >= p.theta_cos and err_ok
                pointer = rid if known else None
            out.chunks.append(ChunkPercept(chunk, h, code, eps, fam, z, known, pointer))
            if remember and not known:
                self._remember(chunk, h)
        return out

    def _remember(self, chunk: str, h: np.ndarray) -> int:
        from srm.memory.payloads import EpisodePayload

        payload = EpisodePayload(episode_id=f"chunk:{hashlib.sha256(chunk.encode()).hexdigest()[:16]}", goal="", kind="percept",
                                 outcome={"text": chunk})
        return self.memory.write(RecordKind.EPISODE, payload.episode_id, h, payload)
