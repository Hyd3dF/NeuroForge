"""Association Field: typed weighted links, spreading activation and Hebbian updates (04 §6)."""

from __future__ import annotations

from collections import defaultdict

EDGE_TYPES = (
    "is_a", "part_of", "has_property", "causes", "prevents", "enables", "associated_with",
    "analogous_to", "used_with", "contradicts", "derived_from", "trained_from", "co_activated",
)
CAUSAL_TYPES = frozenset({"causes", "prevents", "enables"})


class AssociationField:
    def __init__(self) -> None:
        self.out: dict[int, dict[tuple[int, str], float]] = defaultdict(dict)
        self.inc: dict[int, set[tuple[int, str]]] = defaultdict(set)

    def link(self, src: int, dst: int, etype: str, weight: float = 1.0) -> None:
        if etype not in EDGE_TYPES:
            raise ValueError(f"unknown edge type {etype}")
        self.out[src][(dst, etype)] = weight
        self.inc[dst].add((src, etype))

    def weight(self, src: int, dst: int, etype: str) -> float:
        return self.out.get(src, {}).get((dst, etype), 0.0)

    def neighbors(self, src: int, etype: str | None = None) -> list[tuple[int, str, float]]:
        return [(d, t, w) for (d, t), w in self.out.get(src, {}).items() if etype is None or t == etype]

    def unlink(self, src: int, dst: int, etype: str) -> None:
        self.out.get(src, {}).pop((dst, etype), None)
        self.inc.get(dst, set()).discard((src, etype))

    def n_edges(self) -> int:
        return sum(len(v) for v in self.out.values())

    def spread(self, seeds: dict[int, float], gains: dict[str, float], gamma: float, hops: int,
               k: int) -> dict[int, float]:
        """``a_j ← clip(γ·a_j + Σ w_ij · g_type · a_i, 0, 1)``, top-k kept each hop."""
        act = {i: min(1.0, max(0.0, a)) for i, a in seeds.items()}
        for _ in range(hops):
            nxt: dict[int, float] = {i: gamma * a for i, a in act.items()}
            for i, a in act.items():
                for (j, etype), w in self.out.get(i, {}).items():
                    g = gains.get(etype, 0.0)
                    if g > 0.0:
                        nxt[j] = nxt.get(j, 0.0) + w * g * a
            nxt = {i: min(1.0, a) for i, a in nxt.items() if a > 0.0}
            if len(nxt) > k:
                nxt = dict(sorted(nxt.items(), key=lambda t: -t[1])[:k])
            act = nxt
        return act

    def hebbian(self, active: dict[int, float], eta: float, modulator: float, eta_decay: float,
                w_max: float) -> int:
        """``Δw_ij = η·m·a_i·a_j − η_decay·w_ij`` on ``co_activated`` edges among active records."""
        ids = sorted(active)
        updated = 0
        for x, i in enumerate(ids):
            for j in ids[x + 1 :]:
                for s, t in ((i, j), (j, i)):
                    w = self.weight(s, t, "co_activated")
                    w_new = min(w_max, max(0.0, w + eta * modulator * active[s] * active[t] - eta_decay * w))
                    if w_new > 0.0:
                        self.link(s, t, "co_activated", w_new)
                        updated += 1
        return updated

    def to_rows(self) -> list[tuple[int, int, str, float]]:
        return [(s, d, t, w) for s, edges in self.out.items() for (d, t), w in edges.items()]
