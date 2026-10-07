"""Heart: the resource circulation controller (07 §1; D-015 — deterministic allocation).

Each beat: processes post bids ``(resources, v̂, conf)``; values are scaled by per-process
credit; the Heart funds bids greedily by value density under per-beat caps, reserving floor
shares for the Subconscious and the Error Monitor; after execution, credit is updated from
realized value (bid honesty).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from srm.config.build_config import BuildConfig
from srm.interface.messages import Bid


@dataclass
class Allocation:
    funded: list[Bid]
    rejected: list[Bid]
    flops: float
    bytes: float


@dataclass
class Heart:
    cfg: BuildConfig
    credit: dict[str, float] = field(default_factory=dict)
    page_heat: dict[tuple[int, int], float] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def query_budget(self, stakes: float, ne: float = 0.0) -> float:
        c = self.cfg.control
        budget = c.base_budget * (1.0 + c.stakes_gain * stakes) * (1.0 + c.ne_gain * max(0.0, ne))
        return min(budget, c.max_budget_per_query)

    def cost_scalar(self, bid: Bid) -> float:
        prices = self.cfg.control.prices
        return sum(prices.get(r, 0.0) * amount for r, amount in bid.resources.items()) + 1e-12

    def allocate(self, bids: list[Bid], remaining_flops: float, categories: dict[str, str] | None = None) -> Allocation:
        """Greedy by ``v̂·c_p / cost`` within caps; floors reserved for background categories."""
        c = self.cfg.control
        cats = categories or {}
        cap_flops = min(c.max_flops_per_beat, remaining_flops)
        floors = {"subconscious": c.floor_subconscious * cap_flops, "error_monitor": c.floor_error_monitor * cap_flops}
        spent = {"flops": 0.0, "bytes": 0.0}
        funded, rejected = [], []

        def density(b: Bid) -> float:
            return b.predicted_value * self.credit.get(b.process_id, 1.0) / self.cost_scalar(b)

        def fits(b: Bid, limit: float) -> bool:
            return (spent["flops"] + b.resources.get("flops", 0.0) <= limit
                    and spent["bytes"] + b.resources.get("bytes", 0.0) <= c.max_bytes_per_beat)

        # 1) floors: background categories first, each up to its reserved share
        for cat, share in floors.items():
            used = 0.0
            for b in sorted([b for b in bids if cats.get(b.process_id) == cat], key=density, reverse=True):
                need = b.resources.get("flops", 0.0)
                if used + need <= share and fits(b, cap_flops):
                    funded.append(b)
                    used += need
                    spent["flops"] += need
                    spent["bytes"] += b.resources.get("bytes", 0.0)
        # 2) everything else by value density
        for b in sorted([b for b in bids if b not in funded], key=density, reverse=True):
            if b.predicted_value <= 0.0:
                rejected.append(b)
                continue
            if fits(b, cap_flops):
                funded.append(b)
                spent["flops"] += b.resources.get("flops", 0.0)
                spent["bytes"] += b.resources.get("bytes", 0.0)
            else:
                rejected.append(b)
        if not funded and bids and remaining_flops > 0:  # always make progress on the best bid
            best = max(bids, key=density)
            if best.predicted_value > 0:
                funded.append(best)
                rejected = [b for b in rejected if b is not best]
                spent["flops"] += best.resources.get("flops", 0.0)
        self.history.append({"funded": len(funded), "rejected": len(rejected), "flops": spent["flops"]})
        return Allocation(funded, rejected, spent["flops"], spent["bytes"])

    def update_credit(self, process_id: str, predicted: float, realized: float) -> float:
        c = self.cfg.control
        ratio = min(2.0, max(0.0, realized / max(predicted, 1e-6)))
        old = self.credit.get(process_id, 1.0)
        new = min(c.c_max, max(c.c_min, (1 - c.eta_c) * old + c.eta_c * ratio))
        self.credit[process_id] = new
        return new

    def note_page(self, store_id: int, page: int, heat: float = 1.0) -> None:
        """Tier management input (07 §1.6): page heat drives promotion/demotion."""
        key = (store_id, page)
        self.page_heat[key] = 0.9 * self.page_heat.get(key, 0.0) + heat

    def hot_pages(self, threshold: float = 1.0) -> list[tuple[int, int]]:
        return [k for k, v in self.page_heat.items() if v >= threshold]
