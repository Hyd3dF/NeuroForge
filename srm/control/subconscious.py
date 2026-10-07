"""Subconscious (07 §4): background processes that cannot commit; outputs are SUGGESTED.

F0 processes: priming (spreading activation from touched records), prefetch bids for the
pages of primed records, and background stage-1 matching on novel percepts.  Habit
proposals (compiled skills) join in M7.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from srm.control.gate import BufferItem
from srm.memory.store import split_record_id


@dataclass
class Subconscious:
    cfg: Any
    log: list[dict[str, Any]] = field(default_factory=list)

    def prime(self, ctx: Any, seeds: set[int], k: int = 16) -> list[BufferItem]:
        if not seeds:
            return []
        act = ctx.memory.spread({rid: 1.0 for rid in list(seeds)[:64]})
        items = []
        for rid, a in sorted(act.items(), key=lambda t: -t[1]):
            if rid in seeds or len(items) >= k:
                continue
            rec = ctx.memory.record(rid)
            items.append(BufferItem(rid, "priming", relevance=a, novelty=0.0,
                                    trust=max((p.get("trust", 0.5) for p in rec.provenance), default=0.5),
                                    code=rec.content_code))
        self.log.append({"process": "priming", "items": len(items)})
        return items

    def prefetch_pages(self, items: list[BufferItem]) -> list[tuple[int, int]]:
        pages = set()
        for it in items:
            store_id, local = split_record_id(it.record_id)
            pages.add((store_id, local // self.cfg.memory.page_records))
        self.log.append({"process": "prefetch", "pages": len(pages)})
        return sorted(pages)

    def match_percepts(self, ctx: Any, percept: Any, k: int = 4) -> list[BufferItem]:
        items = []
        for ch in getattr(percept, "chunks", []):
            if ch.known:
                continue
            for rid, score in ctx.memory.retrieve(ch.code, k=k):
                rec = ctx.memory.record(rid)
                items.append(BufferItem(rid, "stage1", error=ch.epsilon0, relevance=score, novelty=1.0,
                                        trust=max((p.get("trust", 0.5) for p in rec.provenance), default=0.5),
                                        code=rec.content_code))
        self.log.append({"process": "stage1", "items": len(items)})
        return items
