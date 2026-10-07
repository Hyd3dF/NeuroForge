"""SEF shards (JSON Lines) and dataset manifests (02 §3)."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from srm.data.sef import SEF_VERSION, RecordBase, parse_record, record_to_dict
from srm.util.canonical import canonical_hash, sha256_hex


@dataclass
class Dataset:
    """An in-memory SEF dataset produced by a generator or converter."""

    dataset_id: str
    records: list[RecordBase] = field(default_factory=list)
    generator: str = ""
    generator_version: str = ""
    seed: int | None = None
    meta: dict[str, object] = field(default_factory=dict)

    def add(self, record: RecordBase) -> RecordBase:
        record.with_hash()
        self.records.append(record)
        return record

    def by_kind(self, kind: str) -> list[RecordBase]:
        return [r for r in self.records if r.kind == kind]

    def counts(self) -> dict[str, int]:
        return dict(Counter(r.kind for r in self.records))

    def content_digest(self) -> str:
        return canonical_hash([r.content_hash for r in self.records])


def write_dataset(dataset: Dataset, root: str | Path, shard_records: int = 50_000) -> dict[str, object]:
    """Write ``sef/<category>/<shard>.jsonl`` files plus ``dataset_manifest.json``."""
    root = Path(root)
    by_cat: dict[str, list[RecordBase]] = {}
    for r in dataset.records:
        by_cat.setdefault(r.data_category, []).append(r)
    shards = []
    for cat, recs in sorted(by_cat.items()):
        cat_dir = root / "sef" / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(0, len(recs), shard_records):
            chunk = recs[idx : idx + shard_records]
            path = cat_dir / f"{idx // shard_records:05d}.jsonl"
            text = "".join(json.dumps(record_to_dict(r), ensure_ascii=False, sort_keys=True) + "\n" for r in chunk)
            path.write_text(text, encoding="utf-8")
            shards.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": sha256_hex(text),
                    "records": len(chunk),
                    "kinds": dict(Counter(r.kind for r in chunk)),
                }
            )
    total = len(dataset.records)
    cats = Counter(r.data_category for r in dataset.records)
    manifest = {
        "dataset_id": dataset.dataset_id,
        "sef_version": SEF_VERSION,
        "generator": dataset.generator,
        "generator_version": dataset.generator_version,
        "seed": dataset.seed,
        "shards": shards,
        "record_counts": dataset.counts(),
        "category_proportions": {k: v / total for k, v in sorted(cats.items())} if total else {},
        "content_digest": dataset.content_digest(),
        "meta": dataset.meta,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def iter_shard(path: str | Path) -> Iterator[RecordBase]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield parse_record(json.loads(line))


def read_dataset(root: str | Path, verify: bool = True) -> Dataset:
    root = Path(root)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_hash")
    if verify and canonical_hash(manifest) != claimed:
        raise ValueError("dataset manifest hash mismatch")
    ds = Dataset(
        dataset_id=manifest["dataset_id"], generator=manifest["generator"],
        generator_version=manifest["generator_version"], seed=manifest["seed"], meta=manifest["meta"],
    )
    for shard in manifest["shards"]:
        path = root / shard["path"]
        if verify and sha256_hex(path.read_text(encoding="utf-8")) != shard["sha256"]:
            raise ValueError(f"shard hash mismatch: {shard['path']}")
        ds.records.extend(iter_shard(path))
    return ds


def iter_records(datasets: Iterable[Dataset]) -> Iterator[RecordBase]:
    for ds in datasets:
        yield from ds.records
