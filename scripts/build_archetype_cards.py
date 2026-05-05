#!/usr/bin/env python3
"""Deterministic AGLLM archetype-card builder.

This script converts released cluster metadata plus item metadata into the
fixed archetype cards described in the manuscript. It uses only Python's
standard library so the release path remains lightweight and reproducible.
"""

from __future__ import annotations

import argparse
import csv
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


DURATION_BUCKETS = [
    (0.0, 15.0, "0-15s"),
    (15.0, 30.0, "15-30s"),
    (30.0, 60.0, "30-60s"),
    (60.0, float("inf"), "60s+"),
]


def normalize_text(value: str) -> str:
    return " ".join(
        unicodedata.normalize("NFKC", value or "").strip().split()
    )


def duration_bucket(seconds: float) -> str:
    for lower, upper, label in DURATION_BUCKETS:
        if lower <= seconds < upper:
            return label
    raise ValueError(f"unhandled duration: {seconds}")


def popularity_bucket(interaction_count: float, cutpoints: tuple[float, float, float, float]) -> str:
    q1, q2, q3, q4 = cutpoints
    if interaction_count < q1:
        return "very_low"
    if interaction_count < q2:
        return "low"
    if interaction_count < q3:
        return "medium"
    if interaction_count < q4:
        return "high"
    return "very_high"


def linear_quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, q)) * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def modal_category(rows: list[dict[str, str]]) -> tuple[str, str]:
    pairs = Counter(
        (
            normalize_text(row.get("category_level1", "")),
            normalize_text(row.get("category_level2", "")),
        )
        for row in rows
    )
    return sorted(pairs.items(), key=lambda item: (-item[1], item[0]))[0][0]


def build_cards(item_rows: list[dict[str, str]], cluster_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_item_id = {row["item_id"]: row for row in item_rows}
    items_by_cluster: dict[str, list[dict[str, str]]] = defaultdict(list)
    medoid_by_cluster: dict[str, str] = {}

    for row in cluster_rows:
        cluster_id = row["cluster_id"]
        item_id = row["item_id"]
        items_by_cluster[cluster_id].append(by_item_id[item_id])
        if row.get("is_medoid", "0") == "1":
            medoid_by_cluster[cluster_id] = item_id

    interaction_counts = sorted(float(row.get("interaction_count", "0") or 0.0) for row in item_rows)
    if interaction_counts:
        count_cutpoints = (
            linear_quantile(interaction_counts, 0.2),
            linear_quantile(interaction_counts, 0.4),
            linear_quantile(interaction_counts, 0.6),
            linear_quantile(interaction_counts, 0.8),
        )
    else:
        count_cutpoints = (0.0, 0.0, 0.0, 0.0)

    cards = []
    for archetype_index, cluster_id in enumerate(
        sorted(items_by_cluster, key=lambda cid: (-len(items_by_cluster[cid]), int(cid))),
        start=0,
    ):
        rows = items_by_cluster[cluster_id]
        medoid_item_id = medoid_by_cluster.get(cluster_id)
        if medoid_item_id is None:
            medoid_item_id = min((row["item_id"] for row in rows), key=lambda value: int(value))
        medoid_item = by_item_id[medoid_item_id]
        topic = normalize_text(medoid_item.get("caption", ""))
        if not topic:
            topic = normalize_text(medoid_item.get("category_level2", "")) or normalize_text(
                medoid_item.get("category_level1", "")
            )
        cat1, cat2 = modal_category(rows)
        duration_value = medoid_item.get("duration_sec", medoid_item.get("duration_seconds", "0"))
        duration = duration_bucket(float(duration_value))
        popularity = popularity_bucket(float(medoid_item.get("interaction_count", "0") or 0.0), count_cutpoints)
        cards.append(
            {
                "archetype_id": archetype_index,
                "cluster_id": archetype_index,
                "medoid_item_id": int(medoid_item["item_id"]),
                "schema_version": "agllm_card_v2",
                "index_base": 0,
                "topic_tag": topic,
                "category_level1": cat1,
                "category_level2": cat2,
                "duration_bucket": duration,
                "popularity_bucket": popularity,
            }
        )
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True, help="CSV with item metadata")
    parser.add_argument("--clusters", type=Path, required=True, help="CSV with cluster assignments and medoid flags")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    cards = build_cards(load_rows(args.items), load_rows(args.clusters))
    args.output.write_text(json.dumps(cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
