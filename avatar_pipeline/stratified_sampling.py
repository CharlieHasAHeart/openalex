from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class SamplingThresholds:
    topic_common_min: float = 500.0
    topic_medium_min: float = 50.0
    affiliation_short_max: int = 16
    affiliation_medium_max: int = 53
    affiliation_long_max: int = 145


@dataclass(frozen=True)
class SamplingConfig:
    development_size: int = 2000
    frozen_eval_size: int = 2000
    shadow_size: int = 10000
    seed: int = 42


def _bucket_country(code: str | None) -> str:
    if not code:
        return "UNKNOWN"
    code = code.strip().upper()
    if not code:
        return "UNKNOWN"
    if code in {"US", "CN", "GB", "JP"}:
        return code
    return "OTHER"


def _bucket_institution_type(inst_type: str | None) -> str:
    if not inst_type:
        return "unknown"
    normalized = inst_type.strip().lower()
    if not normalized:
        return "unknown"
    if normalized in {"education", "healthcare", "facility", "company", "nonprofit"}:
        return normalized
    return "other"


def _bucket_name_script(name: str | None) -> str:
    if not name:
        return "unknown"
    for ch in name:
        if ord(ch) > 127:
            return "non_ascii"
    return "ascii_only"


def _bucket_topic_rarity(median_topic_count: float | None, thresholds: SamplingThresholds) -> str:
    if median_topic_count is None:
        return "unknown"
    if median_topic_count > thresholds.topic_common_min:
        return "common"
    if median_topic_count >= thresholds.topic_medium_min:
        return "medium"
    return "rare"


def _bucket_affiliation_length(length: int | None, thresholds: SamplingThresholds) -> str:
    if length is None:
        return "unknown"
    if length <= thresholds.affiliation_short_max:
        return "short"
    if length <= thresholds.affiliation_medium_max:
        return "medium"
    if length <= thresholds.affiliation_long_max:
        return "long"
    return "very_long"


def build_sampling_labels(
    row: dict[str, Any],
    thresholds: SamplingThresholds,
    include_affiliation_bucket: bool = True,
) -> dict[str, Any]:
    dominant_domain = row.get("dominant_domain") or "Unknown"
    country_bucket = _bucket_country(row.get("institution_country_code"))
    inst_type_bucket = _bucket_institution_type(row.get("institution_type"))
    name_script_bucket = _bucket_name_script(row.get("display_name"))
    topic_rarity_bucket = _bucket_topic_rarity(row.get("median_topic_count"), thresholds)
    labels = {
        "author_id": row.get("author_id"),
        "display_name": row.get("display_name"),
        "dominant_domain": dominant_domain,
        "institution_country_bucket": country_bucket,
        "institution_type_bucket": inst_type_bucket,
        "name_script_bucket": name_script_bucket,
        "topic_rarity_bucket": topic_rarity_bucket,
    }
    if include_affiliation_bucket:
        labels["affiliation_length_bucket"] = _bucket_affiliation_length(
            row.get("affiliations_len"),
            thresholds,
        )
    return labels


def build_stratum_key(labels: dict[str, Any]) -> str:
    return "|".join(
        [
            str(labels.get("dominant_domain") or "Unknown"),
            str(labels.get("institution_country_bucket") or "UNKNOWN"),
            str(labels.get("name_script_bucket") or "unknown"),
            str(labels.get("topic_rarity_bucket") or "unknown"),
        ]
    )


def _stratified_sample(
    rows: list[dict[str, Any]],
    total: int,
    seed: int,
    min_per_stratum: int = 1,
) -> list[dict[str, Any]]:
    if total <= 0:
        return []
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["sampling_stratum"]].append(row)

    strata = list(groups.keys())
    rng = random.Random(seed)

    # Shuffle each stratum for stable sampling.
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    if not strata:
        return []

    allocation: dict[str, int] = {key: 0 for key in strata}

    # First pass: allocate minimum per stratum if possible.
    remaining = total
    if remaining >= len(strata) * min_per_stratum:
        for key in strata:
            allocation[key] = min_per_stratum
        remaining -= len(strata) * min_per_stratum

    # Second pass: proportional allocation based on stratum sizes.
    total_available = sum(len(groups[key]) for key in strata)
    if total_available == 0:
        return []

    for key in strata:
        if remaining <= 0:
            break
        proportion = len(groups[key]) / total_available
        extra = int(round(proportion * remaining))
        if extra > 0:
            allocation[key] += extra

    # Adjust if we allocated too many due to rounding.
    allocated = sum(allocation.values())
    if allocated > total:
        overflow = allocated - total
        for key in sorted(strata, key=lambda k: allocation[k], reverse=True):
            if overflow <= 0:
                break
            if allocation[key] > 0:
                allocation[key] -= 1
                overflow -= 1

    # Fill remaining slots by cycling through larger strata.
    allocated = sum(allocation.values())
    if allocated < total:
        remaining = total - allocated
        order = sorted(strata, key=lambda k: len(groups[k]), reverse=True)
        idx = 0
        while remaining > 0 and order:
            key = order[idx % len(order)]
            allocation[key] += 1
            remaining -= 1
            idx += 1

    sampled: list[dict[str, Any]] = []
    for key in strata:
        take = min(allocation[key], len(groups[key]))
        if take > 0:
            sampled.extend(groups[key][:take])

    return sampled


def build_sampling_sets(
    rows: list[dict[str, Any]],
    config: SamplingConfig,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(config.seed)
    pool = list(rows)
    rng.shuffle(pool)

    dev = _stratified_sample(pool, config.development_size, seed=config.seed)
    dev_ids = {row["author_id"] for row in dev}

    remaining = [row for row in pool if row["author_id"] not in dev_ids]
    frozen = _stratified_sample(remaining, config.frozen_eval_size, seed=config.seed + 1)
    frozen_ids = {row["author_id"] for row in frozen}

    remaining_after_frozen = [row for row in remaining if row["author_id"] not in frozen_ids]
    shadow_pool = remaining_after_frozen if len(remaining_after_frozen) >= config.shadow_size else pool
    shadow = _stratified_sample(shadow_pool, config.shadow_size, seed=config.seed + 2)

    return {
        "development": dev,
        "frozen_eval": frozen,
        "shadow": shadow,
    }


def summarize_sampling(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    by_domain: Counter[str] = Counter()
    by_country: Counter[str] = Counter()
    by_topic_rarity: Counter[str] = Counter()
    by_name_script: Counter[str] = Counter()

    for row in rows:
        total += 1
        by_domain[str(row.get("dominant_domain") or "Unknown")] += 1
        by_country[str(row.get("institution_country_bucket") or "UNKNOWN")] += 1
        by_topic_rarity[str(row.get("topic_rarity_bucket") or "unknown")] += 1
        by_name_script[str(row.get("name_script_bucket") or "unknown")] += 1

    return {
        "total": total,
        "dominant_domain": dict(by_domain),
        "institution_country_bucket": dict(by_country),
        "topic_rarity_bucket": dict(by_topic_rarity),
        "name_script_bucket": dict(by_name_script),
    }


def write_sampling_outputs(
    output_dir: str,
    sets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}

    for set_name, rows in sets.items():
        output_rows = [dict(row, sample_set=set_name) for row in rows]
        output_path = Path(output_dir) / f"{set_name}_set.json"
        output_path.write_text(json.dumps(output_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        summary[set_name] = summarize_sampling(output_rows)

    summary_path = Path(output_dir) / "sampling_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "counts": {name: len(rows) for name, rows in sets.items()},
    }
