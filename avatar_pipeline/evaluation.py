from __future__ import annotations

import json
import random
from collections import Counter
from typing import Any


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
            if isinstance(decoded, list):
                return [str(v) for v in decoded]
        except Exception:
            pass
    return []


def summarize_decisions(rows: list[dict[str, Any]], risk_top_n: int = 20) -> dict[str, Any]:
    total = len(rows)
    by_recommendation: Counter[str] = Counter()
    by_decision_mode: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    risk_counter: Counter[str] = Counter()
    fallback_used_count = 0

    for row in rows:
        recommendation = str(row.get("review_recommendation") or "unknown")
        decision_mode = str(row.get("decision_mode") or "unknown")
        status = str(row.get("status") or "unknown")
        by_recommendation[recommendation] += 1
        by_decision_mode[decision_mode] += 1
        by_status[status] += 1
        if bool(row.get("fallback_used")):
            fallback_used_count += 1
        for flag in _as_list(row.get("review_risk_flags")):
            risk_counter[flag] += 1

    def _ratio(v: int) -> float:
        return round((v / total) if total > 0 else 0.0, 4)

    return {
        "total": total,
        "recommendation_distribution": {k: {"count": v, "ratio": _ratio(v)} for k, v in by_recommendation.items()},
        "decision_mode_distribution": {k: {"count": v, "ratio": _ratio(v)} for k, v in by_decision_mode.items()},
        "status_distribution": {k: {"count": v, "ratio": _ratio(v)} for k, v in by_status.items()},
        "fallback_used_count": fallback_used_count,
        "fallback_used_ratio": _ratio(fallback_used_count),
        "risk_flag_top": [{"flag": k, "count": v} for k, v in risk_counter.most_common(max(1, risk_top_n))],
    }


def sample_decisions(rows: list[dict[str, Any]], per_group: int = 20, seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("review_recommendation") or "unknown")
        groups.setdefault(key, []).append(row)
    rng = random.Random(seed)
    sampled: dict[str, list[dict[str, Any]]] = {}
    for key, group_rows in groups.items():
        picks = list(group_rows)
        rng.shuffle(picks)
        sampled[key] = picks[: max(1, per_group)]
    return sampled


def format_review_export(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for row in rows:
        formatted.append(
            {
                "author_id": row.get("author_id"),
                "status": row.get("status"),
                "decision_mode": row.get("decision_mode"),
                "acceptance_score": row.get("acceptance_score"),
                "fallback_used": bool(row.get("fallback_used")),
                "selected_candidate_id": row.get("selected_candidate_id"),
                "updated_at": row.get("updated_at").isoformat() if hasattr(row.get("updated_at"), "isoformat") else row.get("updated_at"),
                "review_recommendation": row.get("review_recommendation"),
                "review_summary": row.get("review_summary"),
                "review_risk_flags": _as_list(row.get("review_risk_flags")),
            }
        )
    return formatted
