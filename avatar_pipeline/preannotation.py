from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ALLOWED_LABELS = {
    "correct",
    "incorrect",
    "uncertain",
    "no_avatar_available",
}

HIGH_RISK_FLAGS = {
    "small_margin",
    "low_llm_confidence",
    "weak_structure_evidence",
    "cluster_support_weak",
}


def load_benchmark_package(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"benchmark package not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("benchmark package must be a JSON array")
    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


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
            return [text]
    return []


def suggest_prelabel(row: dict[str, Any]) -> dict[str, Any]:
    review_recommendation = str(row.get("review_recommendation") or "").strip()
    review_summary = row.get("review_summary")
    review_risk_flags = _as_list(row.get("review_risk_flags"))
    acceptance_score = row.get("acceptance_score")
    fallback_used = bool(row.get("fallback_used"))
    decision_mode = str(row.get("decision_mode") or "").strip()

    suggested_label = "uncertain"
    suggested_reason = "insufficient_or_risky_signal"
    prelabel_confidence = 0.4
    needs_human_review = True
    prelabel_source = "low_confidence_fallback"

    high_risk = any(flag in HIGH_RISK_FLAGS for flag in review_risk_flags)

    if review_recommendation == "auto_accept" and not high_risk and not fallback_used:
        if acceptance_score is None or float(acceptance_score) >= 0.75:
            suggested_label = "correct"
            suggested_reason = "auto_accept_without_high_risk"
            prelabel_confidence = 0.9
            needs_human_review = False
            prelabel_source = "rule_based_from_review_recommendation"
    elif review_recommendation == "ambiguous" or decision_mode == "ambiguous":
        suggested_label = "uncertain"
        suggested_reason = "ambiguous_or_low_confidence"
        prelabel_confidence = 0.25
        needs_human_review = True
        prelabel_source = "rule_based_from_risk_flags"
    elif review_recommendation == "needs_review":
        suggested_label = "correct" if not high_risk else "uncertain"
        suggested_reason = "needs_review_flag"
        prelabel_confidence = 0.6 if suggested_label == "correct" else 0.35
        needs_human_review = True
        prelabel_source = "rule_based_from_review_recommendation"

    return {
        "suggested_label": suggested_label,
        "suggested_reason": suggested_reason,
        "prelabel_confidence": round(float(prelabel_confidence), 3),
        "needs_human_review": bool(needs_human_review),
        "prelabel_source": prelabel_source,
        "review_recommendation": review_recommendation or None,
        "review_summary": review_summary,
        "review_risk_flags": review_risk_flags,
        "decision_mode": decision_mode or None,
        "acceptance_score": acceptance_score,
        "fallback_used": fallback_used,
    }


def build_preannotation_rows(package_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in package_rows:
        suggestion = suggest_prelabel(row)
        enriched = dict(row)
        enriched.update(suggestion)
        rows.append(enriched)
    return rows


def write_preannotation_file(rows: Iterable[dict[str, Any]], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def write_preannotation_review_sheet(rows: Iterable[dict[str, Any]], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "author_id",
        "display_name",
        "dominant_domain",
        "sampling_stratum",
        "review_recommendation",
        "review_summary",
        "review_risk_flags",
        "suggested_label",
        "suggested_reason",
        "prelabel_confidence",
        "needs_human_review",
        "final_label",
        "reviewer_note",
    ]
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            payload = {k: row.get(k) for k in fieldnames}
            payload["final_label"] = ""
            payload["reviewer_note"] = ""
            writer.writerow(payload)


def summarize_preannotations(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    label_counter: Counter[str] = Counter()
    needs_review_count = 0
    by_recommendation: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        total += 1
        label = str(row.get("suggested_label") or "uncertain")
        label_counter[label] += 1
        if row.get("needs_human_review"):
            needs_review_count += 1
        rec = str(row.get("review_recommendation") or "unknown")
        by_recommendation[rec][label] += 1

    return {
        "total": total,
        "suggested_label_distribution": dict(label_counter),
        "needs_human_review_count": needs_review_count,
        "needs_human_review_ratio": round(needs_review_count / total, 4) if total else 0.0,
        "by_review_recommendation": {k: dict(v) for k, v in by_recommendation.items()},
    }
