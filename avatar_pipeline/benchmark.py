from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VALID_LABELS = {"correct", "incorrect", "uncertain", "no_avatar_available"}


def _load_json_or_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl":
        rows = []
        for i, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                raise ValueError(f"invalid jsonl at line {i}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    payload = json.loads(text)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def load_benchmark_authors(path: str) -> list[dict[str, Any]]:
    rows = _load_json_or_jsonl(path)
    out: list[dict[str, Any]] = []
    for row in rows:
        author_id = row.get("author_id")
        if author_id is None:
            continue
        out.append(
            {
                "author_id": str(author_id),
                "bucket": row.get("bucket"),
                "note": row.get("note"),
                "expected_difficulty": row.get("expected_difficulty"),
            }
        )
    return out


def load_annotations(path: str) -> dict[str, dict[str, Any]]:
    rows = _load_json_or_jsonl(path)
    annotations: dict[str, dict[str, Any]] = {}
    for row in rows:
        author_id = row.get("author_id")
        if author_id is None:
            continue
        label = str(row.get("label") or "").strip().lower()
        if label not in VALID_LABELS:
            continue
        annotations[str(author_id)] = {
            "label": label,
            "reviewer_note": row.get("reviewer_note"),
            "expected_source_type": row.get("expected_source_type"),
            "expected_image_url": row.get("expected_image_url"),
        }
    return annotations


def _risk_flags(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        try:
            decoded = json.loads(v)
            if isinstance(decoded, list):
                return [str(x) for x in decoded]
        except Exception:
            pass
    return []


def build_benchmark_rows(
    result_rows: list[dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    benchmark_authors: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    benchmark_map = {str(item["author_id"]): item for item in (benchmark_authors or []) if item.get("author_id") is not None}
    filtered = result_rows
    if benchmark_map:
        filtered = [r for r in result_rows if str(r.get("author_id")) in benchmark_map]

    joined: list[dict[str, Any]] = []
    for row in filtered:
        author_id = str(row.get("author_id"))
        ann = annotations.get(author_id, {})
        meta = benchmark_map.get(author_id, {})
        joined.append(
            {
                "author_id": author_id,
                "status": row.get("status"),
                "decision_mode": row.get("decision_mode"),
                "review_recommendation": row.get("review_recommendation"),
                "acceptance_score": row.get("acceptance_score"),
                "fallback_used": bool(row.get("fallback_used")),
                "selected_candidate_id": row.get("selected_candidate_id"),
                "updated_at": row.get("updated_at"),
                "review_summary": row.get("review_summary"),
                "review_risk_flags": _risk_flags(row.get("review_risk_flags")),
                "label": ann.get("label"),
                "reviewer_note": ann.get("reviewer_note"),
                "expected_source_type": ann.get("expected_source_type"),
                "expected_image_url": ann.get("expected_image_url"),
                "bucket": meta.get("bucket"),
                "note": meta.get("note"),
                "expected_difficulty": meta.get("expected_difficulty"),
            }
        )
    return joined


def _precision(rows: list[dict[str, Any]]) -> float | None:
    correct = sum(1 for r in rows if r.get("label") == "correct")
    incorrect = sum(1 for r in rows if r.get("label") == "incorrect")
    denom = correct + incorrect
    if denom == 0:
        return None
    return round(correct / denom, 4)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [r for r in rows if r.get("label") in VALID_LABELS]
    recommendation_counter = Counter(str(r.get("review_recommendation") or "unknown") for r in rows)
    decision_counter = Counter(str(r.get("decision_mode") or "unknown") for r in rows)
    label_counter = Counter(str(r.get("label") or "unlabeled") for r in rows)
    risk_counter: Counter[str] = Counter()
    fallback_rows = [r for r in labeled if r.get("fallback_used")]
    for r in rows:
        for flag in r.get("review_risk_flags") or []:
            risk_counter[str(flag)] += 1

    return {
        "labeled_total": len(labeled),
        "correct_count": label_counter.get("correct", 0),
        "incorrect_count": label_counter.get("incorrect", 0),
        "uncertain_count": label_counter.get("uncertain", 0),
        "no_avatar_available_count": label_counter.get("no_avatar_available", 0),
        "overall_precision": _precision(labeled),
        "auto_accept_precision": _precision([r for r in labeled if r.get("review_recommendation") == "auto_accept"]),
        "needs_review_precision": _precision([r for r in labeled if r.get("review_recommendation") == "needs_review"]),
        "fallback_precision": _precision(fallback_rows),
        "recommendation_distribution": dict(recommendation_counter),
        "decision_mode_distribution": dict(decision_counter),
        "label_distribution": dict(label_counter),
        "fallback_used_count": len(fallback_rows),
        "risk_flag_frequency": dict(risk_counter.most_common(50)),
    }


def build_error_buckets(rows: list[dict[str, Any]]) -> dict[str, Any]:
    incorrect = [r for r in rows if r.get("label") == "incorrect"]
    by_recommendation = Counter(str(r.get("review_recommendation") or "unknown") for r in incorrect)
    by_decision_mode = Counter(str(r.get("decision_mode") or "unknown") for r in incorrect)
    by_fallback_used = Counter(str(bool(r.get("fallback_used"))).lower() for r in incorrect)
    risk_counter: Counter[str] = Counter()
    source_type_counter: Counter[str] = Counter()

    for r in incorrect:
        for flag in r.get("review_risk_flags") or []:
            risk_counter[str(flag)] += 1
        summary = str(r.get("review_summary") or "").lower()
        for key in ("orcid_external_link", "institution_profile", "institution_directory", "lab_people_page", "generic_search_result"):
            if key in summary:
                source_type_counter[key] += 1

    return {
        "incorrect_total": len(incorrect),
        "by_review_recommendation": dict(by_recommendation),
        "by_decision_mode": dict(by_decision_mode),
        "by_fallback_used": dict(by_fallback_used),
        "by_risk_flags": dict(risk_counter.most_common(50)),
        "by_source_type_hint": dict(source_type_counter),
    }


def sample_errors(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    incorrect = [r for r in rows if r.get("label") == "incorrect"]
    out: list[dict[str, Any]] = []
    for row in incorrect[: max(1, limit)]:
        out.append(
            {
                "author_id": row.get("author_id"),
                "review_recommendation": row.get("review_recommendation"),
                "decision_mode": row.get("decision_mode"),
                "acceptance_score": row.get("acceptance_score"),
                "fallback_used": row.get("fallback_used"),
                "review_summary": row.get("review_summary"),
                "review_risk_flags": row.get("review_risk_flags"),
                "label": row.get("label"),
                "reviewer_note": row.get("reviewer_note"),
            }
        )
    return out


def build_benchmark_report(
    joined_rows: list[dict[str, Any]],
    benchmark_meta: dict[str, Any] | None = None,
    error_sample_limit: int = 20,
) -> dict[str, Any]:
    return {
        "benchmark_meta": benchmark_meta or {},
        "metrics": compute_metrics(joined_rows),
        "error_buckets": build_error_buckets(joined_rows),
        "sample_errors": sample_errors(joined_rows, limit=error_sample_limit),
    }
