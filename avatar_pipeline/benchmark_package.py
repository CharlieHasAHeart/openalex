from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


LABEL_OPTIONS = {
    "correct",
    "incorrect",
    "uncertain",
    "no_avatar_available",
}


def load_sampling_set(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"sampling set file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("sampling set must be a JSON array")
    rows: list[dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict):
            rows.append(item)
    return rows


def build_package_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    packaged: list[dict[str, Any]] = []
    for row in rows:
        packaged.append(
            {
                "author_id": row.get("author_id"),
                "display_name": row.get("display_name"),
                "sample_set": row.get("sample_set"),
                "dominant_domain": row.get("dominant_domain"),
                "institution_country_bucket": row.get("institution_country_bucket"),
                "institution_type_bucket": row.get("institution_type_bucket"),
                "name_script_bucket": row.get("name_script_bucket"),
                "topic_rarity_bucket": row.get("topic_rarity_bucket"),
                "affiliation_length_bucket": row.get("affiliation_length_bucket"),
                "sampling_stratum": row.get("sampling_stratum"),
                "annotation_status": "unlabeled",
                "label": "",
                "reviewer_note": "",
                "expected_source_type": "",
                "expected_image_url": "",
            }
        )
    return packaged


def write_annotation_template(rows: Iterable[dict[str, Any]], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "author_id": row.get("author_id"),
                        "label": "",
                        "reviewer_note": "",
                        "expected_source_type": "",
                        "expected_image_url": "",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_review_sheet_csv(rows: Iterable[dict[str, Any]], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "author_id",
        "display_name",
        "dominant_domain",
        "institution_country_bucket",
        "institution_type_bucket",
        "name_script_bucket",
        "topic_rarity_bucket",
        "affiliation_length_bucket",
        "sampling_stratum",
    ]
    with p.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_package_json(rows: Iterable[dict[str, Any]], output_path: str) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(list(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def build_benchmark_package(
    sampling_set_path: str,
    package_name: str,
    output_dir: str,
) -> dict[str, str]:
    rows = load_sampling_set(sampling_set_path)
    packaged = build_package_rows(rows)

    package_path = str(Path(output_dir) / f"{package_name}_package.json")
    template_path = str(Path(output_dir) / f"{package_name}_annotations_template.jsonl")
    review_path = str(Path(output_dir) / f"{package_name}_review_sheet.csv")

    write_package_json(packaged, package_path)
    write_annotation_template(packaged, template_path)
    write_review_sheet_csv(packaged, review_path)

    return {
        "package_path": package_path,
        "template_path": template_path,
        "review_sheet_path": review_path,
        "label_options": ",".join(sorted(LABEL_OPTIONS)),
    }


def validate_annotation_file(path: str, expected_author_ids: set[str] | None = None) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"annotation file not found: {path}")
    rows = p.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    seen: set[str] = set()
    labeled = 0
    for i, line in enumerate(rows, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            errors.append(f"line {i}: invalid json ({exc})")
            continue
        author_id = obj.get("author_id")
        if author_id is None or str(author_id).strip() == "":
            errors.append(f"line {i}: missing author_id")
            continue
        author_id = str(author_id)
        if author_id in seen:
            errors.append(f"line {i}: duplicate author_id {author_id}")
        seen.add(author_id)
        label = str(obj.get("label") or "").strip()
        if label:
            if label not in LABEL_OPTIONS:
                errors.append(f"line {i}: invalid label {label}")
            else:
                labeled += 1
    if expected_author_ids is not None:
        missing = expected_author_ids - seen
        if missing:
            errors.append(f"missing {len(missing)} author_id entries")
    return {
        "total_lines": len(rows),
        "unique_authors": len(seen),
        "labeled": labeled,
        "errors": errors,
    }
