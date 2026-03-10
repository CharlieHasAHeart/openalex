from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def _find_run_dir(base_dir: Path, run_id: str) -> Path:
    matches = list(base_dir.glob(f"*/{run_id}"))
    if not matches:
        raise FileNotFoundError(f"run_id not found under {base_dir}: {run_id}")
    return matches[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review local OpenAlex avatar pipeline runs")
    parser.add_argument("--runs-dir", default="runs", help="Base runs directory")
    parser.add_argument("--run-id", required=True, help="Target run id")
    parser.add_argument("--show-summary", action="store_true", help="Print summary.json")
    parser.add_argument("--author-id", default="", help="Find one author record in author_runs.jsonl")
    parser.add_argument("--failures-csv", default="", help="Export failures.jsonl to csv path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = _find_run_dir(Path(args.runs_dir), args.run_id.strip())
    summary_path = run_dir / "summary.json"
    author_runs_path = run_dir / "author_runs.jsonl"
    failures_path = run_dir / "failures.jsonl"

    if args.show_summary:
        print(json.dumps(_load_json(summary_path), ensure_ascii=False, indent=2))

    if args.author_id.strip():
        target = args.author_id.strip()
        found = None
        for row in _iter_jsonl(author_runs_path):
            if str(row.get("author_id") or "").strip() == target:
                found = row
                break
        if found is None:
            raise SystemExit(f"author_id not found in run {args.run_id}: {target}")
        print(json.dumps(found, ensure_ascii=False, indent=2))

    if args.failures_csv.strip():
        output_path = Path(args.failures_csv.strip())
        rows = list(_iter_jsonl(failures_path))
        fieldnames = [
            "author_id",
            "display_name",
            "institution_name",
            "final_status",
            "failure_reason",
            "oss_url",
            "content_sha256",
            "timestamp",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in fieldnames})
        print(str(output_path))

    if not args.show_summary and not args.author_id.strip() and not args.failures_csv.strip():
        print(json.dumps(_load_json(summary_path), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
