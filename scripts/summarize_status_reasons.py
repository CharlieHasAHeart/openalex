#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from collections import Counter


PIPELINE_PATTERN = re.compile(
    r"pipeline_result\s+author_id=(?P<author_id>\S+)\s+orcid=(?P<orcid>\S+)\s+status=(?P<status>\S+)\s+qid=(?P<qid>\S+)\s+commons_file=(?P<commons_file>\S+)(?:\s+error_message=(?P<error_message>.*))?$"
)
STATS_PATTERN = re.compile(r"stats=(\{.*\})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize reason breakdown for no_match/error from pipeline log"
    )
    parser.add_argument("--log", default="logs/systemd_pipeline.log")
    parser.add_argument(
        "--status",
        action="append",
        choices=["no_match", "error"],
        default=["no_match", "error"],
        help="Statuses to summarize (repeatable).",
    )
    parser.add_argument(
        "--latest-stats",
        action="store_true",
        help="Use latest stats line as expected totals for comparison.",
    )
    return parser.parse_args()


def parse_latest_stats(log_path: str) -> dict[str, int]:
    latest: dict[str, int] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = STATS_PATTERN.search(line)
            if not m:
                continue
            try:
                obj = ast.literal_eval(m.group(1))
                if isinstance(obj, dict):
                    latest = {str(k): int(v) for k, v in obj.items() if isinstance(v, int)}
            except Exception:
                continue
    return latest


def summarize(log_path: str, statuses: set[str]) -> tuple[dict[str, Counter[str]], dict[str, int]]:
    reasons: dict[str, Counter[str]] = {s: Counter() for s in statuses}
    totals: dict[str, int] = {s: 0 for s in statuses}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = PIPELINE_PATTERN.search(line.rstrip("\n"))
            if not m:
                continue
            status = m.group("status")
            if status not in statuses:
                continue
            totals[status] += 1
            msg = (m.group("error_message") or "").strip()
            reason = msg if msg else "missing_error_message_in_log"
            reasons[status][reason] += 1
    return reasons, totals


def print_status_block(
    status: str,
    reason_counter: Counter[str],
    observed_total: int,
    expected_total: int | None,
) -> None:
    print(f"[{status}]")
    print(f"observed_total={observed_total}")
    if expected_total is not None:
        print(f"expected_total={expected_total}")
        print(f"coverage_gap={expected_total - observed_total}")
    print("reasons:")
    for reason, count in reason_counter.most_common():
        print(f"- {reason}: {count}")
    print("")


def main() -> int:
    args = parse_args()
    statuses = set(args.status)
    reason_counter_by_status, observed_totals = summarize(args.log, statuses)

    expected = parse_latest_stats(args.log) if args.latest_stats else {}

    for status in sorted(statuses):
        print_status_block(
            status=status,
            reason_counter=reason_counter_by_status[status],
            observed_total=observed_totals.get(status, 0),
            expected_total=expected.get(status) if expected else None,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
