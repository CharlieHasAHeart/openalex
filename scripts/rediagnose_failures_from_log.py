#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import logging
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PIPELINE_RE = re.compile(
    r"pipeline_result author_id=(?P<author_id>\S+) orcid=(?P<orcid>\S+) status=(?P<status>\S+) qid=(?P<qid>\S+) commons_file=(?P<commons_file>\S+)(?: error_message=(?P<error_message>.*))?$"
)
STATS_RE = re.compile(r"stats=(\{.*\})")


@dataclass(slots=True)
class FailureRow:
    author_id: str
    orcid: str
    status: str
    original_error_message: str | None
    diagnosed_reason: str
    diagnosed_detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-diagnose no_match/error entries from latest run in log and export CSV"
    )
    parser.add_argument("--log", default="logs/systemd_pipeline.log")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--out-csv", default="logs/failure_rediagnosis.csv")
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--qps", type=float, default=2.0)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def parse_latest_run(log_path: str) -> tuple[list[dict[str, str]], dict[str, int]]:
    runs: list[tuple[list[dict[str, str]], dict[str, int]]] = []
    current: list[dict[str, str]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            m = PIPELINE_RE.search(line)
            if m:
                current.append(
                    {
                        "author_id": m.group("author_id"),
                        "orcid": m.group("orcid"),
                        "status": m.group("status"),
                        "qid": m.group("qid"),
                        "commons_file": m.group("commons_file"),
                        "error_message": (m.group("error_message") or "").strip(),
                    }
                )
                continue

            sm = STATS_RE.search(line)
            if sm:
                stats_raw = sm.group(1)
                stats = ast.literal_eval(stats_raw)
                stats_dict = {str(k): int(v) for k, v in stats.items() if isinstance(v, int)}
                runs.append((current, stats_dict))
                current = []

    if not runs:
        return [], {}
    return runs[-1]


class Diagnoser:
    def __init__(self, config: Any) -> None:
        from avatar_pipeline.commons_client import CommonsClient
        from avatar_pipeline.http import HttpClient, RateLimiter
        from avatar_pipeline.openalex_client import OpenAlexClient
        from avatar_pipeline.wdqs_client import WdqsClient
        from avatar_pipeline.wikidata_api_client import WikidataApiClient

        limiter = RateLimiter(config.global_qps_limit)
        self.http = HttpClient(
            timeout_seconds=config.request_timeout_seconds,
            max_retries=config.max_retries,
            rate_limiter=limiter,
        )
        self.openalex = OpenAlexClient(config.openalex_base_url, self.http, config.openalex_mailto)
        self.wdqs = WdqsClient(config.wdqs_endpoint, self.http)
        self.wikidata = WikidataApiClient(config.wikidata_api_url, self.http)
        self.commons = CommonsClient(config.commons_api_url, self.http)
        self.allowed_mime = config.allowed_mime
        self.thumb_width = config.avatar_thumb_width
        self.qps = config.global_qps_limit
        self._interval = 1.0 / self.qps if self.qps > 0 else 0.0
        self._last = 0.0

    def _wait(self) -> None:
        if self._interval <= 0:
            return
        now = time.monotonic()
        delta = now - self._last
        if delta < self._interval:
            time.sleep(self._interval - delta)
        self._last = time.monotonic()

    def diagnose_no_match(self, author_id: str, orcid_in_log: str) -> tuple[str, str]:
        try:
            if not orcid_in_log or orcid_in_log in {"None", "null"}:
                return "missing_orcid_in_log", "no_orcid"

            qid, qid_error = self.wdqs.find_qid_by_orcid(orcid_in_log)
            if qid:
                return "now_wdqs_has_unique_match", qid
            if qid_error == "qid_not_unique":
                return "wdqs_qid_not_unique", "multiple_qids"

            author = self.openalex.get_author(author_id)
            if not author.display_name:
                return "openalex_name_missing", "empty_display_name"
            candidates = self.wikidata.search_entities(author.display_name)
            if len(candidates) == 0:
                return "wdqs_no_match_name_no_candidate", author.display_name
            if len(candidates) == 1:
                return "wdqs_no_match_name_single_candidate", candidates[0].get("id", "")
            return "wdqs_no_match_name_ambiguous", f"candidates={len(candidates)}"
        except Exception as exc:
            return "diagnosis_error", str(exc)

    def diagnose_error(self, author_id: str) -> tuple[str, str]:
        try:
            author = self.openalex.get_author(author_id)
        except Exception as exc:
            return "openalex_get_author_failed", str(exc)

        if not author.orcid:
            return "missing_orcid", "openalex_author_without_orcid"

        try:
            qid, qid_error = self.wdqs.find_qid_by_orcid(author.orcid)
        except Exception as exc:
            return "wdqs_orcid_lookup_failed", str(exc)
        if qid_error:
            return f"wdqs_orcid_lookup_{qid_error}", author.orcid
        if not qid:
            return "wdqs_orcid_lookup_empty", author.orcid

        try:
            commons_file = self.wdqs.get_p18_image_by_qid(qid)
        except Exception as exc:
            return "wdqs_p18_lookup_failed", str(exc)
        if not commons_file:
            return "no_p18", qid

        try:
            candidate = self.commons.get_image_candidate(commons_file, self.thumb_width)
        except Exception as exc:
            return "commons_imageinfo_failed", str(exc)
        if not candidate:
            return "commons_imageinfo_missing", commons_file

        if candidate.mime not in self.allowed_mime:
            return "invalid_image_mime", candidate.mime

        try:
            self._wait()
            _ = self.commons.download_image(candidate.download_url)
        except Exception as exc:
            return "image_download_failed", str(exc)

        return "pre_upload_checks_passed", f"qid={qid} commons_file={commons_file}"


def write_csv(path: str, rows: list[FailureRow]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "author_id",
                "orcid",
                "status",
                "original_error_message",
                "diagnosed_reason",
                "diagnosed_detail",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.author_id,
                    r.orcid,
                    r.status,
                    r.original_error_message or "",
                    r.diagnosed_reason,
                    r.diagnosed_detail,
                ]
            )


def main() -> int:
    from avatar_pipeline.config import PipelineConfig, load_dotenv

    args = parse_args()
    setup_logging(args.log_level)
    load_dotenv(args.dotenv)
    config = PipelineConfig.from_env()
    config.global_qps_limit = args.qps

    entries, stats = parse_latest_run(args.log)
    if not entries:
        raise SystemExit(f"No run entries found in {args.log}")

    no_match_expected = stats.get("no_match")
    error_expected = stats.get("error")
    targets = [e for e in entries if e["status"] in {"no_match", "error"}]
    if args.limit > 0:
        targets = targets[: args.limit]

    logging.info(
        "latest_run_counts expected_no_match=%s expected_error=%s extracted_targets=%s",
        no_match_expected,
        error_expected,
        len(targets),
    )

    diagnoser = Diagnoser(config)
    out_rows: list[FailureRow] = []
    reason_counter: Counter[str] = Counter()

    for item in targets:
        if item["status"] == "no_match":
            reason, detail = diagnoser.diagnose_no_match(item["author_id"], item["orcid"])
        else:
            reason, detail = diagnoser.diagnose_error(item["author_id"])

        out_rows.append(
            FailureRow(
                author_id=item["author_id"],
                orcid=item["orcid"],
                status=item["status"],
                original_error_message=item["error_message"] or None,
                diagnosed_reason=reason,
                diagnosed_detail=detail,
            )
        )
        reason_counter[f"{item['status']}::{reason}"] += 1

    write_csv(args.out_csv, out_rows)
    logging.info("written_csv=%s rows=%s", args.out_csv, len(out_rows))
    for key, count in reason_counter.most_common():
        logging.info("reason_count %s=%s", key, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
