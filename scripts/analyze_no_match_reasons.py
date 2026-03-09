#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import Counter
from dataclasses import dataclass

import requests


NO_MATCH_PATTERN = re.compile(
    r"author_id=(?P<author_id>\S+)\s+orcid=(?P<orcid>\S+)\s+status=no_match\b"
)


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def env(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    return value


@dataclass(slots=True)
class NoMatchEntry:
    author_id: str
    orcid: str
    reason: str | None = None
    detail: str | None = None


class Analyzer:
    def __init__(self, wdqs_endpoint: str, wikidata_api_url: str, timeout: int, qps: float) -> None:
        self.wdqs_endpoint = wdqs_endpoint
        self.wikidata_api_url = wikidata_api_url
        self.timeout = timeout
        self.interval = 1.0 / qps if qps > 0 else 0
        self.last_ts = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "openalex-avatar-pipeline/1.0 (analysis-no-match-reasons)"}
        )

    def _wait(self) -> None:
        if self.interval <= 0:
            return
        now = time.monotonic()
        delta = now - self.last_ts
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self.last_ts = time.monotonic()

    def _wdqs_find_qid_by_orcid(self, orcid: str) -> tuple[str | None, str | None]:
        query = f'SELECT ?item WHERE {{ ?item wdt:P496 "{orcid}" . }} LIMIT 5'
        headers = {"Accept": "application/sparql-results+json"}
        self._wait()
        resp = self.session.get(
            self.wdqs_endpoint, params={"query": query}, headers=headers, timeout=self.timeout
        )
        resp.raise_for_status()
        bindings = (resp.json().get("results") or {}).get("bindings") or []
        if len(bindings) == 0:
            return None, "wdqs_no_match"
        if len(bindings) > 1:
            return None, "wdqs_qid_not_unique"
        qid = bindings[0]["item"]["value"].rsplit("/", 1)[-1]
        return qid, None

    def _wikidata_search_name(self, name: str) -> list[dict]:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "type": "item",
            "search": name,
            "limit": "5",
        }
        self._wait()
        resp = self.session.get(self.wikidata_api_url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json().get("search") or []

    def fetch_openalex_name(self, author_id: str) -> tuple[str | None, str | None]:
        url = author_id if author_id.startswith("http") else f"https://api.openalex.org/authors/{author_id}"
        self._wait()
        resp = self.session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("display_name"), payload.get("orcid")

    def diagnose(self, entry: NoMatchEntry) -> NoMatchEntry:
        try:
            if entry.orcid in {"None", "null", ""}:
                entry.reason = "missing_orcid_in_log"
                entry.detail = "no_orcid"
                return entry

            qid, wdqs_reason = self._wdqs_find_qid_by_orcid(entry.orcid)
            if qid:
                entry.reason = "now_wdqs_has_match"
                entry.detail = qid
                return entry
            if wdqs_reason == "wdqs_qid_not_unique":
                entry.reason = "wdqs_qid_not_unique"
                entry.detail = "multiple_qid_for_orcid"
                return entry

            display_name, orcid_from_openalex = self.fetch_openalex_name(entry.author_id)
            if not display_name:
                entry.reason = "openalex_author_fetch_failed_name_missing"
                entry.detail = "no_display_name"
                return entry

            if orcid_from_openalex and orcid_from_openalex.strip("/").split("/")[-1] != entry.orcid:
                entry.reason = "orcid_changed_in_openalex"
                entry.detail = orcid_from_openalex
                return entry

            candidates = self._wikidata_search_name(display_name)
            if len(candidates) == 0:
                entry.reason = "wdqs_no_match_and_name_no_match"
                entry.detail = display_name
            elif len(candidates) == 1:
                entry.reason = "wdqs_no_match_name_single_candidate"
                entry.detail = candidates[0].get("id")
            else:
                entry.reason = "wdqs_no_match_name_ambiguous"
                entry.detail = f"candidates={len(candidates)}"
            return entry
        except Exception as exc:
            entry.reason = "diagnosis_error"
            entry.detail = str(exc)
            return entry


def extract_no_match_entries(log_path: str) -> list[NoMatchEntry]:
    entries: dict[str, NoMatchEntry] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = NO_MATCH_PATTERN.search(line)
            if not m:
                continue
            author_id = m.group("author_id")
            orcid = m.group("orcid")
            if author_id not in entries:
                entries[author_id] = NoMatchEntry(author_id=author_id, orcid=orcid)
    return list(entries.values())


def write_csv(path: str, entries: list[NoMatchEntry]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["author_id", "orcid", "reason", "detail"])
        for item in entries:
            writer.writerow([item.author_id, item.orcid, item.reason or "", item.detail or ""])


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze no_match reasons from pipeline log")
    parser.add_argument("--log", default="logs/systemd_pipeline.log", help="Path to pipeline log file")
    parser.add_argument("--dotenv", default=".env", help="Path to .env")
    parser.add_argument(
        "--out-csv", default="logs/no_match_reason_details.csv", help="Output CSV for per-author reasons"
    )
    parser.add_argument("--limit", type=int, default=0, help="Analyze first N unique authors (0=all)")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--qps", type=float, default=2.0)
    args = parser.parse_args()

    load_dotenv(args.dotenv)
    wdqs = env("WDQS_ENDPOINT", "https://query.wikidata.org/sparql")
    wikidata_api = env("WIKIDATA_API_URL", "https://www.wikidata.org/w/api.php")

    entries = extract_no_match_entries(args.log)
    if args.limit > 0:
        entries = entries[: args.limit]

    analyzer = Analyzer(
        wdqs_endpoint=wdqs,
        wikidata_api_url=wikidata_api,
        timeout=args.timeout,
        qps=args.qps,
    )

    results: list[NoMatchEntry] = []
    counter: Counter[str] = Counter()
    for item in entries:
        diagnosed = analyzer.diagnose(item)
        results.append(diagnosed)
        counter[diagnosed.reason or "unknown"] += 1

    write_csv(args.out_csv, results)

    print(f"total_no_match_unique_authors={len(results)}")
    print("reason_counts:")
    for reason, count in counter.most_common():
        print(f"- {reason}: {count}")
    print(f"details_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
