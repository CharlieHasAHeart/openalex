#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import unquote, urlparse

import requests


LINE_PATTERN = re.compile(
    r"author_id=(?P<author_id>\S+)\s+orcid=(?P<orcid>\S+)\s+status=(?P<status>\S+)\s+qid=(?P<qid>\S+)\s+commons_file=(?P<commons_file>.*)$"
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
    return os.getenv(name, default).strip()


@dataclass(slots=True)
class StatusEntry:
    author_id: str
    orcid: str
    status: str
    qid: str | None
    commons_file: str | None
    reason: str | None = None
    detail: str | None = None


class Analyzer:
    def __init__(
        self,
        wdqs_endpoint: str,
        wikidata_api_url: str,
        commons_api_url: str,
        timeout: int,
        qps: float,
        thumb_width: int,
        min_edge_px: int,
        allowed_mime: set[str],
    ) -> None:
        self.wdqs_endpoint = wdqs_endpoint
        self.wikidata_api_url = wikidata_api_url
        self.commons_api_url = commons_api_url
        self.timeout = timeout
        self.interval = 1.0 / qps if qps > 0 else 0.0
        self.last_ts = 0.0
        self.thumb_width = thumb_width
        self.min_edge_px = min_edge_px
        self.allowed_mime = allowed_mime
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "openalex-avatar-pipeline/1.0 (analyze-ambiguous-invalid-image)"}
        )

    def _wait(self) -> None:
        if self.interval <= 0:
            return
        now = time.monotonic()
        delta = now - self.last_ts
        if delta < self.interval:
            time.sleep(self.interval - delta)
        self.last_ts = time.monotonic()

    def _get(self, url: str, **kwargs) -> requests.Response:
        self._wait()
        resp = self.session.get(url, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp

    def _wdqs_find_by_orcid(self, orcid: str) -> list[str]:
        q = f'SELECT ?item WHERE {{ ?item wdt:P496 "{orcid}" . }} LIMIT 10'
        headers = {"Accept": "application/sparql-results+json"}
        data = self._get(self.wdqs_endpoint, params={"query": q}, headers=headers).json()
        bindings = (data.get("results") or {}).get("bindings") or []
        return [item["item"]["value"].rsplit("/", 1)[-1] for item in bindings]

    def _openalex_author_name(self, author_id: str) -> str | None:
        url = author_id if author_id.startswith("http") else f"https://api.openalex.org/authors/{author_id}"
        payload = self._get(url).json()
        return payload.get("display_name")

    def _wikidata_search_count(self, display_name: str) -> int:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "type": "item",
            "search": display_name,
            "limit": "5",
        }
        data = self._get(self.wikidata_api_url, params=params).json()
        return len(data.get("search") or [])

    def _wdqs_p18(self, qid: str) -> str | None:
        q = f"SELECT ?file WHERE {{ wd:{qid} wdt:P18 ?file . }} LIMIT 1"
        headers = {"Accept": "application/sparql-results+json"}
        data = self._get(self.wdqs_endpoint, params={"query": q}, headers=headers).json()
        bindings = (data.get("results") or {}).get("bindings") or []
        if not bindings:
            return None
        file_uri = bindings[0]["file"]["value"]
        path = urlparse(file_uri).path
        return unquote(path.rsplit("/", 1)[-1]) if path else None

    def _commons_imageinfo(self, commons_file: str) -> dict | None:
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": f"File:{commons_file}",
            "iiprop": "url|size|mime",
            "iiurlwidth": str(self.thumb_width),
        }
        payload = self._get(self.commons_api_url, params=params).json()
        pages = (payload.get("query") or {}).get("pages") or {}
        page = next(iter(pages.values()), None)
        if not page:
            return None
        info = (page.get("imageinfo") or [None])[0]
        return info

    def diagnose_ambiguous(self, item: StatusEntry) -> StatusEntry:
        try:
            if item.orcid in {"None", "null", ""}:
                item.reason = "missing_orcid"
                item.detail = "orcid_missing"
                return item
            qids = self._wdqs_find_by_orcid(item.orcid)
            if len(qids) > 1:
                item.reason = "wdqs_qid_not_unique"
                item.detail = ",".join(qids[:5])
                return item
            if len(qids) == 1:
                item.reason = "now_unique_match"
                item.detail = qids[0]
                return item
            name = self._openalex_author_name(item.author_id)
            if not name:
                item.reason = "wdqs_no_match_openalex_name_missing"
                item.detail = "no_name"
                return item
            c = self._wikidata_search_count(name)
            if c > 1:
                item.reason = "name_search_ambiguous"
                item.detail = f"candidates={c}"
            elif c == 1:
                item.reason = "wdqs_no_match_name_single_candidate"
                item.detail = "candidate=1"
            else:
                item.reason = "wdqs_no_match_name_no_candidate"
                item.detail = "candidate=0"
            return item
        except Exception as exc:
            item.reason = "diagnosis_error"
            item.detail = str(exc)
            return item

    def diagnose_invalid_image(self, item: StatusEntry) -> StatusEntry:
        try:
            qid = None if not item.qid or item.qid == "None" else item.qid
            if not qid:
                if item.orcid in {"None", "null", ""}:
                    item.reason = "missing_orcid"
                    item.detail = "no_qid_no_orcid"
                    return item
                qids = self._wdqs_find_by_orcid(item.orcid)
                if len(qids) != 1:
                    item.reason = "qid_not_resolved_now"
                    item.detail = f"qid_count={len(qids)}"
                    return item
                qid = qids[0]

            commons_file = item.commons_file if item.commons_file and item.commons_file != "None" else None
            if not commons_file:
                commons_file = self._wdqs_p18(qid)
            if not commons_file:
                item.reason = "p18_not_found_now"
                item.detail = f"qid={qid}"
                return item

            info = self._commons_imageinfo(commons_file)
            if not info:
                item.reason = "commons_imageinfo_missing"
                item.detail = commons_file
                return item

            url = info.get("thumburl") or info.get("url")
            if not url:
                item.reason = "commons_url_missing"
                item.detail = commons_file
                return item

            mime = (info.get("mime") or "").strip()
            width = int(info.get("thumbwidth") or info.get("width") or 0)
            height = int(info.get("thumbheight") or info.get("height") or 0)
            content = self._get(url).content

            if mime not in self.allowed_mime:
                item.reason = "invalid_image_mime"
                item.detail = mime
                return item
            if width < self.min_edge_px and height < self.min_edge_px:
                item.reason = "invalid_image_too_small"
                item.detail = f"{width}x{height}"
                return item
            if len(content) <= 2048:
                item.reason = "invalid_image_too_small_bytes"
                item.detail = f"bytes={len(content)}"
                return item

            item.reason = "now_valid_image"
            item.detail = f"{mime} {width}x{height} bytes={len(content)}"
            return item
        except Exception as exc:
            item.reason = "diagnosis_error"
            item.detail = str(exc)
            return item


def extract_entries(log_path: str, statuses: set[str]) -> list[StatusEntry]:
    uniq: dict[tuple[str, str], StatusEntry] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = LINE_PATTERN.search(line.rstrip("\n"))
            if not m:
                continue
            status = m.group("status")
            if status not in statuses:
                continue
            author_id = m.group("author_id")
            key = (author_id, status)
            if key in uniq:
                continue
            qid = m.group("qid")
            commons_file = m.group("commons_file").strip()
            uniq[key] = StatusEntry(
                author_id=author_id,
                orcid=m.group("orcid"),
                status=status,
                qid=None if qid == "None" else qid,
                commons_file=None if commons_file == "None" else commons_file,
            )
    return list(uniq.values())


def write_csv(path: str, rows: Iterable[StatusEntry]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["status", "author_id", "orcid", "qid", "commons_file", "reason", "detail"])
        for r in rows:
            w.writerow(
                [
                    r.status,
                    r.author_id,
                    r.orcid,
                    r.qid or "",
                    r.commons_file or "",
                    r.reason or "",
                    r.detail or "",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze ambiguous/invalid_image reasons from pipeline log")
    parser.add_argument("--log", default="logs/systemd_pipeline.log")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--out-csv", default="logs/status_reason_details.csv")
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--qps", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    args = parser.parse_args()

    load_dotenv(args.dotenv)
    wdqs = env("WDQS_ENDPOINT", "https://query.wikidata.org/sparql")
    wikidata_api = env("WIKIDATA_API_URL", "https://www.wikidata.org/w/api.php")
    commons_api = env("COMMONS_API_URL", "https://commons.wikimedia.org/w/api.php")
    thumb_width = int(env("AVATAR_THUMB_WIDTH", "512"))
    min_edge_px = int(env("MIN_IMAGE_EDGE_PX", "200"))
    allowed_mime = {m.strip() for m in env("ALLOWED_MIME", "image/jpeg,image/png,image/webp").split(",")}

    rows = extract_entries(args.log, {"ambiguous", "invalid_image"})
    if args.limit > 0:
        rows = rows[: args.limit]

    analyzer = Analyzer(
        wdqs_endpoint=wdqs,
        wikidata_api_url=wikidata_api,
        commons_api_url=commons_api,
        timeout=args.timeout,
        qps=args.qps,
        thumb_width=thumb_width,
        min_edge_px=min_edge_px,
        allowed_mime=allowed_mime,
    )

    out: list[StatusEntry] = []
    by_status_reason: Counter[str] = Counter()
    for row in rows:
        if row.status == "ambiguous":
            row = analyzer.diagnose_ambiguous(row)
        elif row.status == "invalid_image":
            row = analyzer.diagnose_invalid_image(row)
        out.append(row)
        by_status_reason[f"{row.status}:{row.reason}"] += 1

    write_csv(args.out_csv, out)

    print(f"total_records={len(out)}")
    print("reason_counts:")
    for k, v in by_status_reason.most_common():
        print(f"- {k}: {v}")
    print(f"details_csv={args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
