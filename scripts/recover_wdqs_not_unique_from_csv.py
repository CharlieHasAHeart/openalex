#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover ambiguous wdqs_qid_not_unique entries from CSV and write avatar if image exists"
    )
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--csv-path", default="logs/status_reason_details_full.csv")
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--min-score", type=int, default=40)
    parser.add_argument("--min-gap", type=int, default=15)
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _normalize_text(text: str) -> str:
    value = text.lower().strip()
    value = re.sub(r"[^a-z0-9\s]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


def _token_overlap_score(a: str, b: str) -> int:
    ta = set(_normalize_text(a).split())
    tb = set(_normalize_text(b).split())
    if not ta or not tb:
        return 0
    inter = len(ta & tb)
    union = len(ta | tb)
    return int((inter / union) * 40)


def extract_targets_from_csv(csv_path: str, limit: int) -> list[tuple[str, str, list[str]]]:
    out: list[tuple[str, str, list[str]]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reason = (row.get("reason") or "").strip()
            if reason != "wdqs_qid_not_unique":
                continue
            author_id = (row.get("author_id") or "").strip()
            orcid = (row.get("orcid") or "").strip()
            detail = (row.get("detail") or "").strip()
            if not author_id or not detail:
                continue
            qids = [item.strip() for item in detail.split(",") if item.strip()]
            if not qids:
                continue
            out.append((author_id, orcid, qids))
            if limit > 0 and len(out) >= limit:
                break
    return out


def fetch_qid_profile(http, wikidata_api_url: str, qid: str) -> dict[str, Any]:
    params = {
        "action": "wbgetentities",
        "format": "json",
        "ids": qid,
        "languages": "en",
        "props": "labels|descriptions|aliases|claims",
    }
    data = http.request("GET", wikidata_api_url, params=params).json()
    entity = (data.get("entities") or {}).get(qid) or {}
    labels = entity.get("labels") or {}
    descriptions = entity.get("descriptions") or {}
    aliases = entity.get("aliases") or {}
    claims = entity.get("claims") or {}
    alias_values = [item.get("value", "") for item in aliases.get("en") or [] if item.get("value")]
    return {
        "qid": qid,
        "label": (labels.get("en") or {}).get("value", ""),
        "description": (descriptions.get("en") or {}).get("value", ""),
        "aliases": alias_values,
        "has_p18": "P18" in claims and bool(claims.get("P18")),
    }


def score_profile(display_name: str, profile: dict[str, Any]) -> int:
    score = 0
    label = profile.get("label") or ""
    aliases = profile.get("aliases") or []
    has_p18 = bool(profile.get("has_p18"))

    if label:
        if label.strip().lower() == display_name.strip().lower():
            score += 100
        elif _normalize_text(label) == _normalize_text(display_name):
            score += 80
        score += _token_overlap_score(display_name, label)
    for alias in aliases:
        if alias.strip().lower() == display_name.strip().lower():
            score = max(score, 90)
        elif _normalize_text(alias) == _normalize_text(display_name):
            score = max(score, 75)
        else:
            score = max(score, _token_overlap_score(display_name, alias))
    if has_p18:
        score += 10
    return score


def choose_qid(
    display_name: str,
    qids: list[str],
    http,
    wikidata_api_url: str,
    min_score: int,
    min_gap: int,
) -> tuple[str | None, str]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for qid in qids:
        profile = fetch_qid_profile(http, wikidata_api_url, qid)
        score = score_profile(display_name, profile)
        scored.append((score, profile))

    scored.sort(key=lambda item: item[0], reverse=True)
    if not scored:
        return None, "no_candidates"
    top_score, top_profile = scored[0]
    if top_score < min_score:
        return None, f"score_too_low:{top_score}"
    if len(scored) >= 2 and (top_score - scored[1][0]) < min_gap:
        return None, f"score_gap_too_small:{top_score}-{scored[1][0]}"
    return top_profile["qid"], f"selected_score:{top_score}"


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    from avatar_pipeline.avatar_gate import validate_image_candidate
    from avatar_pipeline.commons_client import CommonsClient
    from avatar_pipeline.config import PipelineConfig, load_dotenv
    from avatar_pipeline.http import HttpClient, RateLimiter
    from avatar_pipeline.models import PipelineResult
    from avatar_pipeline.openalex_client import OpenAlexClient
    from avatar_pipeline.oss_uploader import OssUploader, sha256_hex
    from avatar_pipeline.pg_repository import PgRepository
    from avatar_pipeline.wdqs_client import WdqsClient

    load_dotenv(args.dotenv)
    config = PipelineConfig.from_env()

    targets = extract_targets_from_csv(args.csv_path, args.limit)
    logging.info("targets_from_csv=%s", len(targets))
    if not targets:
        return 0

    limiter = RateLimiter(config.global_qps_limit)
    http = HttpClient(
        timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        rate_limiter=limiter,
    )

    openalex_client = OpenAlexClient(config.openalex_base_url, http, config.openalex_mailto)
    wdqs_client = WdqsClient(config.wdqs_endpoint, http)
    commons_client = CommonsClient(config.commons_api_url, http)
    oss_uploader = OssUploader(
        access_key_id=config.aliyun_oss_access_key_id,
        access_key_secret=config.aliyun_oss_access_key_secret,
        bucket_name=config.aliyun_oss_bucket,
        endpoint=config.aliyun_oss_endpoint,
        public_base_url=config.aliyun_oss_public_base_url,
        key_prefix=config.aliyun_oss_key_prefix,
        cache_control=config.aliyun_oss_cache_control,
    )

    counters = {"processed": 0, "selected": 0, "written_ok": 0, "skipped": 0, "error": 0}

    with PgRepository(
        host=config.pghost,
        port=config.pgport,
        database=config.pgdatabase,
        user=config.pguser,
        password=config.pgpassword,
        sslmode=config.pgsslmode,
    ) as repository:
        for author_id, _orcid, qids in targets:
            counters["processed"] += 1
            try:
                author = openalex_client.get_author(author_id)
                selected_qid, reason = choose_qid(
                    author.display_name,
                    qids,
                    http,
                    config.wikidata_api_url,
                    args.min_score,
                    args.min_gap,
                )
                if not selected_qid:
                    counters["skipped"] += 1
                    logging.info("skip author_id=%s reason=%s", author_id, reason)
                    continue
                counters["selected"] += 1

                commons_file = wdqs_client.get_p18_image_by_qid(selected_qid)
                if not commons_file:
                    counters["skipped"] += 1
                    logging.info(
                        "skip author_id=%s qid=%s reason=no_p18_after_select", author_id, selected_qid
                    )
                    continue

                candidate = commons_client.get_image_candidate(commons_file, config.avatar_thumb_width)
                if not candidate:
                    counters["skipped"] += 1
                    logging.info("skip author_id=%s qid=%s reason=commons_imageinfo_missing", author_id, selected_qid)
                    continue

                image_bytes = commons_client.download_image(candidate.download_url)
                valid, image_error = validate_image_candidate(
                    candidate, image_bytes, config.allowed_mime, config.min_image_edge_px
                )
                if not valid:
                    counters["skipped"] += 1
                    logging.info(
                        "skip author_id=%s qid=%s reason=%s",
                        author_id,
                        selected_qid,
                        image_error,
                    )
                    continue

                sha256 = sha256_hex(image_bytes)
                existing = repository.get_existing_by_author_id(author.author_id)
                if existing and existing.get("content_sha256") == sha256 and existing.get("oss_object_key"):
                    result = PipelineResult(
                        author_id=author.author_id,
                        status="ok",
                        wikidata_qid=selected_qid,
                        commons_file=commons_file,
                        content_sha256=sha256,
                        oss_object_key=existing["oss_object_key"],
                        oss_url=existing.get("oss_url"),
                        error_message=f"csv_qid_disambiguated:{reason}",
                    )
                    repository.upsert_result(result)
                    counters["written_ok"] += 1
                    continue

                object_key = oss_uploader.build_object_key(author.author_id, sha256, candidate.mime)
                oss_url = oss_uploader.upload(object_key, image_bytes, candidate.mime)
                result = PipelineResult(
                    author_id=author.author_id,
                    status="ok",
                    wikidata_qid=selected_qid,
                    commons_file=commons_file,
                    content_sha256=sha256,
                    oss_object_key=object_key,
                    oss_url=oss_url,
                    error_message=f"csv_qid_disambiguated:{reason}",
                )
                repository.upsert_result(result)
                counters["written_ok"] += 1
                logging.info(
                    "written_ok author_id=%s qid=%s commons_file=%s", author.author_id, selected_qid, commons_file
                )
            except Exception as exc:
                counters["error"] += 1
                logging.exception("recover_failed author_id=%s error=%s", author_id, exc)

    logging.info("summary=%s", counters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
