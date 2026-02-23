#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reprocess authors with invalid_image_too_small / invalid_image_too_small_bytes"
    )
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--limit", type=int, default=0, help="0 means all")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--source",
        choices=["csv", "log", "db"],
        default="csv",
        help="Source for author ids. csv is recommended for precise too_small reasons.",
    )
    parser.add_argument("--log-path", default="logs/systemd_pipeline.log")
    parser.add_argument("--csv-path", default="logs/status_reason_details_full.csv")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


LOG_INVALID_IMAGE_PATTERN = re.compile(r"author_id=(?P<author_id>\S+)\s+orcid=\S+\s+status=invalid_image\b")


def _unique_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def extract_author_ids_from_log(log_path: str) -> list[str]:
    ids: list[str] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = LOG_INVALID_IMAGE_PATTERN.search(line)
            if not m:
                continue
            ids.append(m.group("author_id"))
    return _unique_keep_order(ids)


def extract_author_ids_from_csv(csv_path: str) -> list[str]:
    reasons = {"invalid_image_too_small", "invalid_image_too_small_bytes"}
    ids: list[str] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip()
            reason = (row.get("reason") or "").strip()
            error_message = (row.get("error_message") or "").strip()
            if status and status != "invalid_image":
                continue
            if reason in reasons or error_message in reasons:
                aid = (row.get("author_id") or "").strip()
                if aid:
                    ids.append(aid)
    return _unique_keep_order(ids)


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    from avatar_pipeline.commons_client import CommonsClient
    from avatar_pipeline.config import PipelineConfig, load_dotenv
    from avatar_pipeline.http import HttpClient, RateLimiter
    from avatar_pipeline.openalex_client import OpenAlexClient
    from avatar_pipeline.oss_uploader import OssUploader
    from avatar_pipeline.pg_repository import PgRepository
    from avatar_pipeline.pipeline_runner import PipelineRunner
    from avatar_pipeline.wdqs_client import WdqsClient
    from avatar_pipeline.wikidata_api_client import WikidataApiClient

    load_dotenv(args.dotenv)
    config = PipelineConfig.from_env()

    limiter = RateLimiter(config.global_qps_limit)
    http = HttpClient(
        timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        rate_limiter=limiter,
    )

    openalex_client = OpenAlexClient(config.openalex_base_url, http, config.openalex_mailto)
    wdqs_client = WdqsClient(config.wdqs_endpoint, http)
    wikidata_api_client = WikidataApiClient(config.wikidata_api_url, http)
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

    with PgRepository(
        host=config.pghost,
        port=config.pgport,
        database=config.pgdatabase,
        user=config.pguser,
        password=config.pgpassword,
        sslmode=config.pgsslmode,
    ) as repository:
        runner = PipelineRunner(
            config=config,
            openalex_client=openalex_client,
            wdqs_client=wdqs_client,
            wikidata_api_client=wikidata_api_client,
            commons_client=commons_client,
            oss_uploader=oss_uploader,
            pg_repository=repository,
        )

        if args.source == "csv":
            author_ids = extract_author_ids_from_csv(args.csv_path)
        elif args.source == "log":
            author_ids = extract_author_ids_from_log(args.log_path)
        else:
            limit = args.limit if args.limit > 0 else None
            author_ids = repository.list_invalid_image_too_small_author_ids(limit=limit)

        if args.limit > 0:
            author_ids = author_ids[: args.limit]
        logging.info("found_authors_to_recover=%s", len(author_ids))

        for author_id in author_ids:
            try:
                author = openalex_client.get_author(author_id)
            except Exception as exc:
                logging.exception("fetch_author_failed author_id=%s error=%s", author_id, exc)
                continue
            runner.run_for_author(author)

        logging.info("recovery_stats=%s", dict(runner.stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
