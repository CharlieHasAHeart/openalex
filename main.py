from __future__ import annotations

import argparse
import logging
from pathlib import Path
from datetime import date
from datetime import datetime

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAlex author avatar pipeline")
    parser.add_argument(
        "--author-id",
        action="append",
        default=[],
        help="Specific OpenAlex author id (URL or A-ID). Can repeat.",
    )
    parser.add_argument("--top-n", type=int, default=None, help="Top N works in the publication window.")
    parser.add_argument(
        "--top-offset",
        type=int,
        default=0,
        help="Skip first N works in the sorted window before taking top-n.",
    )
    parser.add_argument("--window-years", type=int, default=None, help="Recent Y years publication window.")
    parser.add_argument("--per-page", type=int, default=None, help="OpenAlex per-page size.")
    parser.add_argument("--window-start-date", default=None, help="Override window start date YYYY-MM-DD.")
    parser.add_argument("--window-end-date", default=None, help="Override window end date YYYY-MM-DD.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", default="logs", help="Directory for auto-generated log files.")
    parser.add_argument("--log-file", default=None, help="Optional explicit log file path.")
    return parser.parse_args()


def setup_logging(level: str, log_dir: str, log_file: str | None) -> Path:
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = Path(log_file) if log_file else Path(log_dir) / f"pipeline_{timestamp}.log"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(resolved_level)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(resolved_level)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_path, encoding="utf-8")
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return output_path


def _shift_years_day(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def main() -> int:
    args = parse_args()
    log_path = setup_logging(args.log_level, args.log_dir, args.log_file)
    logging.getLogger(__name__).info("log_file=%s", log_path)

    from avatar_pipeline.commons_client import CommonsClient
    from avatar_pipeline.config import PipelineConfig, load_dotenv
    from avatar_pipeline.http import HttpClient, RateLimiter
    from avatar_pipeline.openalex_client import OpenAlexClient
    from avatar_pipeline.oss_uploader import OssUploader
    from avatar_pipeline.pg_repository import PgRepository
    from avatar_pipeline.pipeline_runner import PipelineRunner
    from avatar_pipeline.wdqs_client import WdqsClient
    from avatar_pipeline.wikidata_api_client import WikidataApiClient

    load_dotenv(".env")
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

        if args.author_id:
            authors = [openalex_client.get_author(item) for item in args.author_id]
            for author in authors:
                runner.run_for_author(author)
        else:
            window_end_date = (
                date.fromisoformat(args.window_end_date) if args.window_end_date else date.today()
            )
            if args.window_start_date:
                window_start_date = date.fromisoformat(args.window_start_date)
            else:
                years = args.window_years if args.window_years is not None else config.works_window_years
                window_start_date = _shift_years_day(window_end_date, -years)
            top_n = args.top_n if args.top_n is not None else config.works_top_n
            per_page = args.per_page if args.per_page is not None else config.works_per_page

            runner.run_from_top_works(
                top_n=top_n,
                top_offset=max(args.top_offset, 0),
                window_start_date=window_start_date,
                window_end_date=window_end_date,
                per_page=per_page,
            )

        logging.getLogger(__name__).info("stats=%s", dict(runner.stats))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
