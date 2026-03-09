from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import threading
import time
from collections import Counter
from urllib.parse import urlparse

import psycopg

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAlex author avatar pipeline")
    parser.add_argument(
        "--author-id",
        action="append",
        default=[],
        help="Specific OpenAlex author id (URL or A-ID). Can repeat.",
    )
    parser.add_argument("--author-limit", type=int, default=0, help="Limit authors loaded from DB authors_analysis table. 0 means all.")
    parser.add_argument("--author-offset", type=int, default=0, help="Skip first N authors from DB authors_analysis table.")
    parser.add_argument("--fetch-batch-size", type=int, default=2000, help="Batch size for loading authors from DB in streaming mode.")
    parser.add_argument("--workers", type=int, default=1, help="Worker threads for concurrent processing. 1 means serial.")
    parser.add_argument("--progress-every", type=int, default=50, help="Emit progress log every N authors. 0 to disable intermediate progress logs.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--export-review-summaries", action="store_true", help="Export recent review summaries and evaluation stats, then exit.")
    parser.add_argument("--review-summary-limit", type=int, default=200, help="How many recent decision summaries to export.")
    parser.add_argument("--review-recommendation", default="", help="Optional filter for review recommendation (auto_accept/needs_review/ambiguous).")
    parser.add_argument("--sample-review", action="store_true", help="Also output stratified review samples by recommendation.")
    parser.add_argument("--sample-per-group", type=int, default=20, help="Sample size per recommendation group.")
    parser.add_argument("--benchmark-authors-file", default="", help="Path to benchmark authors JSON/JSONL file.")
    parser.add_argument("--annotations-file", default="", help="Path to annotations JSON/JSONL file.")
    parser.add_argument("--export-benchmark-report", action="store_true", help="Export benchmark report with annotations join, then exit.")
    parser.add_argument("--benchmark-result-limit", type=int, default=1000, help="How many recent decision summaries to load for benchmark report.")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _build_author_id_candidates(raw_ids: set[str]) -> list[str]:
    candidates: set[str] = set()
    for value in raw_ids:
        cleaned = value.strip().rstrip("/")
        if not cleaned:
            continue
        candidates.add(cleaned)
        if cleaned.startswith("http://") or cleaned.startswith("https://"):
            tail = urlparse(cleaned).path.rsplit("/", 1)[-1].strip()
            if tail:
                candidates.add(tail)
    return sorted(candidates)


def _create_runner(config, limiter, run_id: str | None = None):
    from avatar_pipeline.http import HttpClient
    from avatar_pipeline.llm_matcher import LlmMatcher
    from avatar_pipeline.oss_uploader import OssUploader
    from avatar_pipeline.pg_repository import PgRepository
    from avatar_pipeline.pipeline_runner import PipelineRunner
    from avatar_pipeline.web_search_client import WebSearchClient

    http = HttpClient(
        timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        rate_limiter=limiter,
        retry_base_seconds=config.retry_base_seconds,
        retry_max_seconds=config.retry_max_seconds,
        retry_jitter_ratio=config.retry_jitter_ratio,
        retry_429_min_delay_seconds=config.retry_429_min_delay_seconds,
    )
    web_search_client = WebSearchClient(http, max_results=config.websearch_max_results)
    llm_matcher = LlmMatcher(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        model=config.llm_model,
        timeout_seconds=config.llm_timeout_seconds,
    )
    oss_uploader = OssUploader(
        access_key_id=config.aliyun_oss_access_key_id,
        access_key_secret=config.aliyun_oss_access_key_secret,
        bucket_name=config.aliyun_oss_bucket,
        endpoint=config.aliyun_oss_endpoint,
        public_base_url=config.aliyun_oss_public_base_url,
        key_prefix=config.aliyun_oss_key_prefix,
        cache_control=config.aliyun_oss_cache_control,
    )
    repository = PgRepository(
        host=config.pghost,
        port=config.pgport,
        database=config.pgdatabase,
        user=config.pguser,
        password=config.pgpassword,
        sslmode=config.pgsslmode,
    )
    runner = PipelineRunner(
        config=config,
        web_search_client=web_search_client,
        llm_matcher=llm_matcher,
        oss_uploader=oss_uploader,
        pg_repository=repository,
        run_id=run_id,
    )
    return runner, repository


def _maybe_log_progress(
    logger: logging.Logger,
    done: int,
    total: int | None,
    start_monotonic: float,
    last_status: str,
    progress_every: int,
) -> None:
    if done <= 0:
        return
    if progress_every <= 0:
        return
    if done % progress_every != 0 and (total is None or done != total):
        return
    elapsed = max(time.monotonic() - start_monotonic, 1e-9)
    rate = done / elapsed
    if total is None:
        logger.info(
            "progress done=%s total=unknown rate=%.2f/s last_status=%s",
            done,
            rate,
            last_status,
        )
        return
    logger.info(
        "progress done=%s total=%s pct=%.1f rate=%.2f/s last_status=%s",
        done,
        total,
        (done / total * 100.0),
        rate,
        last_status,
    )


def _run_parallel(
    config,
    limiter,
    run_id: str | None,
    authors,
    workers: int,
    progress_every: int,
    logger: logging.Logger,
    done_start: int,
    total: int | None,
    start_monotonic: float,
) -> tuple[Counter[str], int]:
    stats: Counter[str] = Counter()
    local = threading.local()
    created_repositories = []
    created_lock = threading.Lock()
    done = done_start

    def _get_runner():
        runner = getattr(local, "runner", None)
        if runner is not None:
            return runner
        runner, repository = _create_runner(config, limiter, run_id=run_id)
        local.runner = runner
        local.repository = repository
        with created_lock:
            created_repositories.append(repository)
        return runner

    def _task(item) -> str:
        runner = _get_runner()
        return runner.run_for_author_seed(item)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for stat_key in executor.map(_task, authors):
                stats[stat_key] += 1
                done += 1
                _maybe_log_progress(logger, done, total, start_monotonic, stat_key, progress_every)
    finally:
        for repository in created_repositories:
            repository.close()
    return stats, done


def _run_serial(
    runner,
    authors,
    progress_every: int,
    logger: logging.Logger,
    done_start: int,
    total: int | None,
    start_monotonic: float,
) -> tuple[Counter[str], int]:
    stats: Counter[str] = Counter()
    done = done_start
    for author in authors:
        stat_key = runner.run_for_author_seed(author)
        stats[stat_key] += 1
        done += 1
        _maybe_log_progress(logger, done, total, start_monotonic, stat_key, progress_every)
    return stats, done

def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    from avatar_pipeline.config import PipelineConfig, load_dotenv
    from avatar_pipeline.benchmark import (
        build_benchmark_report,
        build_benchmark_rows,
        load_annotations,
        load_benchmark_authors,
    )
    from avatar_pipeline.evaluation import format_review_export, sample_decisions, summarize_decisions
    from avatar_pipeline.http import RateLimiter
    from avatar_pipeline.pg_repository import PgRepository

    load_dotenv(".env")
    config = PipelineConfig.from_env()

    limiter = RateLimiter(config.global_qps_limit)
    workers = max(args.workers, 1)
    fetch_batch_size = max(args.fetch_batch_size, 1)
    progress_every = max(args.progress_every, 0)
    batch_load_max_retries = 5
    total_target = args.author_limit if args.author_limit > 0 else None
    start_offset = max(args.author_offset, 0)
    start_monotonic = time.monotonic()
    stats: Counter[str] = Counter()
    done = 0

    runner = None
    runner_repository = None
    run_id: str | None = None
    if workers == 1:
        runner, runner_repository = _create_runner(config, limiter, run_id=run_id)

    if args.export_review_summaries:
        with PgRepository(
            host=config.pghost,
            port=config.pgport,
            database=config.pgdatabase,
            user=config.pguser,
            password=config.pgpassword,
            sslmode=config.pgsslmode,
        ) as repository:
            rows = repository.list_recent_decision_summaries(
                limit=max(1, args.review_summary_limit),
                recommendation=args.review_recommendation.strip() or None,
            )
        formatted = format_review_export(rows)
        summary = summarize_decisions(formatted)
        print(json.dumps({"summary": summary}, ensure_ascii=False, indent=2))
        print(json.dumps({"rows": formatted}, ensure_ascii=False, indent=2))
        if args.sample_review:
            sampled = sample_decisions(formatted, per_group=max(1, args.sample_per_group))
            print(json.dumps({"sampled": sampled}, ensure_ascii=False, indent=2))
        return 0

    if args.export_benchmark_report:
        benchmark_authors = []
        annotations: dict[str, dict] = {}
        try:
            if args.benchmark_authors_file.strip():
                benchmark_authors = load_benchmark_authors(args.benchmark_authors_file.strip())
            if args.annotations_file.strip():
                annotations = load_annotations(args.annotations_file.strip())
        except Exception as exc:
            logger.exception("benchmark_input_load_failed error=%s", exc)
            return 2

        author_ids = [str(item.get("author_id")) for item in benchmark_authors if item.get("author_id") is not None]
        with PgRepository(
            host=config.pghost,
            port=config.pgport,
            database=config.pgdatabase,
            user=config.pguser,
            password=config.pgpassword,
            sslmode=config.pgsslmode,
        ) as repository:
            rows = repository.list_recent_decision_summaries(
                limit=max(1, args.benchmark_result_limit),
                author_ids=author_ids or None,
            )
        formatted = format_review_export(rows)
        joined = build_benchmark_rows(
            result_rows=formatted,
            annotations=annotations,
            benchmark_authors=benchmark_authors or None,
        )
        report = build_benchmark_report(
            joined_rows=joined,
            benchmark_meta={
                "authors_file": args.benchmark_authors_file.strip() or None,
                "annotations_file": args.annotations_file.strip() or None,
                "benchmark_author_count": len(benchmark_authors),
                "joined_rows": len(joined),
                "result_limit": max(1, args.benchmark_result_limit),
            },
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    try:
        with PgRepository(
            host=config.pghost,
            port=config.pgport,
            database=config.pgdatabase,
            user=config.pguser,
            password=config.pgpassword,
            sslmode=config.pgsslmode,
        ) as repository:
            trigger_type = "manual_single" if args.author_id and len(args.author_id) == 1 else ("manual_batch" if args.author_limit != 1 else "manual_single")
            run_id = repository.create_pipeline_run(trigger_type=trigger_type, status="running")
            logger.info("pipeline_run_started run_id=%s trigger_type=%s", run_id, trigger_type)
            if workers == 1:
                if runner_repository is not None:
                    runner_repository.close()
                runner, runner_repository = _create_runner(config, limiter, run_id=run_id)
            if args.author_id:
                requested_ids = {item.strip() for item in args.author_id if item.strip()}
                id_candidates = _build_author_id_candidates(requested_ids)
                logger.info(
                    "starting load authors mode=author_id requested=%s workers=%s progress_every=%s",
                    len(requested_ids),
                    workers,
                    progress_every,
                )
                authors = repository.list_author_records_from_authors_analysis_by_ids(id_candidates)
                logger.info(
                    "authors_loaded_from_authors_analysis=%s requested=%s workers=%s progress_every=%s",
                    len(authors),
                    len(requested_ids),
                    workers,
                    progress_every,
                )
                total = len(authors)
                if workers == 1 and runner is not None:
                    batch_stats, done = _run_serial(
                        runner,
                        authors,
                        progress_every,
                        logger,
                        done_start=done,
                        total=total,
                        start_monotonic=start_monotonic,
                    )
                else:
                    batch_stats, done = _run_parallel(
                        config,
                        limiter,
                        run_id,
                        authors,
                        workers,
                        progress_every,
                        logger,
                        done_start=done,
                        total=total,
                        start_monotonic=start_monotonic,
                    )
                stats.update(batch_stats)
            else:
                logger.info(
                    "starting load authors mode=stream offset=%s limit=%s fetch_batch_size=%s workers=%s progress_every=%s",
                    start_offset,
                    total_target if total_target is not None else "all",
                    fetch_batch_size,
                    workers,
                    progress_every,
                )
                while True:
                    remaining = None if total_target is None else max(total_target - done, 0)
                    if remaining == 0:
                        break
                    current_limit = fetch_batch_size if remaining is None else min(fetch_batch_size, remaining)
                    current_offset = start_offset + done
                    retry = 0
                    while True:
                        try:
                            authors = repository.list_author_records_from_authors_analysis(
                                limit=current_limit,
                                offset=current_offset,
                            )
                            break
                        except psycopg.OperationalError as exc:
                            retry += 1
                            if retry > batch_load_max_retries:
                                raise
                            wait_seconds = min(2 ** (retry - 1), 8)
                            logger.warning(
                                "authors_batch_load_retry offset=%s limit=%s retry=%s/%s wait_seconds=%s error=%s",
                                current_offset,
                                current_limit,
                                retry,
                                batch_load_max_retries,
                                wait_seconds,
                                str(exc),
                            )
                            repository.reconnect()
                            time.sleep(wait_seconds)
                    if not authors:
                        logger.info("authors_batch_loaded=0 offset=%s stop_reason=no_more_rows", current_offset)
                        break

                    logger.info(
                        "authors_batch_loaded=%s offset=%s requested_batch=%s",
                        len(authors),
                        current_offset,
                        current_limit,
                    )

                    if workers == 1 and runner is not None:
                        batch_stats, done = _run_serial(
                            runner,
                            authors,
                            progress_every,
                            logger,
                            done_start=done,
                            total=total_target,
                            start_monotonic=start_monotonic,
                        )
                    else:
                        batch_stats, done = _run_parallel(
                            config,
                            limiter,
                            run_id,
                            authors,
                            workers,
                            progress_every,
                            logger,
                            done_start=done,
                            total=total_target,
                            start_monotonic=start_monotonic,
                        )
                    stats.update(batch_stats)

                    if len(authors) < current_limit:
                        logger.info(
                            "authors_batch_loaded=%s offset=%s stop_reason=last_partial_batch",
                            len(authors),
                            current_offset,
                        )
                        break

            final_run_status = "partial_failed" if stats.get("error", 0) > 0 else "success"
            repository.finish_pipeline_run(run_id=run_id, status=final_run_status)
            logger.info("pipeline_run_finished run_id=%s status=%s", run_id, final_run_status)
    except Exception:
        if run_id:
            with PgRepository(
                host=config.pghost,
                port=config.pgport,
                database=config.pgdatabase,
                user=config.pguser,
                password=config.pgpassword,
                sslmode=config.pgsslmode,
            ) as fail_repo:
                fail_repo.finish_pipeline_run(run_id=run_id, status="failed")
            logger.exception("pipeline_run_failed run_id=%s", run_id)
        raise

    if runner_repository is not None:
        runner_repository.close()

    logger.info("stats=%s", dict(stats))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
