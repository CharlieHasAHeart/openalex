from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import threading
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAlex author avatar pipeline")
    parser.add_argument(
        "--author-id",
        action="append",
        default=[],
        help="Specific OpenAlex author id (URL or A-ID). Can repeat.",
    )
    parser.add_argument("--author-ids-file", default="", help="Path to JSON/JSONL file containing author_id values.")
    parser.add_argument("--author-limit", type=int, default=0, help="Limit authors loaded from DB author input set. 0 means all.")
    parser.add_argument("--author-offset", type=int, default=0, help="Skip first N authors in DB author input set.")
    parser.add_argument("--author-id-start", default="", help="Start DB scan from this author_id (inclusive).")
    parser.add_argument("--workers", type=int, default=1, help="Worker threads for concurrent processing. 1 means serial.")
    parser.add_argument("--progress-every", type=int, default=50, help="Emit progress log every N authors. 0 disables progress logs.")
    parser.add_argument("--runs-dir", default="runs", help="Directory for local run artifacts.")
    parser.add_argument("--resume-run-id", default="", help="Resume from an existing run_id under runs-dir.")
    parser.add_argument("--log-level", default="INFO")
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
            candidate_id = urlparse(cleaned).path.rsplit("/", 1)[-1].strip()
            if candidate_id:
                candidates.add(candidate_id)
    return sorted(candidates)


def _load_author_ids_from_file(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"author ids file not found: {path}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if p.suffix.lower() == ".jsonl":
        rows: list[object] = []
        for i, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except Exception as exc:
                raise ValueError(f"invalid jsonl at line {i}: {exc}") from exc
    else:
        payload = json.loads(text)
        rows = payload if isinstance(payload, list) else [payload]

    author_ids: list[str] = []
    for item in rows:
        if isinstance(item, dict):
            author_id = item.get("author_id")
            author_id_text = str(author_id).strip() if author_id is not None else ""
            if author_id_text:
                author_ids.append(author_id_text)
        elif isinstance(item, str) and item.strip():
            author_ids.append(item.strip())
    return author_ids


def _resolve_db_scan_cursor(
    user_start_author_id: str | None,
    resume_max_processed_author_id: str | None,
) -> tuple[str | None, bool]:
    if user_start_author_id and resume_max_processed_author_id:
        if user_start_author_id > resume_max_processed_author_id:
            return user_start_author_id, False
        return resume_max_processed_author_id, True
    if user_start_author_id:
        return user_start_author_id, False
    if resume_max_processed_author_id:
        return resume_max_processed_author_id, True
    return None, False


def _create_runner(config, limiter, run_store):
    from avatar_pipeline.http import HttpClient
    from avatar_pipeline.oss_uploader import OssUploader
    from avatar_pipeline.pg_repository import PgRepository
    from avatar_pipeline.pipeline_runner import PipelineRunner
    from avatar_pipeline.web_search_client import WebSearchClient

    http_client = HttpClient(
        timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        rate_limiter=limiter,
        retry_base_seconds=config.retry_base_seconds,
        retry_max_seconds=config.retry_max_seconds,
        retry_jitter_ratio=config.retry_jitter_ratio,
        retry_429_min_delay_seconds=config.retry_429_min_delay_seconds,
    )
    web_search_client = WebSearchClient(
        http=http_client,
        max_candidates=config.qwen_max_candidates,
        qwen_api_key=config.qwen_api_key,
        qwen_base_url=config.qwen_base_url,
        qwen_model=config.qwen_model,
        qwen_timeout_seconds=config.qwen_timeout_seconds,
        qwen_min_call_interval_seconds=config.qwen_min_call_interval_seconds,
        qwen_enable_web_search=config.qwen_enable_web_search,
        qwen_min_confidence=config.qwen_min_confidence,
        qwen_max_output_tokens=config.qwen_max_output_tokens,
        qwen_sdk_max_retries=config.qwen_sdk_max_retries,
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
    pg_repository = PgRepository(
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
        oss_uploader=oss_uploader,
        pg_repository=pg_repository,
        run_store=run_store,
    )
    return runner, pg_repository


def _maybe_log_progress(
    logger: logging.Logger,
    done: int,
    total: int | None,
    start_monotonic: float,
    last_status: str,
    progress_every: int,
    stats: Counter[str] | None = None,
) -> None:
    if done <= 0 or progress_every <= 0:
        return
    if done % progress_every != 0 and (total is None or done != total):
        return
    elapsed = max(time.monotonic() - start_monotonic, 1e-9)
    rate = done / elapsed
    ok_count = int((stats or {}).get("ok", 0))
    failed_count = done - ok_count
    if total is None:
        logger.info(
            "progress done=%s total=unknown remaining=unknown ok=%s failed=%s rate=%.2f/s last_status=%s",
            done,
            ok_count,
            failed_count,
            rate,
            last_status,
        )
        return
    remaining = max(0, total - done)
    logger.info(
        "progress done=%s total=%s remaining=%s pct=%.1f ok=%s failed=%s rate=%.2f/s last_status=%s",
        done,
        total,
        remaining,
        done / total * 100.0,
        ok_count,
        failed_count,
        rate,
        last_status,
    )


def _run_parallel(config, limiter, run_store, authors, workers: int, progress_every: int, logger, total, start_monotonic):
    stats: Counter[str] = Counter()
    local = threading.local()
    repositories = []
    repositories_lock = threading.Lock()
    done = 0

    def _get_runner():
        runner = getattr(local, "runner", None)
        if runner is not None:
            return runner
        runner, repository = _create_runner(config, limiter, run_store)
        local.runner = runner
        with repositories_lock:
            repositories.append(repository)
        return runner

    def _task(author):
        runner = _get_runner()
        return runner.run_for_author_seed(author)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for stat_key in executor.map(_task, authors):
                stats[stat_key] += 1
                done += 1
                _maybe_log_progress(logger, done, total, start_monotonic, stat_key, progress_every, stats=stats)
                run_store.update_progress(
                    elapsed_seconds=time.monotonic() - start_monotonic,
                    last_status=stat_key,
                    progress_done=done,
                    progress_total=total or 0,
                )
    finally:
        for repository in repositories:
            repository.close()
    return stats


def _run_serial(runner, run_store, authors, progress_every: int, logger, total, start_monotonic):
    stats: Counter[str] = Counter()
    for done, author in enumerate(authors, start=1):
        stat_key = runner.run_for_author_seed(author)
        stats[stat_key] += 1
        _maybe_log_progress(logger, done, total, start_monotonic, stat_key, progress_every, stats=stats)
        run_store.update_progress(
            elapsed_seconds=time.monotonic() - start_monotonic,
            last_status=stat_key,
            progress_done=done,
            progress_total=total or 0,
        )
    return stats


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    from avatar_pipeline.config import PipelineConfig, load_dotenv
    from avatar_pipeline.http import RateLimiter
    from avatar_pipeline.local_run_store import LocalRunStore
    from avatar_pipeline.pg_repository import PgRepository

    load_dotenv(".env")
    config = PipelineConfig.from_env()
    limiter = RateLimiter(config.global_qps_limit)
    start_monotonic = time.monotonic()

    file_author_ids: list[str] = []
    if args.author_ids_file.strip():
        try:
            file_author_ids = _load_author_ids_from_file(args.author_ids_file.strip())
        except Exception as exc:
            logger.exception("author_ids_file_load_failed path=%s error=%s", args.author_ids_file, exc)
            return 2

    requested_author_ids = [item.strip() for item in args.author_id if item.strip()] + file_author_ids
    has_requested_author_ids = bool(requested_author_ids)
    user_start_author_id = args.author_id_start.strip() or None

    run_store = LocalRunStore(
        base_dir=args.runs_dir,
        resume_run_id=args.resume_run_id.strip() or None,
        config_snapshot={
            "qwen_model": config.qwen_model,
            "qwen_enable_web_search": config.qwen_enable_web_search,
            "qwen_min_confidence": config.qwen_min_confidence,
            "qwen_max_output_tokens": config.qwen_max_output_tokens,
            "qwen_max_candidates": config.qwen_max_candidates,
            "qwen_min_call_interval_seconds": config.qwen_min_call_interval_seconds,
            "qwen_sdk_max_retries": config.qwen_sdk_max_retries,
        },
    )
    resume_processed = run_store.processed_author_ids()
    resume_max_processed_author_id = run_store.max_processed_author_id() if args.resume_run_id.strip() else None
    db_cursor_author_id, db_cursor_exclusive = _resolve_db_scan_cursor(
        user_start_author_id=user_start_author_id,
        resume_max_processed_author_id=resume_max_processed_author_id,
    )

    with PgRepository(
        host=config.pghost,
        port=config.pgport,
        database=config.pgdatabase,
        user=config.pguser,
        password=config.pgpassword,
        sslmode=config.pgsslmode,
    ) as repository:
        if has_requested_author_ids:
            normalized_requested_ids = _build_author_id_candidates(set(requested_author_ids))
            authors = repository.list_author_records_by_ids(normalized_requested_ids)
        else:
            limit = args.author_limit if args.author_limit > 0 else None
            authors = repository.list_author_records(
                limit=limit,
                offset=max(args.author_offset, 0),
                start_author_id=db_cursor_author_id,
                start_exclusive=db_cursor_exclusive,
            )

    loaded_author_count = len(authors)
    logger.info(
        "pipeline_run_loaded_authors total=%s explicit=%s workers=%s runs_dir=%s db_cursor_author_id=%s db_cursor_exclusive=%s",
        loaded_author_count,
        has_requested_author_ids,
        max(1, args.workers),
        args.runs_dir,
        db_cursor_author_id,
        db_cursor_exclusive,
    )
    if resume_processed:
        authors = [author for author in authors if author.author_id not in resume_processed]
    scheduled_author_count = len(authors)
    if args.resume_run_id.strip():
        logger.info(
            "pipeline_run_resume_loaded run_id=%s source_total=%s already_processed=%s remaining=%s",
            run_store.run_id,
            loaded_author_count,
            len(resume_processed),
            scheduled_author_count,
        )
    run_store.set_run_scope(
        source_total_authors=loaded_author_count,
        scheduled_authors=scheduled_author_count,
        input_summary={
            "explicit_author_ids": len(requested_author_ids),
            "author_offset": max(args.author_offset, 0),
            "author_limit": args.author_limit,
            "author_id_start": user_start_author_id,
            "db_cursor_author_id": db_cursor_author_id,
            "db_cursor_exclusive": db_cursor_exclusive,
            "resume_run_id": args.resume_run_id.strip() or None,
            "already_processed": len(resume_processed),
        },
        planned_authors=authors,
    )

    if max(1, args.workers) == 1:
        runner, runner_repository = _create_runner(config, limiter, run_store)
        try:
            stats = _run_serial(
                runner=runner,
                run_store=run_store,
                authors=authors,
                progress_every=max(args.progress_every, 0),
                logger=logger,
                total=scheduled_author_count,
                start_monotonic=start_monotonic,
            )
        finally:
            runner_repository.close()
    else:
        stats = _run_parallel(
            config=config,
            limiter=limiter,
            run_store=run_store,
            authors=authors,
            workers=max(1, args.workers),
            progress_every=max(args.progress_every, 0),
            logger=logger,
            total=scheduled_author_count,
            start_monotonic=start_monotonic,
        )

    elapsed_seconds = time.monotonic() - start_monotonic
    run_store.finalize(
        stats=dict(stats),
        total_authors=loaded_author_count,
        elapsed_seconds=elapsed_seconds,
        input_summary={
            "explicit_author_ids": len(requested_author_ids),
            "author_offset": max(args.author_offset, 0),
            "author_limit": args.author_limit,
            "author_id_start": user_start_author_id,
            "db_cursor_author_id": db_cursor_author_id,
            "db_cursor_exclusive": db_cursor_exclusive,
            "resume_run_id": args.resume_run_id.strip() or None,
            "already_processed": len(resume_processed),
            "remaining_after_filter": scheduled_author_count,
        },
    )
    logger.info(
        "pipeline_run_finished run_id=%s total=%s elapsed_seconds=%.2f stats=%s",
        run_store.run_id,
        scheduled_author_count,
        elapsed_seconds,
        dict(stats),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
