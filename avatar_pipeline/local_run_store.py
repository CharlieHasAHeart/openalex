from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avatar_pipeline.models import AuthorRecord, PipelineResult


class LocalRunStore:
    def __init__(
        self,
        base_dir: str,
        config_snapshot: dict[str, Any] | None = None,
        resume_run_id: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        self._base_dir = Path(base_dir)
        self._config_snapshot = config_snapshot or {}
        self._lock = threading.Lock()
        self._author_count = 0
        self._failure_count = 0
        self._success_count = 0
        self._status_counts: Counter[str] = Counter()
        self._processed_author_ids: set[str] = set()
        self._usage_total_tokens_sum = 0
        self._usage_total_tokens_count = 0
        self._source_total_authors = 0
        self._scheduled_authors = 0
        self._input_summary: dict[str, Any] = {}
        self._resumed = bool(resume_run_id)

        if resume_run_id:
            self.run_id = resume_run_id
            self.base_path = self._find_existing_run_path(self._base_dir, resume_run_id)
            self.run_date = self.base_path.parent.name
            self.started_at = now.isoformat()
        else:
            self.run_id = str(uuid.uuid4())
            self.run_date = now.strftime("%Y-%m-%d")
            self.started_at = now.isoformat()
            self.base_path = self._base_dir / self.run_date / self.run_id
            self.base_path.mkdir(parents=True, exist_ok=True)

        self.summary_path = self.base_path / "summary.json"
        self.author_runs_path = self.base_path / "author_runs.jsonl"
        self.successes_path = self.base_path / "successes.jsonl"
        self.failures_path = self.base_path / "failures.jsonl"
        self.planned_authors_path = self.base_path / "planned_authors.jsonl"
        if resume_run_id:
            self.started_at = self._load_started_at() or self.started_at
        self.author_runs_path.touch(exist_ok=True)
        self.successes_path.touch(exist_ok=True)
        self.failures_path.touch(exist_ok=True)
        self.planned_authors_path.touch(exist_ok=True)
        self._load_existing_records()
        self._write_summary_snapshot(
            elapsed_seconds=0.0,
            run_status="running",
            last_status=None,
            progress_done=0,
            progress_total=0,
        )

    def _find_existing_run_path(self, base_dir: Path, run_id: str) -> Path:
        candidates = sorted(base_dir.glob(f"*/{run_id}"))
        if not candidates:
            raise FileNotFoundError(f"resume run_id not found under {base_dir}: {run_id}")
        return candidates[0]

    def _load_started_at(self) -> str | None:
        if not self.summary_path.exists():
            return None
        try:
            payload = json.loads(self.summary_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        value = payload.get("started_at")
        return str(value).strip() if value else None

    def _load_existing_records(self) -> None:
        if not self.author_runs_path.exists():
            return
        try:
            with self.author_runs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    author_id = str(row.get("author_id") or "").strip()
                    status = str(row.get("final_status") or "").strip()
                    if author_id:
                        self._processed_author_ids.add(author_id)
                    if status:
                        self._status_counts[status] += 1
                        if status == "ok":
                            self._success_count += 1
                        else:
                            self._failure_count += 1
                    usage_total_tokens = row.get("usage_total_tokens")
                    if isinstance(usage_total_tokens, int) and usage_total_tokens >= 0:
                        self._usage_total_tokens_sum += usage_total_tokens
                        self._usage_total_tokens_count += 1
                    self._author_count += 1
        except Exception:
            return

    def processed_author_ids(self) -> set[str]:
        with self._lock:
            return set(self._processed_author_ids)

    def set_run_scope(
        self,
        source_total_authors: int,
        scheduled_authors: int,
        input_summary: dict[str, Any] | None = None,
        planned_authors: list[AuthorRecord] | None = None,
    ) -> None:
        with self._lock:
            self._source_total_authors = max(0, int(source_total_authors))
            self._scheduled_authors = max(0, int(scheduled_authors))
            self._input_summary = input_summary or {}
            if planned_authors is not None:
                with self.planned_authors_path.open("w", encoding="utf-8") as f:
                    for author in planned_authors:
                        f.write(
                            json.dumps(
                                {
                                    "author_id": author.author_id,
                                    "display_name": author.display_name,
                                    "institution_name": author.institution_name,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
            self._write_summary_snapshot(
                elapsed_seconds=0.0,
                run_status="running",
                last_status=None,
                progress_done=0,
                progress_total=self._scheduled_authors,
            )

    def update_progress(
        self,
        *,
        elapsed_seconds: float,
        last_status: str | None,
        progress_done: int,
        progress_total: int,
    ) -> None:
        with self._lock:
            self._write_summary_snapshot(
                elapsed_seconds=elapsed_seconds,
                run_status="running",
                last_status=last_status,
                progress_done=progress_done,
                progress_total=progress_total,
            )

    def record_author(self, author: AuthorRecord, result: PipelineResult) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        row = {
            "author_id": author.author_id,
            "display_name": author.display_name,
            "institution_name": author.institution_name,
            "source_pages": result.source_pages,
            "image_candidates": result.image_candidates,
            "filtered_candidates": result.filtered_candidates,
            "selected_candidate": result.selected_candidate,
            "final_status": result.status,
            "failure_reason": result.failure_reason or result.error_message,
            "abandon_reason_log": result.abandon_reason_log,
            "oss_url": result.oss_url,
            "content_sha256": result.content_sha256,
            "raw_content": result.raw_content,
            "response_text": result.response_text,
            "usage_total_tokens": result.usage_total_tokens,
            "timestamp": timestamp,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._author_count += 1
            self._processed_author_ids.add(author.author_id)
            self._status_counts[result.status] += 1
            if isinstance(result.usage_total_tokens, int) and result.usage_total_tokens >= 0:
                self._usage_total_tokens_sum += result.usage_total_tokens
                self._usage_total_tokens_count += 1
            with self.author_runs_path.open("a", encoding="utf-8") as f:
                f.write(line)
            if result.status == "ok":
                self._success_count += 1
                with self.successes_path.open("a", encoding="utf-8") as f:
                    f.write(line)
            else:
                self._failure_count += 1
                with self.failures_path.open("a", encoding="utf-8") as f:
                    f.write(line)

    def _write_summary_snapshot(
        self,
        *,
        elapsed_seconds: float,
        run_status: str,
        last_status: str | None,
        progress_done: int,
        progress_total: int,
        finished_at: str | None = None,
    ) -> None:
        if self._source_total_authors > 0:
            remaining = max(0, self._source_total_authors - self._author_count)
        else:
            remaining = max(0, progress_total - progress_done)
        usage_avg = (
            self._usage_total_tokens_sum / self._usage_total_tokens_count
            if self._usage_total_tokens_count > 0
            else None
        )
        payload = {
            "run_id": self.run_id,
            "run_date": self.run_date,
            "started_at": self.started_at,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
            "finished_at": finished_at,
            "run_status": run_status,
            "resumed": self._resumed,
            "source_total_authors": self._source_total_authors,
            "scheduled_authors": self._scheduled_authors,
            "recorded_authors": self._author_count,
            "success_count": self._success_count,
            "failure_count": self._failure_count,
            "remaining_authors": remaining,
            "elapsed_seconds": round(float(elapsed_seconds), 3),
            "stats": dict(self._status_counts),
            "last_status": last_status,
            "progress_done": progress_done,
            "progress_total": progress_total,
            "usage_total_tokens_sum": self._usage_total_tokens_sum,
            "usage_total_tokens_count": self._usage_total_tokens_count,
            "usage_total_tokens_avg": round(usage_avg, 3) if usage_avg is not None else None,
            "config_snapshot": self._config_snapshot,
            "input_summary": self._input_summary,
            "files": {
                "summary": str(self.summary_path),
                "planned_authors": str(self.planned_authors_path),
                "author_runs": str(self.author_runs_path),
                "successes": str(self.successes_path),
                "failures": str(self.failures_path),
            },
        }
        self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def finalize(
        self,
        stats: dict[str, int],
        total_authors: int,
        elapsed_seconds: float,
        input_summary: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._status_counts = Counter(stats)
            self._input_summary = input_summary or self._input_summary
            self._source_total_authors = max(0, total_authors)
            self._write_summary_snapshot(
                elapsed_seconds=elapsed_seconds,
                run_status="finished",
                last_status=None,
                progress_done=self._author_count,
                progress_total=max(self._scheduled_authors, self._author_count),
                finished_at=datetime.now(timezone.utc).isoformat(),
            )
