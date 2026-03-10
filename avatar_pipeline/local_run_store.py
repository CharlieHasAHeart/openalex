from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from avatar_pipeline.models import AuthorRecord, PipelineResult


class LocalRunStore:
    def __init__(self, base_dir: str, config_snapshot: dict[str, Any] | None = None) -> None:
        now = datetime.now(timezone.utc)
        self.run_id = str(uuid.uuid4())
        self.run_date = now.strftime("%Y-%m-%d")
        self.started_at = now.isoformat()
        self._config_snapshot = config_snapshot or {}
        self._lock = threading.Lock()
        self._author_count = 0
        self._failure_count = 0

        self.base_path = Path(base_dir) / self.run_date / self.run_id
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.summary_path = self.base_path / "summary.json"
        self.author_runs_path = self.base_path / "author_runs.jsonl"
        self.failures_path = self.base_path / "failures.jsonl"
        self.author_runs_path.touch(exist_ok=True)
        self.failures_path.touch(exist_ok=True)

    def record_author(self, author: AuthorRecord, result: PipelineResult) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        row = {
            "author_id": author.author_id,
            "display_name": author.display_name,
            "institution_name": author.institution_name,
            "profile_pages": result.profile_pages,
            "image_candidates": result.image_candidates,
            "filtered_candidates": result.filtered_candidates,
            "selected_candidate": result.selected_candidate,
            "final_status": result.status,
            "failure_reason": result.failure_reason or result.error_message,
            "oss_url": result.oss_url,
            "content_sha256": result.content_sha256,
            "timestamp": timestamp,
        }
        line = json.dumps(row, ensure_ascii=False) + "\n"
        with self._lock:
            self._author_count += 1
            with self.author_runs_path.open("a", encoding="utf-8") as f:
                f.write(line)
            if result.status != "ok":
                self._failure_count += 1
                with self.failures_path.open("a", encoding="utf-8") as f:
                    f.write(line)

    def finalize(
        self,
        stats: dict[str, int],
        total_authors: int,
        elapsed_seconds: float,
        input_summary: dict[str, Any] | None = None,
    ) -> None:
        payload = {
            "run_id": self.run_id,
            "run_date": self.run_date,
            "started_at": self.started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "total_authors": total_authors,
            "recorded_authors": self._author_count,
            "failure_count": self._failure_count,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "stats": stats,
            "config_snapshot": self._config_snapshot,
            "input_summary": input_summary or {},
            "files": {
                "summary": str(self.summary_path),
                "author_runs": str(self.author_runs_path),
                "failures": str(self.failures_path),
            },
        }
        self.summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
