from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone

from avatar_pipeline.avatar_gate import validate_image_candidate
from avatar_pipeline.config import PipelineConfig
from avatar_pipeline.llm_matcher import LlmMatcher
from avatar_pipeline.models import AuthorRecord, ImageCandidate, PipelineResult
from avatar_pipeline.oss_uploader import OssUploader, sha256_hex
from avatar_pipeline.pg_repository import PgRepository
from avatar_pipeline.web_search_client import WebSearchClient


logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        web_search_client: WebSearchClient,
        llm_matcher: LlmMatcher,
        oss_uploader: OssUploader,
        pg_repository: PgRepository,
    ) -> None:
        self._config = config
        self._web_search = web_search_client
        self._matcher = llm_matcher
        self._oss = oss_uploader
        self._repo = pg_repository
        self._stats: Counter[str] = Counter()

    @property
    def stats(self) -> Counter[str]:
        return self._stats

    def run_for_author_seed(self, author: AuthorRecord) -> str:
        try:
            existing_avatar = self._repo.get_author_avatar_record(author.author_id)
            if existing_avatar and self._should_skip_by_db_state(existing_avatar):
                status = existing_avatar.get("status") or "unknown"
                stat_key = f"skipped_recent_{status}"
                self._stats[stat_key] += 1
                return stat_key

            if not author.orcid:
                stat_key = "skipped_no_orcid"
                self._stats[stat_key] += 1
                return stat_key

            self._stats["authors_processed"] += 1
            result = self.run_for_author(author, existing_avatar)
            return result.status
        except Exception as exc:
            self._stats["error"] += 1
            logger.exception(
                "pipeline_seed_failed author_id=%s orcid=%s error_message=%s",
                author.author_id,
                author.orcid,
                str(exc),
            )
            return "error"

    def run_for_author(self, author: AuthorRecord, existing_avatar: dict | None = None) -> PipelineResult:
        try:
            result = self._process(author, existing_avatar)
        except Exception as exc:
            result = PipelineResult(
                author_id=author.author_id,
                status="error",
                error_message=str(exc),
            )

        persist_error: str | None = None
        try:
            self._repo.upsert_result(result)
        except Exception as exc:
            persist_error = str(exc)
            if result.status != "error":
                result = PipelineResult(
                    author_id=author.author_id,
                    status="error",
                    error_message=f"persist_failed:{persist_error}",
                    wikidata_qid=result.wikidata_qid,
                    commons_file=result.commons_file,
                )
            logger.error(
                "pipeline_persist_failed author_id=%s orcid=%s status_before=%s error_message=%s",
                author.author_id,
                author.orcid,
                result.status,
                persist_error,
            )

        self._stats[result.status] += 1
        logger.info(
            "pipeline_result author_id=%s orcid=%s status=%s qid=%s commons_file=%s error_message=%s persist_error=%s",
            author.author_id,
            author.orcid,
            result.status,
            result.wikidata_qid,
            result.commons_file,
            result.error_message,
            persist_error,
        )
        return result

    def _should_skip_by_db_state(self, state: dict) -> bool:
        status = state.get("status")
        updated_at = state.get("updated_at")
        if not status or updated_at is None:
            return False

        window_by_status = {
            "ok": self._config.refresh_ok_days,
            "no_image": self._config.refresh_no_image_days,
            "error": self._config.refresh_error_days,
            "invalid_image": self._config.refresh_error_days,
            "ambiguous": self._config.refresh_ambiguous_days,
            "no_match": self._config.refresh_no_match_days,
        }
        refresh_days = window_by_status.get(status)
        if refresh_days is None:
            return False
        if refresh_days < 0:
            return True

        now = datetime.now(timezone.utc)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=timezone.utc)
        return (now - updated_at) < timedelta(days=refresh_days)

    def _process(self, author: AuthorRecord, existing_avatar: dict | None = None) -> PipelineResult:
        if not author.orcid:
            return PipelineResult(
                author_id=author.author_id,
                status="no_match",
                error_message="missing_orcid",
            )

        candidates = self._web_search.search_image_candidates(author)
        if not candidates:
            return PipelineResult(author_id=author.author_id, status="no_image", error_message="websearch_no_candidates")

        decision = self._matcher.choose_best(author, candidates)
        if not decision:
            return PipelineResult(author_id=author.author_id, status="ambiguous", error_message="llm_no_confident_match")

        chosen = candidates[decision.selected_index]
        image_bytes, actual_mime = self._web_search.download_image(chosen.image_url)
        candidate = ImageCandidate(
            commons_file=chosen.source_url,
            download_url=chosen.image_url,
            mime=actual_mime,
            width=0,
            height=0,
            size_bytes=len(image_bytes),
        )
        is_valid, image_error = validate_image_candidate(
            candidate,
            image_bytes,
            self._config.allowed_mime,
            self._config.min_image_edge_px,
        )
        if not is_valid:
            return PipelineResult(
                author_id=author.author_id,
                status="invalid_image",
                commons_file=chosen.source_url,
                error_message=image_error,
            )

        sha256 = sha256_hex(image_bytes)
        existing = existing_avatar
        if existing and existing.get("content_sha256") == sha256 and existing.get("oss_object_key"):
            return PipelineResult(
                author_id=author.author_id,
                status="ok",
                commons_file=chosen.source_url,
                content_sha256=sha256,
                oss_object_key=existing["oss_object_key"],
                oss_url=existing.get("oss_url"),
            )

        object_key = self._oss.build_object_key(author.orcid or author.author_id, sha256, candidate.mime)
        oss_url = self._oss.upload(object_key, image_bytes, candidate.mime)
        return PipelineResult(
            author_id=author.author_id,
            status="ok",
            commons_file=chosen.source_url,
            content_sha256=sha256,
            oss_object_key=object_key,
            oss_url=oss_url,
        )
