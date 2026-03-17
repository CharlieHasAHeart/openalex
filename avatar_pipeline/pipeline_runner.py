from __future__ import annotations

import logging
import time
from collections import Counter

from avatar_pipeline.avatar_gate import validate_image_candidate
from avatar_pipeline.config import PipelineConfig
from avatar_pipeline.models import AuthorRecord, ImageCandidate, PipelineResult
from avatar_pipeline.oss_uploader import OssUploader, sha256_hex
from avatar_pipeline.pg_repository import PgRepository
from avatar_pipeline.web_search_client import SearchCandidate, WebSearchClient

logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        web_search_client: WebSearchClient,
        oss_uploader: OssUploader,
        pg_repository: PgRepository,
        run_store,
    ) -> None:
        self._config = config
        self._web_search = web_search_client
        self._oss = oss_uploader
        self._repo = pg_repository
        self._run_store = run_store
        self._stats: Counter[str] = Counter()

    def _log_step(self, author: AuthorRecord, step: str, **fields: object) -> None:
        extra = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
        if extra:
            logger.info(
                "pipeline_step run_id=%s author_id=%s display_name=%r step=%s %s",
                self._run_store.run_id,
                author.author_id,
                author.display_name,
                step,
                extra,
            )
            return
        logger.info(
            "pipeline_step run_id=%s author_id=%s display_name=%r step=%s",
            self._run_store.run_id,
            author.author_id,
            author.display_name,
            step,
        )

    @property
    def stats(self) -> Counter[str]:
        return self._stats

    def run_for_author_seed(self, author: AuthorRecord) -> str:
        try:
            result = self._process(author)
        except Exception as exc:
            logger.exception("pipeline_author_failed author_id=%s error_message=%s", author.author_id, str(exc))
            result = PipelineResult(
                author_id=author.author_id,
                status="error",
                error_message=str(exc),
                failure_reason="pipeline_error",
            )
        self._run_store.record_author(author, result)
        self._stats[result.status] += 1
        return result.status

    def _candidate_to_dict(self, candidate: SearchCandidate) -> dict[str, object]:
        return {
            "image_url": candidate.image_url,
            "source_url": candidate.source_url,
            "linked_profile_url": candidate.linked_profile_url,
            "linked_profile_domain": candidate.linked_profile_domain,
            "title": candidate.title,
            "snippet": candidate.snippet,
            "mime": candidate.mime,
            "width": candidate.width,
            "height": candidate.height,
            "size_bytes": candidate.size_bytes,
            "is_valid_image": candidate.is_valid_image,
            "invalid_reason": candidate.invalid_reason,
            "score": candidate.score,
            "source_type": candidate.source_type,
            "alt_text": candidate.alt_text,
        }

    def _failure_result(
        self,
        author: AuthorRecord,
        outcome,
        failure_reason: str,
        *,
        selected_candidate: dict[str, object] | None = None,
        status: str = "no_image",
    ) -> PipelineResult:
        return PipelineResult(
            author_id=author.author_id,
            status=status,
            error_message=failure_reason,
            failure_reason=failure_reason,
            selected_candidate=selected_candidate,
            profile_pages=outcome.profile_pages,
            image_candidates=outcome.image_candidates,
            filtered_candidates=outcome.filtered_candidates,
            raw_content=outcome.raw_content,
            response_text=outcome.response_text,
            abandon_reason_log=outcome.abandon_reason_log,
            usage_total_tokens=outcome.usage_total_tokens,
        )

    def _process(self, author: AuthorRecord) -> PipelineResult:
        existing_avatar = self._repo.get_author_avatar_record(author.author_id)
        self._log_step(author, "load_author", has_existing_avatar=bool(existing_avatar))
        max_attempts = 2
        attempt = 1
        self._log_step(
            author,
            "qwen_search_image_start",
            provider_mode="qwen_web_search_image",
            attempt=attempt,
            max_attempts=max_attempts,
        )
        outcome = self._web_search.search_author(author)
        while outcome.failure_reason == "qwen_request_timeout" and attempt < max_attempts:
            attempt += 1
            self._log_step(
                author,
                "qwen_search_image_retry",
                failure_reason=outcome.failure_reason,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            time.sleep(3)
            outcome = self._web_search.search_author(author)
        self._log_step(
            author,
            "qwen_search_image_done",
            profile_pages=len(outcome.profile_pages),
            image_candidates=len(outcome.image_candidates),
            filtered_candidates=len(outcome.filtered_candidates),
            failure_reason=outcome.failure_reason,
            attempts=attempt,
            provider_mode="qwen_web_search_image",
        )
        if not outcome.candidates:
            failure_reason = outcome.failure_reason or "qwen_web_search_image_no_candidates"
            self._log_step(author, "qwen_search_empty", failure_reason=failure_reason)
            return self._failure_result(author, outcome, failure_reason)

        selected = outcome.candidates[0]
        selected = self._web_search.enrich_candidate_image_metadata(selected, self._config.allowed_mime)
        selected_dict = self._candidate_to_dict(selected)
        self._log_step(
            author,
            "select_first_extracted_url",
            image_url=selected.image_url,
            source_url=selected.source_url,
        )
        if selected.is_valid_image is False:
            failure_reason = selected.invalid_reason or "invalid_image"
            self._log_step(author, "selected_candidate_invalid", failure_reason=failure_reason)
            return self._failure_result(
                author,
                outcome,
                failure_reason,
                selected_candidate=selected_dict,
                status="invalid_image",
            )

        self._log_step(author, "prepare_upload_start", image_url=selected.image_url)
        image_bytes, actual_mime = self._web_search.download_image(selected.image_url)
        candidate = ImageCandidate(
            commons_file=selected.source_url,
            download_url=selected.image_url,
            mime=actual_mime,
            width=selected.width or 0,
            height=selected.height or 0,
            size_bytes=len(image_bytes),
        )
        is_valid, image_error = validate_image_candidate(
            candidate,
            image_bytes,
            self._config.allowed_mime,
            self._config.min_image_edge_px,
        )
        if not is_valid:
            self._log_step(author, "prepare_upload_invalid", failure_reason=image_error)
            return self._failure_result(
                author,
                outcome,
                image_error or "invalid_image",
                selected_candidate=selected_dict,
                status="invalid_image",
            )

        sha256 = sha256_hex(image_bytes)
        if existing_avatar and existing_avatar.get("content_sha256") == sha256 and existing_avatar.get("oss_object_key"):
            self._log_step(author, "reuse_existing_avatar", content_sha256=sha256, oss_url=existing_avatar.get("oss_url"))
            result = PipelineResult(
                author_id=author.author_id,
                status="ok",
                commons_file=selected.source_url,
                content_sha256=sha256,
                oss_object_key=existing_avatar["oss_object_key"],
                oss_url=existing_avatar.get("oss_url"),
                selected_candidate=selected_dict,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=outcome.raw_content,
                response_text=outcome.response_text,
                abandon_reason_log=outcome.abandon_reason_log,
                usage_total_tokens=outcome.usage_total_tokens,
            )
            # Refresh updated_at/commons_file/oss_url to keep DB row consistent with current run.
            self._repo.upsert_result(result)
            return result

        self._log_step(author, "upload_oss_start", content_sha256=sha256)
        object_key = self._oss.build_object_key(author.orcid or author.author_id, sha256, actual_mime)
        oss_url = self._oss.upload(object_key, image_bytes, actual_mime)
        self._log_step(author, "upload_oss_done", oss_url=oss_url)
        result = PipelineResult(
            author_id=author.author_id,
            status="ok",
            commons_file=selected.source_url,
            content_sha256=sha256,
            oss_object_key=object_key,
            oss_url=oss_url,
            selected_candidate=selected_dict,
            profile_pages=outcome.profile_pages,
            image_candidates=outcome.image_candidates,
            filtered_candidates=outcome.filtered_candidates,
            raw_content=outcome.raw_content,
            response_text=outcome.response_text,
            abandon_reason_log=outcome.abandon_reason_log,
            usage_total_tokens=outcome.usage_total_tokens,
        )
        self._log_step(author, "upsert_authors_avatars_start")
        self._repo.upsert_result(result)
        self._log_step(author, "upsert_authors_avatars_done", oss_url=result.oss_url)
        return result
