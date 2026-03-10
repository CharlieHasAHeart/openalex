from __future__ import annotations

import logging
from collections import Counter

from avatar_pipeline.avatar_gate import validate_image_candidate
from avatar_pipeline.config import PipelineConfig
from avatar_pipeline.models import AuthorRecord, ImageCandidate, PipelineResult
from avatar_pipeline.oss_uploader import OssUploader, sha256_hex
from avatar_pipeline.pg_repository import PgRepository
from avatar_pipeline.web_search_client import SearchCandidate, SearchOutcome, WebSearchClient

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
        self._context_enrich_limit = 5
        self._image_enrich_limit = 5

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

    def _process(self, author: AuthorRecord) -> PipelineResult:
        existing_avatar = self._repo.get_author_avatar_record(author.author_id)
        self._log_step(author, "load_author", has_existing_avatar=bool(existing_avatar))
        self._log_step(author, "qwen_search_start", provider_mode="qwen")

        outcome = self._web_search.search_author(author)
        self._log_step(
            author,
            "qwen_search_done",
            profile_pages=len(outcome.profile_pages),
            image_candidates=len(outcome.image_candidates),
            filtered_candidates=len(outcome.filtered_candidates),
            failure_reason=outcome.failure_reason,
        )
        if outcome.failure_reason and "institutional mismatch" in outcome.failure_reason.lower():
            self._log_step(author, "qwen_search_identity_mismatch", failure_reason=outcome.failure_reason)
            return PipelineResult(
                author_id=author.author_id,
                status="no_image",
                error_message=outcome.failure_reason,
                failure_reason=outcome.failure_reason,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=outcome.raw_content,
                response_text=outcome.response_text,
                abandon_reason_log=outcome.abandon_reason_log,
            )
        if not outcome.candidates:
            failure_reason = outcome.failure_reason or "qwen_no_candidates"
            self._log_step(author, "qwen_search_empty", failure_reason=failure_reason)
            return PipelineResult(
                author_id=author.author_id,
                status="no_image",
                error_message=failure_reason,
                failure_reason=failure_reason,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=outcome.raw_content,
                response_text=outcome.response_text,
                abandon_reason_log=outcome.abandon_reason_log,
            )

        self._log_step(author, "normalize_candidates", candidates=len(outcome.candidates))
        candidates = self._web_search.enrich_candidates_context(outcome.candidates, limit=self._context_enrich_limit)
        self._log_step(author, "enrich_context_done", candidates=len(candidates))
        candidates = self._web_search.enrich_candidates_image_metadata(
            candidates,
            limit=self._image_enrich_limit,
            allowed_mime=self._config.allowed_mime,
            min_edge_px=self._config.min_image_edge_px,
        )
        self._log_step(author, "enrich_image_metadata_done", candidates=len(candidates))
        candidates = self._web_search.cluster_candidates(candidates, fingerprint_top_n=8)
        self._log_step(author, "dedupe_cluster_done", candidates=len(candidates))
        candidate_pool = list(candidates)
        ranked_candidates = [self._ranked_candidate_summary(candidate) for candidate in candidate_pool[:5]]
        logger.info(
            "pipeline_candidates_for_llm run_id=%s author_id=%s candidates=%s",
            self._run_store.run_id,
            author.author_id,
            ranked_candidates,
        )
        shortlist = candidate_pool[:8]
        compatible_shortlist, dropped_for_multimodal = self._web_search.filter_candidates_for_multimodal(shortlist)
        self._log_step(
            author,
            "llm_input_url_precheck_done",
            shortlist=len(shortlist),
            compatible=len(compatible_shortlist),
            dropped=len(dropped_for_multimodal),
        )
        if not compatible_shortlist:
            failure_reason = "llm_input_no_multimodal_compatible_candidates"
            self._log_step(author, "llm_select_candidate_skipped", failure_reason=failure_reason)
            return PipelineResult(
                author_id=author.author_id,
                status="no_image",
                error_message=failure_reason,
                failure_reason=failure_reason,
                ranked_candidates=ranked_candidates,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=outcome.raw_content,
                response_text=outcome.response_text,
                abandon_reason_log=f"multimodal_url_precheck_dropped={dropped_for_multimodal[:8]}",
            )
        self._log_step(author, "llm_select_candidate_start", candidates=len(candidate_pool), shortlist=len(compatible_shortlist))
        selection = self._web_search.select_avatar_candidate(author, compatible_shortlist, outcome.profile_pages)
        if selection.failure_reason:
            self._log_step(author, "llm_select_candidate_failed", failure_reason=selection.failure_reason)
            return PipelineResult(
                author_id=author.author_id,
                status="no_image",
                error_message=selection.reason or selection.failure_reason,
                failure_reason=selection.failure_reason,
                ranked_candidates=ranked_candidates,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=selection.raw_content or outcome.raw_content,
                response_text=selection.response_text or outcome.response_text,
                abandon_reason_log=outcome.abandon_reason_log,
            )
        if selection.selected_index < 0 or selection.selected_index >= len(compatible_shortlist):
            failure_reason = "llm_avatar_select_no_candidate"
            self._log_step(author, "llm_select_candidate_empty", failure_reason=failure_reason)
            return PipelineResult(
                author_id=author.author_id,
                status="no_image",
                error_message=selection.reason or failure_reason,
                failure_reason=failure_reason,
                ranked_candidates=ranked_candidates,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=selection.raw_content or outcome.raw_content,
                response_text=selection.response_text or outcome.response_text,
                abandon_reason_log=outcome.abandon_reason_log,
            )
        selected = compatible_shortlist[selection.selected_index]

        selected_dict = self._candidate_to_dict(selected)
        selected_dict["llm_selection"] = {
            "selected_index": selection.selected_index,
            "confidence": selection.confidence,
            "reason": selection.reason,
            "failure_reason": selection.failure_reason,
            "shortlist_count": len(compatible_shortlist),
        }
        self._log_step(
            author,
            "llm_select_candidate_done",
            source_url=selected.source_url,
            image_url=selected.image_url,
            is_valid_image=selected.is_valid_image,
        )
        if selected.is_valid_image is False:
            failure_reason = selected.invalid_reason or "invalid_image"
            self._log_step(author, "selected_candidate_invalid", failure_reason=failure_reason)
            return PipelineResult(
                author_id=author.author_id,
                status="invalid_image",
                error_message=failure_reason,
                failure_reason=failure_reason,
                commons_file=selected.source_url,
                ranked_candidates=ranked_candidates,
                selected_candidate=selected_dict,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=outcome.raw_content,
                response_text=outcome.response_text,
                abandon_reason_log=outcome.abandon_reason_log,
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
            return PipelineResult(
                author_id=author.author_id,
                status="invalid_image",
                error_message=image_error,
                failure_reason=image_error,
                commons_file=selected.source_url,
                ranked_candidates=ranked_candidates,
                selected_candidate=selected_dict,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=outcome.raw_content,
                response_text=outcome.response_text,
                abandon_reason_log=outcome.abandon_reason_log,
            )

        sha256 = sha256_hex(image_bytes)
        if existing_avatar and existing_avatar.get("content_sha256") == sha256 and existing_avatar.get("oss_object_key"):
            self._log_step(author, "reuse_existing_avatar", content_sha256=sha256, oss_url=existing_avatar.get("oss_url"))
            return PipelineResult(
                author_id=author.author_id,
                status="ok",
                commons_file=selected.source_url,
                content_sha256=sha256,
                oss_object_key=existing_avatar["oss_object_key"],
                oss_url=existing_avatar.get("oss_url"),
                ranked_candidates=ranked_candidates,
                selected_candidate=selected_dict,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=outcome.raw_content,
                response_text=outcome.response_text,
                abandon_reason_log=outcome.abandon_reason_log,
            )

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
            ranked_candidates=ranked_candidates,
            selected_candidate=selected_dict,
            profile_pages=outcome.profile_pages,
            image_candidates=outcome.image_candidates,
            filtered_candidates=outcome.filtered_candidates,
            raw_content=outcome.raw_content,
            response_text=outcome.response_text,
            abandon_reason_log=outcome.abandon_reason_log,
        )
        self._log_step(author, "upsert_authors_avatars_start")
        self._repo.upsert_result(result)
        self._log_step(author, "upsert_authors_avatars_done", oss_url=result.oss_url)
        logger.info(
            "pipeline_result run_id=%s author_id=%s status=%s source_url=%s oss_url=%s",
            self._run_store.run_id,
            author.author_id,
            result.status,
            selected.source_url,
            result.oss_url,
        )
        return result

    def _candidate_to_dict(self, candidate: SearchCandidate) -> dict[str, object]:
        return {
            "image_url": candidate.image_url,
            "source_url": candidate.source_url,
            "title": candidate.title,
            "snippet": candidate.snippet,
            "mime": candidate.mime,
            "page_title": candidate.page_title,
            "page_h1": candidate.page_h1,
            "page_meta_description": candidate.page_meta_description,
            "image_alt": candidate.image_alt,
            "nearby_text": candidate.nearby_text,
            "source_domain": candidate.source_domain,
            "source_type": candidate.source_type,
            "width": candidate.width,
            "height": candidate.height,
            "size_bytes": candidate.size_bytes,
            "is_valid_image": candidate.is_valid_image,
            "invalid_reason": candidate.invalid_reason,
            "pre_rank_score": candidate.pre_rank_score,
            "merged_count": candidate.merged_count,
            "supporting_source_domains": candidate.supporting_source_domains,
            "supporting_source_types": candidate.supporting_source_types,
            "cluster_evidence_summary": candidate.cluster_evidence_summary,
        }

    def _ranked_candidate_summary(self, candidate: SearchCandidate) -> dict[str, object]:
        return {
            "image_url": candidate.image_url,
            "source_url": candidate.source_url,
            "pre_rank_score": candidate.pre_rank_score,
            "is_valid_image": candidate.is_valid_image,
            "invalid_reason": candidate.invalid_reason,
            "width": candidate.width,
            "height": candidate.height,
            "mime": candidate.mime,
        }
