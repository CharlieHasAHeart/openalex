from __future__ import annotations

import logging
import re
from collections import Counter
from urllib.parse import urlparse

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
        ranked = self._rank_candidates(author, candidates)
        ranked_candidates = [self._ranked_candidate_summary(candidate) for candidate in ranked[:3]]
        logger.info(
            "pipeline_ranked_candidates run_id=%s author_id=%s ranked_candidates=%s",
            self._run_store.run_id,
            author.author_id,
            ranked_candidates,
        )
        if self._should_fail_small_avatar_fallback(ranked):
            failure_reason = "small_avatar_only_background_fallback"
            self._log_step(author, "select_best_candidate_rejected", failure_reason=failure_reason)
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
                abandon_reason_log=outcome.abandon_reason_log,
            )
        self._log_step(author, "select_best_candidate_start", ranked=len(ranked))
        selected = self._select_candidate(ranked)
        if selected is None:
            failure_reason = outcome.failure_reason or "no_ranked_candidates"
            self._log_step(author, "select_best_candidate_empty", failure_reason=failure_reason)
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
                abandon_reason_log=outcome.abandon_reason_log,
            )

        selected_dict = self._candidate_to_dict(selected)
        self._log_step(
            author,
            "select_best_candidate_done",
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

        self._log_step(author, "llm_avatar_review_start", image_url=selected.image_url)
        review = self._web_search.review_avatar_candidate(author, selected, outcome.profile_pages)
        selected_dict["llm_avatar_review"] = {
            "is_avatar": review.is_avatar,
            "confidence": review.confidence,
            "reason": review.reason,
            "risk_flags": review.risk_flags,
            "failure_reason": review.failure_reason,
        }
        if review.failure_reason:
            self._log_step(author, "llm_avatar_review_failed", failure_reason=review.failure_reason)
            return PipelineResult(
                author_id=author.author_id,
                status="no_image",
                error_message=review.failure_reason,
                failure_reason=review.failure_reason,
                commons_file=selected.source_url,
                ranked_candidates=ranked_candidates,
                selected_candidate=selected_dict,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=review.raw_content or outcome.raw_content,
                response_text=review.response_text or outcome.response_text,
                abandon_reason_log=outcome.abandon_reason_log,
            )
        self._log_step(
            author,
            "llm_avatar_review_done",
            is_avatar=review.is_avatar,
            confidence=f"{review.confidence:.3f}",
        )
        if not review.is_avatar:
            failure_reason = "llm_avatar_review_rejected"
            self._log_step(author, "llm_avatar_review_rejected", failure_reason=failure_reason)
            return PipelineResult(
                author_id=author.author_id,
                status="no_image",
                error_message=review.reason or failure_reason,
                failure_reason=failure_reason,
                commons_file=selected.source_url,
                ranked_candidates=ranked_candidates,
                selected_candidate=selected_dict,
                profile_pages=outcome.profile_pages,
                image_candidates=outcome.image_candidates,
                filtered_candidates=outcome.filtered_candidates,
                raw_content=review.raw_content or outcome.raw_content,
                response_text=review.response_text or outcome.response_text,
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

    def _select_candidate(self, ranked: list[SearchCandidate]) -> SearchCandidate | None:
        valid = [candidate for candidate in ranked if candidate.is_valid_image is True]
        if valid:
            return valid[0]
        return ranked[0] if ranked else None

    def _should_fail_small_avatar_fallback(self, ranked: list[SearchCandidate]) -> bool:
        if not ranked:
            return False
        top = ranked[0]
        if top.is_valid_image is not False or top.invalid_reason != "invalid_image_too_small":
            return False
        top_url = (top.image_url or "").lower()
        top_text = " ".join([top.image_url or "", top.image_alt or "", top.nearby_text or ""]).lower()
        if not any(token in (top_url + " " + top_text) for token in ("avatar", "portrait", "headshot", "profile photo", "profile_image", "basic_photo_1")):
            return False
        valid_candidates = [candidate for candidate in ranked[1:] if candidate.is_valid_image is True]
        if not valid_candidates:
            return False
        for candidate in valid_candidates:
            text = " ".join([candidate.image_url or "", candidate.image_alt or "", candidate.nearby_text or ""]).lower()
            if not any(token in text for token in ("cover", "cover_picture", "background", "bg-", "hero", "header", "jumbotron", "masthead", "banner")):
                return False
        return True

    def _score_candidate(self, author: AuthorRecord, candidate: SearchCandidate) -> float:
        name = (author.display_name or "").strip().lower()
        institution = (author.institution_name or "").strip().lower()
        domain = (candidate.source_domain or urlparse(candidate.source_url).hostname or "").lower()
        text_blob = " ".join(
            [
                candidate.title or "",
                candidate.snippet or "",
                candidate.page_title or "",
                candidate.page_h1 or "",
                candidate.page_meta_description or "",
                candidate.image_alt or "",
                candidate.nearby_text or "",
            ]
        ).lower()

        score = float(candidate.discovery_score or 0.0)
        if name and name in text_blob:
            score += 2.5
        elif name:
            name_tokens = [token for token in re.split(r"\s+", name) if token]
            if len(name_tokens) >= 2 and all(token in text_blob for token in name_tokens[:2]):
                score += 1.2
            elif any(token in text_blob for token in name_tokens):
                score += 0.6

        if institution and institution in text_blob:
            score += 1.2
        elif institution:
            institution_tokens = [token for token in re.split(r"\s+", institution) if len(token) > 3]
            if any(token in text_blob for token in institution_tokens):
                score += 0.4

        if domain.endswith(".edu") or ".edu." in domain:
            score += 1.0
        if "orcid.org" in domain:
            score += 0.8
        if any(token in domain for token in ("university", "institute", "research", "lab")):
            score += 0.5

        if candidate.is_valid_image is True:
            score += 1.0
        elif candidate.is_valid_image is False:
            score -= 2.0

        if candidate.width and candidate.height:
            min_edge = min(candidate.width, candidate.height)
            if min_edge >= self._config.min_image_edge_px:
                score += 0.6
            aspect_ratio = max(candidate.width / candidate.height, candidate.height / candidate.width)
            if aspect_ratio > 4.5:
                score -= 1.5
            if candidate.width > candidate.height:
                score -= min((candidate.width / max(candidate.height, 1)) - 1.0, 1.5)

        if candidate.merged_count and candidate.merged_count > 1:
            score += min((candidate.merged_count - 1) * 0.25, 1.0)

        image_blob = " ".join(
            [
                candidate.image_url or "",
                candidate.image_alt or "",
                candidate.nearby_text or "",
                candidate.title or "",
                candidate.snippet or "",
                candidate.page_title or "",
                candidate.page_h1 or "",
                candidate.page_meta_description or "",
            ]
        ).lower()
        if any(token in image_blob for token in ("syuugou", "group photo", "team photo", "basic_photo_2")):
            score -= 2.5
        if any(token in image_blob for token in ("cover", "cover_picture", "background", "bg-", "hero", "header", "jumbotron", "masthead", "loading", "spinner")):
            score -= 3.0
        if any(token in image_blob for token in ("cnrs", "inria", "inrae", "coretrustseal", "seal", "emblem", "logo")) and "/public/" in (candidate.image_url or "").lower():
            score -= 4.0
        if any(token in image_blob for token in ("avatar", "portrait", "headshot", "profile photo", "profile_image", "basic_photo_1")):
            score += 1.0
        image_url_lower = (candidate.image_url or "").lower()
        if domain == "researchmap.jp" and image_url_lower.endswith("/avatar.jpg"):
            score += 4.0
        if domain == "researchmap.jp" and "cover_picture" in image_url_lower:
            score -= 6.0

        candidate.pre_rank_score = score
        return score

    def _rank_candidates(self, author: AuthorRecord, candidates: list[SearchCandidate]) -> list[SearchCandidate]:
        ranked = list(candidates)
        ranked.sort(key=lambda candidate: self._score_candidate(author, candidate), reverse=True)
        return ranked
