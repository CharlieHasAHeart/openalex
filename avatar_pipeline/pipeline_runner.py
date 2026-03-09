from __future__ import annotations

import logging
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from avatar_pipeline.avatar_gate import validate_image_candidate
from avatar_pipeline.config import PipelineConfig
from avatar_pipeline.llm_matcher import LlmMatcher
from avatar_pipeline.models import AuthorRecord, FinalDecisionAssessment, ImageCandidate, PipelineResult
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
        run_id: str | None = None,
    ) -> None:
        self._config = config
        self._web_search = web_search_client
        self._matcher = llm_matcher
        self._oss = oss_uploader
        self._repo = pg_repository
        self._run_id = run_id
        self._stats: Counter[str] = Counter()
        self._context_enrich_limit = 5
        self._image_enrich_limit = 5
        self._llm_top_k = 5

    @property
    def stats(self) -> Counter[str]:
        return self._stats

    def run_for_author_seed(self, author: AuthorRecord) -> str:
        try:
            existing_avatar = self._repo.get_author_avatar_record(author.author_id)
            existing_state = self._repo.get_avatar_state(author.author_id)
            if existing_state and self._should_skip_by_db_state(existing_state):
                status = existing_state.get("status") or "unknown"
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

        run_persist_error: str | None = None
        if self._run_id:
            try:
                self._repo.insert_author_run(
                    run_id=self._run_id,
                    author_id=int(author.author_id),
                    status=result.status,
                    error_code=None,
                    error_message=result.error_message,
                    selected_candidate_id=result.selected_candidate_id,
                    llm_score=result.llm_confidence,
                    final_score=result.llm_confidence,
                    finished_at=datetime.now(timezone.utc),
                )
            except Exception as exc:
                run_persist_error = str(exc)
                logger.error(
                    "pipeline_author_run_persist_failed run_id=%s author_id=%s status=%s error_message=%s",
                    self._run_id,
                    author.author_id,
                    result.status,
                    run_persist_error,
                )

        persist_error: str | None = None
        if result.status == "ok":
            try:
                self._repo.upsert_result(result)
            except Exception as exc:
                persist_error = str(exc)
                logger.error(
                    "pipeline_persist_failed run_id=%s author_id=%s orcid=%s status_before=%s error_message=%s",
                    self._run_id,
                    author.author_id,
                    author.orcid,
                    result.status,
                    persist_error,
                )

        self._stats[result.status] += 1
        logger.info(
            "pipeline_result run_id=%s author_id=%s orcid=%s status=%s qid=%s commons_file=%s selected_candidate_id=%s llm_confidence=%s decision_mode=%s acceptance_score=%s fallback_used=%s error_message=%s persist_error=%s run_persist_error=%s",
            self._run_id,
            author.author_id,
            author.orcid,
            result.status,
            result.wikidata_qid,
            result.commons_file,
            result.selected_candidate_id,
            result.llm_confidence,
            result.decision_mode,
            result.acceptance_score,
            result.fallback_used,
            result.error_message,
            persist_error,
            run_persist_error,
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

        try:
            candidates = self._web_search.enrich_candidates_context(candidates, limit=self._context_enrich_limit)
        except Exception as exc:
            logger.error(
                "pipeline_candidate_enrich_failed run_id=%s author_id=%s error_message=%s",
                self._run_id,
                author.author_id,
                str(exc),
            )

        try:
            candidates = self._web_search.enrich_candidates_image_metadata(
                candidates,
                limit=self._image_enrich_limit,
                allowed_mime=self._config.allowed_mime,
                min_edge_px=self._config.min_image_edge_px,
            )
        except Exception as exc:
            logger.error(
                "pipeline_candidate_image_enrich_failed run_id=%s author_id=%s error_message=%s",
                self._run_id,
                author.author_id,
                str(exc),
            )

        try:
            clustered_candidates = self._web_search.cluster_candidates(candidates, fingerprint_top_n=8)
            if clustered_candidates:
                candidates = clustered_candidates
        except Exception as exc:
            logger.error(
                "pipeline_candidate_cluster_failed run_id=%s author_id=%s error_message=%s",
                self._run_id,
                author.author_id,
                str(exc),
            )

        candidate_id_by_index: dict[int, int] = {}
        if self._run_id:
            for idx, raw_candidate in enumerate(candidates):
                source_domain: str | None = None
                try:
                    source_domain = raw_candidate.source_domain or urlparse(raw_candidate.source_url).hostname
                except Exception:
                    source_domain = None
                try:
                    candidate_id = self._repo.insert_candidate_image(
                        run_id=self._run_id,
                        author_id=int(author.author_id),
                        source_page_url=raw_candidate.source_url,
                        source_image_url=raw_candidate.image_url,
                        source_domain=source_domain,
                        page_title=raw_candidate.page_title or raw_candidate.title,
                        snippet=raw_candidate.snippet,
                        nearby_text=raw_candidate.nearby_text,
                        image_alt=raw_candidate.image_alt,
                        mime_type=raw_candidate.mime,
                        width=raw_candidate.width,
                        height=raw_candidate.height,
                        size_bytes=raw_candidate.size_bytes,
                        face_count=raw_candidate.face_count,
                        is_portrait=raw_candidate.is_portrait,
                        is_valid_image=raw_candidate.is_valid_image,
                        invalid_reason=raw_candidate.invalid_reason,
                    )
                    candidate_id_by_index[idx] = candidate_id
                except Exception as exc:
                    logger.error(
                        "pipeline_candidate_persist_failed run_id=%s author_id=%s candidate_index=%s source_url=%s error_message=%s",
                        self._run_id,
                        author.author_id,
                        idx,
                        raw_candidate.source_url,
                        str(exc),
                    )

        ranked_with_index = self._rank_candidates(author, candidates)
        top_ranked = ranked_with_index[: max(1, min(self._llm_top_k, len(ranked_with_index)))]
        top_index_to_original = [idx for idx, _ in top_ranked]
        llm_candidates = [candidate for _, candidate in top_ranked]

        decision = self._matcher.choose_best(author, llm_candidates)
        if not decision:
            top = llm_candidates[0] if llm_candidates else None
            reason_hint = ""
            if top is not None:
                reason_hint = f":top_source_type={top.source_type},top_score={top.pre_rank_score}"
            return PipelineResult(
                author_id=author.author_id,
                status="ambiguous",
                error_message=f"llm_no_confident_match{reason_hint}",
                decision_mode="ambiguous",
                fallback_used=False,
            )
        if decision.selected_index < 0 or decision.selected_index >= len(llm_candidates):
            return PipelineResult(
                author_id=author.author_id,
                status="ambiguous",
                error_message="llm_invalid_selected_index",
                decision_mode="ambiguous",
                fallback_used=False,
            )

        assessment: FinalDecisionAssessment
        try:
            assessment = self._assess_final_decision(author, llm_candidates, decision)
        except Exception as exc:
            logger.error(
                "pipeline_decision_assessment_failed run_id=%s author_id=%s error_message=%s",
                self._run_id,
                author.author_id,
                str(exc),
            )
            # Fallback to direct LLM decision.
            selected = decision.selected_index
            top_other = max(
                [float(c.pre_rank_score or 0.0) for i, c in enumerate(llm_candidates) if i != selected] or [-999.0]
            )
            assessment = FinalDecisionAssessment(
                accept=True,
                fallback_used=False,
                selected_index=selected,
                acceptance_score=float(decision.confidence),
                score_margin=float(llm_candidates[selected].pre_rank_score or 0.0) - top_other,
                decision_reason="assessment_failed_fallback_to_direct_llm",
                decision_mode="direct_accept",
            )

        if not assessment.accept or assessment.selected_index is None:
            return PipelineResult(
                author_id=author.author_id,
                status="ambiguous",
                error_message=f"final_gating_ambiguous:{assessment.decision_reason}",
                llm_confidence=decision.confidence,
                decision_reason=assessment.decision_reason,
                decision_mode="ambiguous",
                acceptance_score=assessment.acceptance_score,
                fallback_used=assessment.fallback_used,
            )

        selected_llm_index = assessment.selected_index
        original_selected_index = top_index_to_original[selected_llm_index]
        chosen = candidates[original_selected_index]
        selected_candidate_id = candidate_id_by_index.get(original_selected_index)
        if self._run_id and selected_candidate_id is not None:
            try:
                self._repo.insert_candidate_decision(
                    candidate_id=selected_candidate_id,
                    run_id=self._run_id,
                    author_id=int(author.author_id),
                    decision="match",
                    llm_score=decision.confidence,
                    final_score=decision.confidence,
                    decision_reason=decision.reason,
                    evidence={
                        "selected_index": selected_llm_index,
                        "original_index": original_selected_index,
                        "acceptance_score": assessment.acceptance_score,
                        "score_margin": assessment.score_margin,
                        "decision_mode": assessment.decision_mode,
                        "fallback_used": assessment.fallback_used,
                        "acceptance_reason": assessment.decision_reason,
                        "top_candidate_count_considered": len(llm_candidates),
                        "fallback_from_index": decision.selected_index if assessment.fallback_used else None,
                        "fallback_to_index": selected_llm_index if assessment.fallback_used else None,
                        "fallback_reason": assessment.decision_reason if assessment.fallback_used else None,
                        "source_type": chosen.source_type,
                        "discovery_score": chosen.discovery_score,
                        "discovery_evidence": chosen.discovery_evidence,
                        "merged_count": chosen.merged_count,
                        "supporting_source_types": chosen.supporting_source_types,
                        "supporting_source_domains": chosen.supporting_source_domains,
                        "content_deduped": chosen.content_deduped,
                        "cluster_evidence_summary": chosen.cluster_evidence_summary,
                        "page_image_role": chosen.page_image_role,
                        "page_image_position_score": chosen.page_image_position_score,
                        "name_proximity_score": chosen.name_proximity_score,
                        "context_block_type": chosen.context_block_type,
                        "structure_evidence": chosen.structure_evidence,
                        "source_url": chosen.source_url,
                        "image_url": chosen.image_url,
                        "source_domain": chosen.source_domain,
                        "page_title": chosen.page_title,
                        "page_h1": chosen.page_h1,
                        "image_alt": chosen.image_alt,
                        "nearby_text": chosen.nearby_text,
                        "pre_rank_score": chosen.pre_rank_score,
                        "name_match_score": chosen.name_match_score,
                        "institution_match_score": chosen.institution_match_score,
                        "source_trust_score": chosen.source_trust_score,
                        "image_precheck_score": chosen.image_precheck_score,
                        "mime_type": chosen.mime,
                        "width": chosen.width,
                        "height": chosen.height,
                        "size_bytes": chosen.size_bytes,
                        "face_count": chosen.face_count,
                        "is_portrait": chosen.is_portrait,
                        "is_valid_image": chosen.is_valid_image,
                        "invalid_reason": chosen.invalid_reason,
                    },
                )
            except Exception as exc:
                logger.error(
                    "pipeline_decision_persist_failed run_id=%s author_id=%s candidate_id=%s error_message=%s",
                    self._run_id,
                    author.author_id,
                    selected_candidate_id,
                    str(exc),
                )
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
                selected_candidate_id=selected_candidate_id,
                llm_confidence=decision.confidence,
                decision_reason=assessment.decision_reason,
                decision_mode=assessment.decision_mode,
                acceptance_score=assessment.acceptance_score,
                fallback_used=assessment.fallback_used,
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
                selected_candidate_id=selected_candidate_id,
                llm_confidence=decision.confidence,
                decision_reason=assessment.decision_reason,
                decision_mode=assessment.decision_mode,
                acceptance_score=assessment.acceptance_score,
                fallback_used=assessment.fallback_used,
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
            selected_candidate_id=selected_candidate_id,
            llm_confidence=decision.confidence,
            decision_reason=assessment.decision_reason,
            decision_mode=assessment.decision_mode,
            acceptance_score=assessment.acceptance_score,
            fallback_used=assessment.fallback_used,
        )

    def _source_type_reliability(self, source_type: str | None) -> float:
        st = (source_type or "generic_search_result").lower()
        return {
            "orcid_external_link": 1.0,
            "institution_profile": 0.95,
            "institution_directory": 0.8,
            "lab_people_page": 0.72,
            "orcid_profile": 0.7,
            "generic_search_result": 0.45,
        }.get(st, 0.45)

    def _structure_reliability(self, candidate) -> float:
        role = (candidate.page_image_role or "").lower()
        base = {
            "profile_headshot": 1.0,
            "card_headshot": 0.85,
            "generic_page_image": 0.35,
            "decorative_or_logo": 0.0,
        }.get(role, 0.4)
        base += min(float(candidate.name_proximity_score or 0.0), 1.0) * 0.35
        base += min(float(candidate.page_image_position_score or 0.0), 1.0) * 0.25
        return max(0.0, min(base, 1.2))

    def _acceptance_score(self, candidate, llm_confidence: float) -> float:
        pre_rank = float(candidate.pre_rank_score or 0.0)
        pre_rank_component = max(0.0, min((pre_rank + 2.0) / 10.0, 1.0))
        cluster_component = min(max(int(candidate.merged_count or 1) - 1, 0) * 0.08, 0.24)
        source_component = self._source_type_reliability(candidate.source_type) * 0.22
        structure_component = self._structure_reliability(candidate) * 0.22
        llm_component = max(0.0, min(llm_confidence, 1.0)) * 0.34
        penalty = 0.0
        if candidate.is_valid_image is False:
            penalty += 0.28
        if candidate.invalid_reason:
            penalty += 0.18
        score = (pre_rank_component * 0.22) + llm_component + source_component + structure_component + cluster_component - penalty
        return max(0.0, min(score, 1.0))

    def _assess_final_decision(self, author: AuthorRecord, llm_candidates: list, llm_decision) -> FinalDecisionAssessment:
        selected_index = llm_decision.selected_index
        if selected_index < 0 or selected_index >= len(llm_candidates):
            return FinalDecisionAssessment(
                accept=False,
                selected_index=None,
                acceptance_score=0.0,
                score_margin=None,
                decision_reason="llm_selected_index_out_of_range",
                decision_mode="ambiguous",
            )

        top = llm_candidates[selected_index]
        others = [(i, c) for i, c in enumerate(llm_candidates) if i != selected_index]
        best_other_idx, best_other = max(others, key=lambda x: float(x[1].pre_rank_score or -999.0)) if others else (None, None)
        top_score = float(top.pre_rank_score or 0.0)
        best_other_score = float(best_other.pre_rank_score or -999.0) if best_other is not None else -999.0
        score_margin = top_score - best_other_score if best_other is not None else 999.0
        acceptance_score = self._acceptance_score(top, float(llm_decision.confidence))

        # Conservative direct-accept gate.
        if (
            float(llm_decision.confidence) >= 0.68
            and acceptance_score >= 0.62
            and score_margin >= 0.35
            and top.is_valid_image is not False
        ):
            return FinalDecisionAssessment(
                accept=True,
                fallback_used=False,
                selected_index=selected_index,
                acceptance_score=acceptance_score,
                score_margin=score_margin,
                decision_reason="strong_top1_with_clear_margin",
                decision_mode="direct_accept",
            )

        # Early ambiguous for weak evidence.
        if float(llm_decision.confidence) < 0.55 or acceptance_score < 0.48:
            return FinalDecisionAssessment(
                accept=False,
                fallback_used=False,
                selected_index=None,
                acceptance_score=acceptance_score,
                score_margin=score_margin if best_other is not None else None,
                decision_reason="weak_confidence_or_acceptance",
                decision_mode="ambiguous",
            )

        # Controlled fallback: only when alternative is clearly cleaner.
        if best_other is not None and best_other_idx is not None:
            top_rel = self._source_type_reliability(top.source_type) + self._structure_reliability(top)
            alt_rel = self._source_type_reliability(best_other.source_type) + self._structure_reliability(best_other)
            alt_cleaner = best_other.is_valid_image is not False and top.is_valid_image is False
            alt_better_source = alt_rel > top_rel + 0.2
            close_scores = abs(top_score - best_other_score) <= 0.6
            if close_scores and (alt_cleaner or alt_better_source):
                alt_acceptance = self._acceptance_score(best_other, float(llm_decision.confidence) * 0.92)
                if alt_acceptance >= 0.58:
                    return FinalDecisionAssessment(
                        accept=True,
                        fallback_used=True,
                        selected_index=best_other_idx,
                        acceptance_score=alt_acceptance,
                        score_margin=score_margin,
                        decision_reason="fallback_to_cleaner_or_more_trustworthy_candidate",
                        decision_mode="fallback_accept",
                    )

        # Conservative final rule: if margin is too tight, choose ambiguous.
        if score_margin < 0.2 or acceptance_score < 0.58:
            return FinalDecisionAssessment(
                accept=False,
                fallback_used=False,
                selected_index=None,
                acceptance_score=acceptance_score,
                score_margin=score_margin if best_other is not None else None,
                decision_reason="insufficient_margin_or_acceptance",
                decision_mode="ambiguous",
            )

        return FinalDecisionAssessment(
            accept=True,
            fallback_used=False,
            selected_index=selected_index,
            acceptance_score=acceptance_score,
            score_margin=score_margin if best_other is not None else None,
            decision_reason="moderate_acceptance_passed_threshold",
            decision_mode="direct_accept",
        )

    def _score_candidate(self, author: AuthorRecord, candidate) -> tuple[float, float, float, float]:
        name_score = 0.0
        inst_score = 0.0
        trust_score = 0.0
        name = (author.display_name or "").strip().lower()
        inst = (author.institution_name or "").strip().lower()
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

        if name and name in text_blob:
            name_score += 2.5
        elif name:
            name_tokens = [t for t in re.split(r"\\s+", name) if t]
            if len(name_tokens) >= 2 and all(token in text_blob for token in name_tokens[:2]):
                name_score += 1.3
            elif any(token in text_blob for token in name_tokens):
                name_score += 0.8

        if inst and inst in text_blob:
            inst_score += 1.5
        elif inst:
            inst_tokens = [t for t in re.split(r"\\s+", inst) if len(t) > 3]
            if any(token in text_blob for token in inst_tokens):
                inst_score += 0.6

        if domain.endswith(".edu") or ".edu." in domain:
            trust_score += 1.4
        if domain.endswith("orcid.org"):
            trust_score += 1.8
        if any(key in domain for key in ("university", "institute", "ac.", "lab", "research")):
            trust_score += 0.8
        if any(key in domain for key in ("pinterest", "facebook", "instagram", "stock", "shutterstock")):
            trust_score -= 0.8
        if candidate.snippet and any(k in candidate.snippet.lower() for k in ("professor", "faculty", "researcher", "profile")):
            trust_score += 0.4

        source_type_score = 0.0
        source_type = (candidate.source_type or "generic_search_result").lower()
        if source_type == "orcid_external_link":
            source_type_score += 2.5
        elif source_type == "institution_profile":
            source_type_score += 2.2
        elif source_type == "institution_directory":
            source_type_score += 1.6
        elif source_type == "lab_people_page":
            source_type_score += 1.1
        elif source_type == "orcid_profile":
            source_type_score += 1.2
        else:
            source_type_score += 0.2
        trust_score += source_type_score

        discovery_score = float(candidate.discovery_score or 0.0)
        trust_score += min(discovery_score * 0.35, 1.8)
        cluster_bonus = 0.0
        merged_count = int(candidate.merged_count or 1)
        if merged_count > 1:
            cluster_bonus += min((merged_count - 1) * 0.25, 1.0)
        domains = candidate.supporting_source_domains or []
        source_types = candidate.supporting_source_types or []
        if len(domains) >= 2:
            cluster_bonus += 0.4
        if any(t in {"orcid_external_link", "institution_profile", "institution_directory", "lab_people_page"} for t in source_types):
            cluster_bonus += 0.3
        trust_score += cluster_bonus

        structure_score = 0.0
        role = (candidate.page_image_role or "").lower()
        if role == "profile_headshot":
            structure_score += 1.4
        elif role == "card_headshot":
            structure_score += 1.1
        elif role == "decorative_or_logo":
            structure_score -= 1.8
        else:
            structure_score += 0.1

        structure_score += min(float(candidate.page_image_position_score or 0.0), 1.0) * 0.9
        structure_score += min(float(candidate.name_proximity_score or 0.0), 1.0) * 1.2
        block_type = (candidate.context_block_type or "").lower()
        if block_type in {"people_card", "faculty_card", "profile_header", "name_matched_block"}:
            structure_score += 0.8
        elif block_type == "generic_page":
            structure_score += 0.1

        image_score = 0.0
        if candidate.is_valid_image is False:
            image_score -= 2.0
        elif candidate.is_valid_image is True:
            image_score += 0.8

        if candidate.invalid_reason and (
            candidate.invalid_reason.startswith("invalid_image_mime:")
            or candidate.invalid_reason in {"invalid_image_too_small", "image_metadata_fetch_failed", "image_dimension_unknown"}
        ):
            image_score -= 1.8
        if candidate.width and candidate.height:
            min_edge = min(candidate.width, candidate.height)
            if min_edge >= self._config.min_image_edge_px:
                image_score += 0.8
            else:
                image_score -= 1.2
            aspect = max(candidate.width / candidate.height, candidate.height / candidate.width)
            if aspect > 4.5:
                image_score -= 1.5
            elif aspect > 2.0:
                image_score -= 0.4
            else:
                image_score += 0.3
        if candidate.face_count is not None:
            if candidate.face_count == 1:
                image_score += 0.5
            elif candidate.face_count > 1:
                image_score -= 0.3
        if candidate.is_portrait is False:
            image_score -= 0.3

        candidate.image_precheck_score = image_score
        pre_rank = name_score + inst_score + trust_score + image_score + structure_score
        return name_score, inst_score, trust_score, pre_rank

    def _rank_candidates(self, author: AuthorRecord, candidates: list) -> list[tuple[int, object]]:
        ranked: list[tuple[int, object]] = []
        for idx, candidate in enumerate(candidates):
            try:
                name_score, inst_score, trust_score, pre_rank = self._score_candidate(author, candidate)
            except Exception as exc:
                logger.error(
                    "pipeline_candidate_score_failed run_id=%s author_id=%s candidate_index=%s error_message=%s",
                    self._run_id,
                    author.author_id,
                    idx,
                    str(exc),
                )
                name_score, inst_score, trust_score, pre_rank = 0.0, 0.0, 0.0, -1.0
            candidate.name_match_score = name_score
            candidate.institution_match_score = inst_score
            candidate.source_trust_score = trust_score
            candidate.pre_rank_score = pre_rank
            ranked.append((idx, candidate))
        ranked.sort(key=lambda x: x[1].pre_rank_score if x[1].pre_rank_score is not None else -999.0, reverse=True)
        return ranked
