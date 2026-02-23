from __future__ import annotations

import logging
from collections import Counter
from datetime import date, datetime, timedelta, timezone

from avatar_pipeline.avatar_gate import validate_image_candidate
from avatar_pipeline.commons_client import CommonsClient
from avatar_pipeline.config import PipelineConfig
from avatar_pipeline.models import AuthorCandidate, AuthorRecord, PipelineResult
from avatar_pipeline.openalex_client import OpenAlexClient
from avatar_pipeline.oss_uploader import OssUploader, sha256_hex
from avatar_pipeline.pg_repository import PgRepository
from avatar_pipeline.wdqs_client import WdqsClient
from avatar_pipeline.wikidata_api_client import WikidataApiClient


logger = logging.getLogger(__name__)


class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        openalex_client: OpenAlexClient,
        wdqs_client: WdqsClient,
        wikidata_api_client: WikidataApiClient,
        commons_client: CommonsClient,
        oss_uploader: OssUploader,
        pg_repository: PgRepository,
    ) -> None:
        self._config = config
        self._openalex = openalex_client
        self._wdqs = wdqs_client
        self._wikidata = wikidata_api_client
        self._commons = commons_client
        self._oss = oss_uploader
        self._repo = pg_repository
        self._stats: Counter[str] = Counter()

    @property
    def stats(self) -> Counter[str]:
        return self._stats

    def run_from_top_works(
        self,
        top_n: int,
        top_offset: int,
        window_start_date: date,
        window_end_date: date,
        per_page: int,
    ) -> None:
        candidates: dict[str, AuthorCandidate] = {}
        works_seen = 0
        works_with_authorships = 0
        authors_extracted = 0

        for work in self._openalex.iter_top_works(
            top_n=top_n,
            top_offset=top_offset,
            from_publication_date=window_start_date,
            to_publication_date=window_end_date,
            per_page=per_page,
        ):
            works_seen += 1
            cited_by_count = int(work.get("cited_by_count") or 0)
            if cited_by_count <= 0:
                continue

            authorships = work.get("authorships") or []
            if not authorships:
                continue

            works_with_authorships += 1
            work_id = work.get("id", "")
            for authorship in authorships:
                author_data = (authorship or {}).get("author") or {}
                author_id = author_data.get("id")
                if not author_id:
                    continue
                authors_extracted += 1

                display_name = author_data.get("display_name", "")
                orcid = author_data.get("orcid") or (authorship or {}).get("orcid")
                author_record = AuthorRecord(
                    author_id=author_id,
                    display_name=display_name,
                    orcid_url=orcid,
                )

                if author_id not in candidates:
                    candidates[author_id] = AuthorCandidate(
                        author=author_record,
                        seed_work_id=work_id,
                        seed_work_cited_by_count=cited_by_count,
                    )
                    continue

                existing = candidates[author_id]
                existing.appearance_count += 1
                if cited_by_count > existing.seed_work_cited_by_count:
                    existing.seed_work_id = work_id
                    existing.seed_work_cited_by_count = cited_by_count
                    existing.author.display_name = author_record.display_name or existing.author.display_name
                    existing.author.orcid_url = author_record.orcid_url or existing.author.orcid_url

        ordered_candidates = sorted(
            candidates.values(),
            key=lambda item: (item.seed_work_cited_by_count, item.appearance_count),
            reverse=True,
        )

        self._stats["works_seen"] += works_seen
        self._stats["works_with_authorships"] += works_with_authorships
        self._stats["authors_extracted"] += authors_extracted
        self._stats["authors_deduped"] += len(ordered_candidates)

        for candidate in ordered_candidates:
            author_id = candidate.author.author_id
            avatar_state = self._repo.get_avatar_state(author_id)
            if avatar_state and self._should_skip_by_db_state(avatar_state):
                status = avatar_state.get("status") or "unknown"
                self._stats[f"skipped_recent_{status}"] += 1
                continue

            author = candidate.author
            if not author.orcid:
                try:
                    author = self._openalex.get_author(author.author_id)
                except Exception as exc:
                    result = PipelineResult(
                        author_id=author.author_id,
                        status="error",
                        error_message=f"fetch_author_failed:{exc}",
                    )
                    self._repo.upsert_result(result)
                    self._stats["error"] += 1
                    continue

            if not author.orcid:
                self._stats["skipped_no_orcid"] += 1
                continue

            self._stats["authors_processed"] += 1
            self.run_for_author(author)

        logger.info(
            (
                "job_summary window_start=%s window_end=%s top_n=%s per_page=%s "
                "top_offset=%s "
                "works_seen=%s works_with_authorships=%s authors_extracted=%s authors_deduped=%s"
            ),
            window_start_date.isoformat(),
            window_end_date.isoformat(),
            top_n,
            per_page,
            top_offset,
            works_seen,
            works_with_authorships,
            authors_extracted,
            len(ordered_candidates),
        )

    def run_for_author(self, author: AuthorRecord) -> PipelineResult:
        try:
            result = self._process(author)
        except Exception as exc:
            result = PipelineResult(
                author_id=author.author_id,
                status="error",
                error_message=str(exc),
            )

        self._repo.upsert_result(result)
        self._stats[result.status] += 1
        logger.info(
            "pipeline_result author_id=%s orcid=%s status=%s qid=%s commons_file=%s error_message=%s",
            author.author_id,
            author.orcid,
            result.status,
            result.wikidata_qid,
            result.commons_file,
            result.error_message,
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

    def _process(self, author: AuthorRecord) -> PipelineResult:
        if not author.orcid:
            return PipelineResult(
                author_id=author.author_id,
                status="no_match",
                error_message="missing_orcid",
            )

        qid, qid_error = self._wdqs.find_qid_by_orcid(author.orcid)
        if qid_error:
            return PipelineResult(
                author_id=author.author_id,
                status="ambiguous" if qid_error == "qid_not_unique" else "no_match",
                error_message=qid_error,
            )

        if not qid and self._config.allow_name_fallback:
            candidates = self._wikidata.search_entities(author.display_name)
            if len(candidates) == 1:
                qid = candidates[0].get("id")
            elif len(candidates) > 1:
                return PipelineResult(
                    author_id=author.author_id,
                    status="ambiguous",
                    error_message="name_fallback_multiple_candidates",
                )
            else:
                return PipelineResult(
                    author_id=author.author_id,
                    status="no_match",
                    error_message="name_fallback_no_match",
                )

        if not qid:
            return PipelineResult(
                author_id=author.author_id,
                status="no_match",
                error_message="no_qid",
            )

        commons_file = self._wdqs.get_p18_image_by_qid(qid)
        if not commons_file:
            return PipelineResult(
                author_id=author.author_id,
                status="no_image",
                wikidata_qid=qid,
                error_message="p18_not_found",
            )

        candidate = self._commons.get_image_candidate(commons_file, self._config.avatar_thumb_width)
        if not candidate:
            return PipelineResult(
                author_id=author.author_id,
                status="invalid_image",
                wikidata_qid=qid,
                commons_file=commons_file,
                error_message="commons_imageinfo_missing",
            )

        image_bytes = self._commons.download_image(candidate.download_url)
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
                wikidata_qid=qid,
                commons_file=commons_file,
                error_message=image_error,
            )

        sha256 = sha256_hex(image_bytes)
        existing = self._repo.get_existing_by_author_id(author.author_id)
        if existing and existing.get("content_sha256") == sha256 and existing.get("oss_object_key"):
            return PipelineResult(
                author_id=author.author_id,
                status="ok",
                wikidata_qid=qid,
                commons_file=commons_file,
                content_sha256=sha256,
                oss_object_key=existing["oss_object_key"],
                oss_url=existing.get("oss_url"),
            )

        object_key = self._oss.build_object_key(author.author_id, sha256, candidate.mime)
        oss_url = self._oss.upload(object_key, image_bytes, candidate.mime)
        return PipelineResult(
            author_id=author.author_id,
            status="ok",
            wikidata_qid=qid,
            commons_file=commons_file,
            content_sha256=sha256,
            oss_object_key=object_key,
            oss_url=oss_url,
        )
