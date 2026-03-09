from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from requests import HTTPError

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class QwenSearchResult:
    profile_pages: list[dict[str, Any]]
    image_candidates: list[dict[str, Any]]
    filtered_candidates: list[dict[str, Any]]
    failure_reason: str | None = None
    raw_content: str | None = None
    invalid_profile_page_count: int = 0
    invalid_image_candidate_count: int = 0
    invalid_filtered_candidate_count: int = 0
    schema_issue_count: int = 0


class QwenToolsClient:
    def __init__(
        self,
        http: HttpClient,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: int,
        enable_web_search: bool,
        enable_t2i_search: bool,
        min_confidence: float,
        response_path: str = "/responses",
    ) -> None:
        self._http = http
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = max(5, timeout_seconds)
        self._enable_web_search = enable_web_search
        self._enable_t2i_search = enable_t2i_search
        self._min_confidence = max(0.0, min(1.0, min_confidence))
        self._response_path = response_path if response_path.startswith("/") else f"/{response_path}"

    @property
    def model(self) -> str:
        return self._model

    def _build_prompt(self, author: AuthorRecord, max_candidates: int) -> str:
        author_ctx = {
            "author_id": author.author_id,
            "display_name": author.display_name,
            "orcid": author.orcid,
            "institution_name": author.institution_name,
            "concept_names": author.concept_names or [],
        }
        return (
            "You are an author-avatar candidate discovery engine.\n"
            "Workflow:\n"
            "1) Use web_search to find official profile pages (faculty/staff/person/people/team/member/bio).\n"
            "2) Use t2i_search to find portrait/headshot image candidates.\n"
            "3) Cross-verify image candidates against profile/source evidence.\n"
            "4) Return ONLY one JSON object (no markdown, no prose).\n"
            "Hard exclusions: logo, banner, poster, collage/news montage, social avatar, unclear group photo, non-human image.\n"
            "Confidence must be float in [0,1].\n"
            "Output schema:\n"
            "{"
            '"profile_pages":[{"url":"","title":"","snippet":"","source_type":"","confidence":0.0,"reason":""}],'
            '"image_candidates":[{"image_url":"","source_url":"","title":"","snippet":"","source_type":"","image_alt":"","nearby_text":"","confidence":0.0,"reason":""}],'
            '"filtered_candidates":[{"image_url":"","source_url":"","title":"","snippet":"","source_type":"","image_alt":"","nearby_text":"","confidence":0.0,"reason":""}],'
            '"failure_reason":""'
            "}\n"
            f"`filtered_candidates` is the final list for pipeline consumption (max {max_candidates}).\n"
            f"Minimum confidence threshold reference: {self._min_confidence:.2f}.\n"
            f"author={json.dumps(author_ctx, ensure_ascii=False)}"
        )

    def _extract_json_object(self, content: str) -> dict[str, Any] | None:
        stripped = content.strip()
        if not stripped:
            return None
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _extract_response_text(self, payload: dict[str, Any]) -> str | None:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        output = payload.get("output")
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text = part.get("text")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks)

        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            content = (((choices[0] or {}).get("message") or {}).get("content") or "")
            if isinstance(content, str) and content.strip():
                return content
        return None

    def _validate_profile_page(self, row: object) -> tuple[dict[str, Any] | None, bool]:
        if not isinstance(row, dict):
            return None, False
        url = str(row.get("url") or "").strip()
        source_type = str(row.get("source_type") or "generic_search_result").strip()
        confidence_raw = row.get("confidence")
        try:
            confidence = float(confidence_raw)
        except Exception:
            return None, False
        if not url.startswith("http"):
            return None, False
        if confidence < 0 or confidence > 1:
            return None, False
        normalized = {
            "url": url,
            "title": str(row.get("title") or "").strip(),
            "snippet": str(row.get("snippet") or "").strip(),
            "source_type": source_type or "generic_search_result",
            "confidence": confidence,
            "reason": str(row.get("reason") or "").strip(),
        }
        return normalized, True

    def _validate_candidate_item(self, row: object) -> tuple[dict[str, Any] | None, bool]:
        if not isinstance(row, dict):
            return None, False
        image_url = str(row.get("image_url") or "").strip()
        source_url = str(row.get("source_url") or "").strip()
        source_type = str(row.get("source_type") or "t2i_result").strip()
        confidence_raw = row.get("confidence")
        try:
            confidence = float(confidence_raw)
        except Exception:
            return None, False
        if not image_url.startswith("http") or not source_url.startswith("http"):
            return None, False
        if confidence < 0 or confidence > 1:
            return None, False
        normalized = {
            "image_url": image_url,
            "source_url": source_url,
            "title": str(row.get("title") or "").strip(),
            "snippet": str(row.get("snippet") or "").strip(),
            "source_type": source_type or "t2i_result",
            "image_alt": str(row.get("image_alt") or "").strip(),
            "nearby_text": str(row.get("nearby_text") or "").strip(),
            "confidence": confidence,
            "reason": str(row.get("reason") or "").strip(),
        }
        return normalized, True

    def _normalize_response_payload(self, obj: dict[str, Any]) -> QwenSearchResult:
        schema_issue_count = 0

        profile_rows = obj.get("profile_pages")
        if not isinstance(profile_rows, list):
            profile_rows = []
            schema_issue_count += 1

        image_rows = obj.get("image_candidates")
        if not isinstance(image_rows, list):
            image_rows = []
            schema_issue_count += 1

        filtered_rows = obj.get("filtered_candidates")
        if not isinstance(filtered_rows, list):
            filtered_rows = []
            schema_issue_count += 1

        profile_pages: list[dict[str, Any]] = []
        invalid_profile = 0
        for row in profile_rows:
            normalized, ok = self._validate_profile_page(row)
            if not ok or normalized is None:
                invalid_profile += 1
                continue
            profile_pages.append(normalized)

        image_candidates: list[dict[str, Any]] = []
        invalid_image = 0
        for row in image_rows:
            normalized, ok = self._validate_candidate_item(row)
            if not ok or normalized is None:
                invalid_image += 1
                continue
            image_candidates.append(normalized)

        filtered_candidates: list[dict[str, Any]] = []
        invalid_filtered = 0
        for row in filtered_rows:
            normalized, ok = self._validate_candidate_item(row)
            if not ok or normalized is None:
                invalid_filtered += 1
                continue
            filtered_candidates.append(normalized)

        failure_reason = str(obj.get("failure_reason") or "").strip() or None
        if not filtered_candidates and filtered_rows:
            failure_reason = failure_reason or "qwen_schema_invalid"
        if not filtered_candidates and not filtered_rows:
            failure_reason = failure_reason or "qwen_empty_filtered_candidates"
        if schema_issue_count > 0 and failure_reason is None:
            failure_reason = "qwen_schema_invalid"

        logger.info(
            "qwen_schema_validation_summary profile_pages=%s image_candidates=%s filtered_candidates=%s invalid_profile=%s invalid_image=%s invalid_filtered=%s schema_issues=%s failure_reason=%s",
            len(profile_pages),
            len(image_candidates),
            len(filtered_candidates),
            invalid_profile,
            invalid_image,
            invalid_filtered,
            schema_issue_count,
            failure_reason,
        )
        return QwenSearchResult(
            profile_pages=profile_pages,
            image_candidates=image_candidates,
            filtered_candidates=filtered_candidates,
            failure_reason=failure_reason,
            invalid_profile_page_count=invalid_profile,
            invalid_image_candidate_count=invalid_image,
            invalid_filtered_candidate_count=invalid_filtered,
            schema_issue_count=schema_issue_count,
        )

    def _response_url(self) -> str:
        return f"{self._base_url}{self._response_path}"

    def search_author(self, author: AuthorRecord, max_candidates: int) -> QwenSearchResult:
        if not self._api_key:
            return QwenSearchResult([], [], [], failure_reason="qwen_api_key_missing")

        prompt = self._build_prompt(author, max_candidates=max_candidates)
        tools: list[dict[str, Any]] = []
        if self._enable_web_search:
            tools.append({"type": "web_search"})
        if self._enable_t2i_search:
            tools.append({"type": "t2i_search"})
        payload: dict[str, Any] = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            ],
            "temperature": 0,
        }
        if tools:
            payload["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        logger.info(
            "qwen_request_started author_id=%s model=%s endpoint=%s web_search=%s t2i_search=%s",
            author.author_id,
            self._model,
            self._response_url(),
            self._enable_web_search,
            self._enable_t2i_search,
        )
        try:
            resp = self._http.request(
                "POST",
                self._response_url(),
                headers=headers,
                json=payload,
                timeout=self._timeout_seconds,
            )
        except HTTPError as exc:
            logger.warning("qwen_http_error author_id=%s error_message=%s", author.author_id, str(exc))
            return QwenSearchResult([], [], [], failure_reason="qwen_http_error")
        except Exception as exc:
            logger.warning("qwen_request_failed author_id=%s error_message=%s", author.author_id, str(exc))
            return QwenSearchResult([], [], [], failure_reason="qwen_request_failed")

        logger.info("qwen_response_received author_id=%s status_code=%s", author.author_id, resp.status_code)
        try:
            data = resp.json()
        except Exception:
            return QwenSearchResult([], [], [], failure_reason="qwen_response_decode_failed")

        response_text = self._extract_response_text(data if isinstance(data, dict) else {})
        if not response_text:
            return QwenSearchResult([], [], [], failure_reason="qwen_output_missing")

        obj = self._extract_json_object(response_text)
        if obj is None:
            return QwenSearchResult([], [], [], failure_reason="qwen_output_not_json", raw_content=response_text[:1200])

        normalized = self._normalize_response_payload(obj)
        normalized.raw_content = response_text[:1200]
        return normalized
