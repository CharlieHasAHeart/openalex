from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

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
    ) -> None:
        self._http = http
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = max(5, timeout_seconds)
        self._enable_web_search = enable_web_search
        self._enable_t2i_search = enable_t2i_search
        self._min_confidence = max(0.0, min(1.0, min_confidence))

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
            "You are building author avatar candidates.\n"
            "Use this workflow:\n"
            "1) Use web_search to find likely official profile pages (faculty/staff/person/people/team/member/bio).\n"
            "2) Use t2i_search to find candidate portrait/headshot images.\n"
            "3) Cross-check image and profile evidence, keep only trustworthy candidate images.\n"
            "Exclude logo, banner, poster, collage/news montage, social avatars, group photos with unclear subject, non-human images.\n"
            "Return strict JSON with keys:\n"
            "{"
            '"profile_pages":[{"url":"","title":"","snippet":"","source_type":"","confidence":0..1,"reason":""}],'
            '"image_candidates":[{"image_url":"","source_url":"","title":"","snippet":"","source_type":"","image_alt":"","nearby_text":"","confidence":0..1,"reason":""}],'
            '"filtered_candidates":[same_schema_as_image_candidates],'
            '"failure_reason":"<optional>"'
            "}\n"
            f"Keep at most {max_candidates} items in filtered_candidates.\n"
            f"Minimum confidence threshold: {self._min_confidence:.2f}.\n"
            f"author={json.dumps(author_ctx, ensure_ascii=False)}"
        )

    def _extract_json(self, content: str) -> dict[str, Any] | None:
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
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
        return None

    def search_author(self, author: AuthorRecord, max_candidates: int) -> QwenSearchResult:
        if not self._api_key:
            return QwenSearchResult([], [], [], failure_reason="qwen_api_key_missing")

        prompt = self._build_prompt(author, max_candidates=max_candidates)
        payload: dict[str, Any] = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "enable_web_search": self._enable_web_search,
            "enable_t2i_search": self._enable_t2i_search,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = self._http.request(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=self._timeout_seconds,
            )
        except Exception as exc:
            logger.warning("qwen_tools_request_failed author_id=%s error_message=%s", author.author_id, str(exc))
            return QwenSearchResult([], [], [], failure_reason="qwen_web_search_failed")

        try:
            data = resp.json()
        except Exception:
            return QwenSearchResult([], [], [], failure_reason="qwen_parse_failed")
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        obj = self._extract_json(content)
        if obj is None:
            return QwenSearchResult([], [], [], failure_reason="qwen_parse_failed", raw_content=content)

        profile_pages = obj.get("profile_pages") if isinstance(obj.get("profile_pages"), list) else []
        image_candidates = obj.get("image_candidates") if isinstance(obj.get("image_candidates"), list) else []
        filtered_candidates = obj.get("filtered_candidates") if isinstance(obj.get("filtered_candidates"), list) else []
        failure_reason = obj.get("failure_reason") if isinstance(obj.get("failure_reason"), str) else None
        return QwenSearchResult(
            profile_pages=profile_pages,
            image_candidates=image_candidates,
            filtered_candidates=filtered_candidates,
            failure_reason=failure_reason,
            raw_content=content,
        )
