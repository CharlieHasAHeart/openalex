from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord

logger = logging.getLogger(__name__)
_QWEN_CALL_LOCK = threading.Lock()
_QWEN_LAST_CALL_TS = 0.0


@dataclass(slots=True)
class QwenSearchResult:
    profile_pages: list[dict[str, Any]]
    image_candidates: list[dict[str, Any]]
    filtered_candidates: list[dict[str, Any]]
    failure_reason: str | None = None
    raw_content: str | None = None
    response_text: str | None = None
    response_format_mode: str | None = None
    abandon_reason_log: str | None = None
    invalid_profile_page_count: int = 0
    invalid_image_candidate_count: int = 0
    invalid_filtered_candidate_count: int = 0
    schema_issue_count: int = 0


@dataclass(slots=True)
class QwenAvatarSelectionResult:
    selected_index: int
    confidence: float
    reason: str
    failure_reason: str | None = None
    raw_content: str | None = None
    response_text: str | None = None


class QwenToolsClient:
    def __init__(
        self,
        http: HttpClient,
        api_key: str | None,
        base_url: str,
        model: str,
        timeout_seconds: int,
        min_call_interval_seconds: float,
        enable_web_search: bool,
        min_confidence: float,
        response_path: str = "/responses",
    ) -> None:
        self._http = http
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout_seconds = max(5, timeout_seconds)
        self._min_call_interval_seconds = max(0.0, float(min_call_interval_seconds))
        self._enable_web_search = enable_web_search
        self._min_confidence = max(0.0, min(1.0, min_confidence))
        self._response_path = response_path if response_path.startswith("/") else f"/{response_path}"
        self._client = (
            OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                max_retries=2,
            )
            if self._api_key
            else None
        )

    @property
    def model(self) -> str:
        return self._model

    def _build_response_format(self) -> dict[str, Any]:
        return {"type": "json_object"}

    def _response_url(self) -> str:
        return f"{self._base_url}{self._response_path}"

    def _author_ctx(self, author: AuthorRecord) -> dict[str, Any]:
        return {
            "author_id": author.author_id,
            "display_name": author.display_name,
            "orcid": author.orcid,
            "institution_name": author.institution_name,
            "institution_names": author.institution_names or [],
            "institution_country_codes": author.institution_country_codes or [],
        }

    def _build_prompt(self, author: AuthorRecord) -> str:
        return (
            "You are a scholar avatar evidence finder.\n"
            "Use web_search only.\n"
            "Goal: return the webpages most relevant for finding this author's official portrait later.\n"
            "Do not search for images and do not invent image URLs.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Keep only high-confidence author-related webpages such as ORCID, institutional faculty/staff/profile pages, institutional news pages about the author, and authoritative researcher databases.\n"
            "Reject social media, generic aggregators, and weakly related search results.\n"
            "Do not reject institutional news pages by default; keep them when the page is clearly about this author.\n"
            "Prefer pages whose institution, affiliation, ORCID, and researcher identity match the provided author context.\n"
            "If you see a same-name person with conflicting institution evidence, lower confidence or exclude that page.\n"
            'Schema: {"profile_pages":[{"url":"","title":"","snippet":"","source_type":"","confidence":0.0,"reason":""}],"image_candidates":[],"filtered_candidates":[],"failure_reason":""}\n'
            "Use arrays, not null. Additional keys are forbidden.\n"
            f"Minimum confidence threshold reference: {self._min_confidence:.2f}.\n"
            f"author={json.dumps(self._author_ctx(author), ensure_ascii=False)}"
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
        decoder = json.JSONDecoder()
        start = stripped.find("{")
        if start < 0:
            return None
        try:
            obj, end = decoder.raw_decode(stripped[start:])
        except Exception:
            match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if not match:
                return None
            try:
                obj = json.loads(match.group(0))
            except Exception:
                return None
            return obj if isinstance(obj, dict) else None
        trailing = stripped[start + end :].strip()
        if trailing:
            logger.warning("qwen_output_has_trailing_text trailing=%r", trailing[:160])
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

    def _parse_avatar_selection(self, obj: dict[str, Any]) -> QwenAvatarSelectionResult:
        try:
            selected_index = int(obj.get("selected_index"))
        except Exception:
            selected_index = -1
        try:
            confidence = float(obj.get("confidence"))
        except Exception:
            confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        reason = str(obj.get("reason") or "").strip()
        return QwenAvatarSelectionResult(
            selected_index=selected_index,
            confidence=confidence,
            reason=reason,
        )

    def _validate_profile_page(self, row: object) -> tuple[dict[str, Any] | None, bool]:
        if not isinstance(row, dict):
            return None, False
        url = str(row.get("url") or "").strip()
        source_type = str(row.get("source_type") or "generic_search_result").strip()
        try:
            confidence = float(row.get("confidence"))
        except Exception:
            return None, False
        if not url.startswith("http") or not (0.0 <= confidence <= 1.0):
            return None, False
        return {
            "url": url,
            "title": str(row.get("title") or "").strip(),
            "snippet": str(row.get("snippet") or "").strip(),
            "source_type": source_type or "generic_search_result",
            "confidence": confidence,
            "reason": str(row.get("reason") or "").strip(),
        }, True

    def _normalize_response_payload(self, obj: dict[str, Any]) -> QwenSearchResult:
        schema_issue_count = 0
        profile_rows = obj.get("profile_pages")
        image_rows = obj.get("image_candidates")
        filtered_rows = obj.get("filtered_candidates")
        if not isinstance(profile_rows, list):
            profile_rows = []
            schema_issue_count += 1
        if not isinstance(image_rows, list):
            image_rows = []
            schema_issue_count += 1
        if not isinstance(filtered_rows, list):
            filtered_rows = []
            schema_issue_count += 1

        profile_pages: list[dict[str, Any]] = []
        invalid_profile = 0
        for row in profile_rows:
            normalized, ok = self._validate_profile_page(row)
            if ok and normalized is not None:
                profile_pages.append(normalized)
            else:
                invalid_profile += 1

        failure_reason = str(obj.get("failure_reason") or "").strip() or None
        allowed_keys = {"profile_pages", "image_candidates", "filtered_candidates", "failure_reason"}
        extra_keys = sorted(set(obj.keys()) - allowed_keys)
        if extra_keys:
            schema_issue_count += len(extra_keys)
        if schema_issue_count > 0 and failure_reason is None:
            failure_reason = "qwen_schema_invalid"
        if not profile_pages and failure_reason is None:
            failure_reason = "qwen_no_profile_pages"

        return QwenSearchResult(
            profile_pages=profile_pages,
            image_candidates=[],
            filtered_candidates=[],
            failure_reason=failure_reason,
            invalid_profile_page_count=invalid_profile,
            invalid_image_candidate_count=0,
            invalid_filtered_candidate_count=0,
            schema_issue_count=schema_issue_count,
        )

    def _post_responses(
        self,
        payload: dict[str, Any],
        *,
        timeout_seconds: int | None = None,  # compatibility; handled by SDK client timeout
        max_retries: int | None = None,  # compatibility; handled by SDK client max_retries
    ) -> tuple[dict[str, Any], str]:
        del timeout_seconds, max_retries
        if self._client is None:
            raise RuntimeError("qwen client not initialized")
        if self._min_call_interval_seconds > 0:
            global _QWEN_LAST_CALL_TS
            with _QWEN_CALL_LOCK:
                now = time.monotonic()
                delta = now - _QWEN_LAST_CALL_TS
                if delta < self._min_call_interval_seconds:
                    sleep_seconds = self._min_call_interval_seconds - delta
                    logger.debug("qwen_call_sleep seconds=%.3f", sleep_seconds)
                    time.sleep(sleep_seconds)
                _QWEN_LAST_CALL_TS = time.monotonic()
        model = payload.get("model") or self._model
        input_payload = payload.get("input") or []
        extra_body: dict[str, Any] = {}
        for key in ("temperature", "tools", "response_format", "stream"):
            if key in payload:
                extra_body[key] = payload[key]
        response = self._client.responses.create(
            model=model,
            input=input_payload,
            extra_body=extra_body if extra_body else None,
        )
        if hasattr(response, "model_dump"):
            data = response.model_dump()
        elif hasattr(response, "to_dict"):
            data = response.to_dict()
        else:
            data = json.loads(json.dumps(response))
        return data if isinstance(data, dict) else {}, "json_object"

    def _exception_response_text(self, exc: Exception) -> str | None:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        text = getattr(response, "text", None)
        if text is None:
            return None
        if isinstance(text, bytes):
            try:
                text = text.decode("utf-8", errors="ignore")
            except Exception:
                text = str(text)
        return str(text)[:4000]

    def _exception_status_code(self, exc: Exception) -> int | None:
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        response = getattr(exc, "response", None)
        resp_status = getattr(response, "status_code", None)
        return resp_status if isinstance(resp_status, int) else None

    def search_author(self, author: AuthorRecord) -> QwenSearchResult:
        if not self._api_key:
            return QwenSearchResult([], [], [], failure_reason="qwen_api_key_missing")
        payload: dict[str, Any] = {
            "model": self._model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": self._build_prompt(author)}]}],
            "temperature": 0,
            "tools": ([{"type": "web_search"}] if self._enable_web_search else []),
            "response_format": self._build_response_format(),
        }
        response_format_mode = "json_object"
        logger.info("qwen_web_search_started author_id=%s model=%s", author.author_id, self._model)
        try:
            data, _ = self._post_responses(payload)
        except APIStatusError as exc:
            response_text = self._exception_response_text(exc)
            return QwenSearchResult([], [], [], failure_reason="qwen_http_error", raw_content=response_text, response_text=response_text, response_format_mode=response_format_mode)
        except APITimeoutError as exc:
            return QwenSearchResult([], [], [], failure_reason="qwen_request_timeout", raw_content=str(exc)[:4000], response_text=str(exc)[:4000], response_format_mode=response_format_mode)
        except APIConnectionError as exc:
            return QwenSearchResult([], [], [], failure_reason="qwen_request_failed", raw_content=str(exc)[:4000], response_text=str(exc)[:4000], response_format_mode=response_format_mode)
        except Exception as exc:
            return QwenSearchResult([], [], [], failure_reason="qwen_request_failed", raw_content=str(exc)[:4000], response_text=str(exc)[:4000], response_format_mode=response_format_mode)

        response_text = self._extract_response_text(data if isinstance(data, dict) else {})
        if not response_text:
            return QwenSearchResult([], [], [], failure_reason="qwen_output_missing", raw_content=json.dumps(data, ensure_ascii=False)[:4000], response_format_mode=response_format_mode)
        obj = self._extract_json_object(response_text)
        if obj is None:
            return QwenSearchResult([], [], [], failure_reason="qwen_output_not_json", raw_content=response_text[:4000], response_text=response_text[:4000], response_format_mode=response_format_mode)
        normalized = self._normalize_response_payload(obj)
        normalized.raw_content = response_text[:4000]
        normalized.response_text = response_text[:4000]
        normalized.response_format_mode = response_format_mode
        logger.info("qwen_web_search_finished author_id=%s profile_pages=%s failure_reason=%s", author.author_id, len(normalized.profile_pages), normalized.failure_reason)
        return normalized

    def select_avatar_candidate(
        self,
        author: AuthorRecord,
        candidates: list[dict[str, Any]],
        profile_pages: list[dict[str, Any]],
    ) -> QwenAvatarSelectionResult:
        if not self._api_key:
            return QwenAvatarSelectionResult(
                selected_index=-1,
                confidence=0.0,
                reason="missing qwen api key",
                failure_reason="qwen_api_key_missing",
            )
        normalized_candidates: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            image_url = str(candidate.get("image_url") or "").strip()
            if not image_url.startswith("http"):
                continue
            normalized_candidates.append(
                {
                    "index": index,
                    "image_url": image_url,
                    "source_url": str(candidate.get("source_url") or "").strip(),
                    "title": str(candidate.get("title") or "").strip(),
                    "snippet": str(candidate.get("snippet") or "").strip(),
                    "image_alt": str(candidate.get("image_alt") or "").strip(),
                    "nearby_text": str(candidate.get("nearby_text") or "").strip(),
                    "width": candidate.get("width"),
                    "height": candidate.get("height"),
                    "mime": candidate.get("mime"),
                    "pre_rank_score": candidate.get("pre_rank_score"),
                }
            )
        if not normalized_candidates:
            return QwenAvatarSelectionResult(
                selected_index=-1,
                confidence=0.0,
                reason="no valid candidate urls",
                failure_reason="qwen_avatar_select_no_candidates",
            )

        prompt = (
            "You are a strict avatar selector.\n"
            "Select exactly one best image candidate that is an avatar/person photo.\n"
            "If none are person photos, return selected_index=-1.\n"
            "Reason must be concise (<= 20 words).\n"
            "Return exactly one JSON object and nothing else.\n"
            'Schema: {"selected_index":0,"confidence":0.0,"reason":"","failure_reason":""}\n'
            f"author={json.dumps(self._author_ctx(author), ensure_ascii=False)}\n"
            f"profile_pages={json.dumps(profile_pages[:5], ensure_ascii=False)}\n"
            f"candidates={json.dumps(normalized_candidates, ensure_ascii=False)}"
        )
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for candidate in normalized_candidates[:6]:
            content.append({"type": "input_image", "image_url": candidate["image_url"]})
        payload: dict[str, Any] = {
            "model": self._model,
            "input": [
                {
                    "role": "user",
                    "content": content,
                }
            ],
            "temperature": 0,
            "response_format": self._build_response_format(),
        }
        logger.info("qwen_avatar_select_started author_id=%s candidates=%s", author.author_id, len(normalized_candidates))
        try:
            data, _ = self._post_responses(payload)
        except APIStatusError as exc:
            status_code = self._exception_status_code(exc)
            response_text = self._exception_response_text(exc)
            if status_code in {400, 404, 422}:
                fallback_payload = dict(payload)
                fallback_payload["input"] = [
                    {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                ]
                try:
                    data, _ = self._post_responses(fallback_payload)
                except Exception as fallback_exc:
                    return QwenAvatarSelectionResult(
                        selected_index=-1,
                        confidence=0.0,
                        reason=str(fallback_exc),
                        failure_reason="qwen_avatar_select_request_failed",
                        raw_content=response_text,
                        response_text=response_text,
                    )
            else:
                return QwenAvatarSelectionResult(
                    selected_index=-1,
                    confidence=0.0,
                    reason=str(exc),
                    failure_reason="qwen_avatar_select_http_error",
                    raw_content=response_text,
                    response_text=response_text,
                )
        except APITimeoutError as exc:
            return QwenAvatarSelectionResult(
                selected_index=-1,
                confidence=0.0,
                reason=str(exc),
                failure_reason="qwen_avatar_select_timeout",
                raw_content=str(exc)[:4000],
                response_text=str(exc)[:4000],
            )
        except APIConnectionError as exc:
            return QwenAvatarSelectionResult(
                selected_index=-1,
                confidence=0.0,
                reason=str(exc),
                failure_reason="qwen_avatar_select_request_failed",
                raw_content=str(exc)[:4000],
                response_text=str(exc)[:4000],
            )
        except Exception as exc:
            return QwenAvatarSelectionResult(
                selected_index=-1,
                confidence=0.0,
                reason=str(exc),
                failure_reason="qwen_avatar_select_request_failed",
                raw_content=str(exc)[:4000],
                response_text=str(exc)[:4000],
            )

        response_text = self._extract_response_text(data if isinstance(data, dict) else {})
        if not response_text:
            return QwenAvatarSelectionResult(
                selected_index=-1,
                confidence=0.0,
                reason="missing model output",
                failure_reason="qwen_avatar_select_output_missing",
                raw_content=json.dumps(data, ensure_ascii=False)[:4000],
            )
        obj = self._extract_json_object(response_text)
        if obj is None:
            return QwenAvatarSelectionResult(
                selected_index=-1,
                confidence=0.0,
                reason="model output is not valid json object",
                failure_reason="qwen_avatar_select_output_not_json",
                raw_content=response_text[:4000],
                response_text=response_text[:4000],
            )
        selection = self._parse_avatar_selection(obj)
        selection.raw_content = response_text[:4000]
        selection.response_text = response_text[:4000]
        logger.info(
            "qwen_avatar_select_finished author_id=%s selected_index=%s confidence=%.3f",
            author.author_id,
            selection.selected_index,
            selection.confidence,
        )
        return selection
