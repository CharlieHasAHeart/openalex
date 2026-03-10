from __future__ import annotations

import json
import logging
import re
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord

logger = logging.getLogger(__name__)
_QWEN_CALL_LOCK = threading.Lock()
_QWEN_LAST_CALL_TS = 0.0

warnings.filterwarnings(
    "ignore",
    message=r".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"pydantic\.main",
)


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

    def _author_ctx(self, author: AuthorRecord) -> dict[str, Any]:
        return {
            "author_id": author.author_id,
            "display_name": author.display_name,
            "orcid": author.orcid,
            "affiliations": author.affiliations,
            "last_known_institutions": author.last_known_institutions,
        }

    def _build_prompt(self, author: AuthorRecord) -> str:
        return (
            "Use web_search_image to find candidate avatar/person-photo images for this exact author.\n"
            "Focus on author identity match using display_name, ORCID, affiliations, and last_known_institutions.\n"
            "Avoid logos, banners, icons, unrelated same-name people, and generic illustrations.\n"
            "Return exactly one JSON object and nothing else.\n"
            'Schema: {"profile_pages":[],"image_candidates":[{"image_url":"","source_url":"","reason":""}],"filtered_candidates":[],"failure_reason":""}\n'
            f"author={json.dumps(self._author_ctx(author), ensure_ascii=False)}"
        )

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
        return None

    def _extract_json_object(self, content: str) -> dict[str, Any] | None:
        stripped = (content or "").strip()
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

    def _collect_urls_from_object(self, obj: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        candidates = obj.get("image_candidates") if isinstance(obj, dict) else None
        if isinstance(candidates, list):
            for row in candidates:
                if not isinstance(row, dict):
                    continue
                image_url = str(row.get("image_url") or row.get("url") or "").strip()
                if not image_url.startswith("http"):
                    continue
                rows.append(
                    {
                        "image_url": image_url,
                        "source_url": str(row.get("source_url") or "").strip(),
                        "reason": str(row.get("reason") or "output_text_candidate").strip(),
                        "title": str(row.get("title") or "").strip(),
                    }
                )
        return rows

    def _collect_urls_from_tool_calls(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        output = payload.get("output")
        if not isinstance(output, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") != "web_search_image_call":
                continue
            raw = item.get("output")
            if not raw:
                continue
            try:
                result_rows = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                continue
            if not isinstance(result_rows, list):
                continue
            for row in result_rows:
                if not isinstance(row, dict):
                    continue
                image_url = str(row.get("url") or row.get("image_url") or "").strip()
                if not image_url.startswith("http"):
                    continue
                rows.append(
                    {
                        "image_url": image_url,
                        "source_url": str(row.get("source_url") or row.get("url") or "").strip(),
                        "reason": "web_search_image_tool_call",
                        "title": str(row.get("title") or "").strip(),
                    }
                )
        return rows

    def _dedupe_candidates(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            image_url = str(row.get("image_url") or "").strip()
            if not image_url or image_url in seen:
                continue
            seen.add(image_url)
            deduped.append(
                {
                    "image_url": image_url,
                    "source_url": str(row.get("source_url") or "").strip(),
                    "title": str(row.get("title") or "").strip(),
                    "snippet": "",
                    "source_type": "web_search_image",
                    "confidence": 1.0,
                    "reason": str(row.get("reason") or "web_search_image").strip(),
                }
            )
        return deduped

    def _post_responses(self, payload: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if self._client is None:
            raise RuntimeError("qwen client not initialized")
        if self._min_call_interval_seconds > 0:
            global _QWEN_LAST_CALL_TS
            with _QWEN_CALL_LOCK:
                now = time.monotonic()
                delta = now - _QWEN_LAST_CALL_TS
                if delta < self._min_call_interval_seconds:
                    time.sleep(self._min_call_interval_seconds - delta)
                _QWEN_LAST_CALL_TS = time.monotonic()

        response = self._client.responses.create(
            model=payload["model"],
            input=payload["input"],
            tools=payload.get("tools"),
            extra_body={
                "temperature": payload.get("temperature", 0),
                "response_format": payload.get("response_format"),
                "enable_thinking": False,
            },
        )
        if hasattr(response, "to_dict"):
            data = response.to_dict()
        elif hasattr(response, "model_dump"):
            data = response.model_dump(warnings=False)
        else:
            data = {}
        return data if isinstance(data, dict) else {}, "json_object"

    def _exception_response_text(self, exc: Exception) -> str | None:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        text = getattr(response, "text", None)
        if text is None:
            return None
        return str(text)[:4000]

    def search_author(self, author: AuthorRecord) -> QwenSearchResult:
        if not self._api_key:
            return QwenSearchResult([], [], [], failure_reason="qwen_api_key_missing")

        payload = {
            "model": self._model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": self._build_prompt(author)}]}],
            "temperature": 0,
            "response_format": self._build_response_format(),
            "tools": ([{"type": "web_search_image"}] if self._enable_web_search else []),
        }
        logger.info("qwen_web_search_image_started author_id=%s model=%s", author.author_id, self._model)
        try:
            data, response_mode = self._post_responses(payload)
        except APIStatusError as exc:
            text = self._exception_response_text(exc)
            return QwenSearchResult([], [], [], failure_reason="qwen_http_error", raw_content=text, response_text=text)
        except APITimeoutError as exc:
            text = str(exc)[:4000]
            return QwenSearchResult([], [], [], failure_reason="qwen_request_timeout", raw_content=text, response_text=text)
        except APIConnectionError as exc:
            text = str(exc)[:4000]
            return QwenSearchResult([], [], [], failure_reason="qwen_request_failed", raw_content=text, response_text=text)
        except Exception as exc:
            text = str(exc)[:4000]
            return QwenSearchResult([], [], [], failure_reason="qwen_request_failed", raw_content=text, response_text=text)

        response_text = self._extract_response_text(data)
        rows_from_text: list[dict[str, Any]] = []
        if response_text:
            obj = self._extract_json_object(response_text)
            if isinstance(obj, dict):
                rows_from_text = self._collect_urls_from_object(obj)
        rows_from_tool = self._collect_urls_from_tool_calls(data)
        deduped = self._dedupe_candidates(rows_from_text + rows_from_tool)
        if not deduped:
            return QwenSearchResult(
                profile_pages=[],
                image_candidates=[],
                filtered_candidates=[],
                failure_reason="qwen_web_search_image_no_results",
                raw_content=json.dumps(data, ensure_ascii=False)[:4000],
                response_text=(response_text or "")[:4000] or None,
                response_format_mode=response_mode,
                abandon_reason_log="web_search_image returned no usable urls",
            )
        logger.info("qwen_web_search_image_finished author_id=%s extracted_urls=%s", author.author_id, len(deduped))
        return QwenSearchResult(
            profile_pages=[],
            image_candidates=deduped,
            filtered_candidates=deduped,
            failure_reason=None,
            raw_content=json.dumps(data, ensure_ascii=False)[:4000],
            response_text=(response_text or "")[:4000] or None,
            response_format_mode=response_mode,
        )
