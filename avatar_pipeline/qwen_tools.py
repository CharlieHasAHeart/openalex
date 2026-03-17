from __future__ import annotations

import json
import logging
import re
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord

logger = logging.getLogger(__name__)
_QWEN_CALL_LOCK = threading.Lock()
_QWEN_LAST_CALL_TS = 0.0
_URL_PATTERN = re.compile(r"https?://[^\s\]\[\)\(\"'<>]+", re.IGNORECASE)

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
    usage_total_tokens: int | None = None


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
        max_candidates: int,
        max_output_tokens: int,
        sdk_max_retries: int,
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
        self._max_candidates = max(1, int(max_candidates))
        self._max_output_tokens = max(64, int(max_output_tokens))
        self._sdk_max_retries = max(0, int(sdk_max_retries))
        self._response_path = response_path if response_path.startswith("/") else f"/{response_path}"
        self._client = (
            OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout_seconds,
                max_retries=self._sdk_max_retries,
            )
            if self._api_key
            else None
        )

    @property
    def model(self) -> str:
        return self._model

    def _build_response_format(self) -> dict[str, Any]:
        return {"type": "json_object"}

    def _truncate_text(self, value: Any, max_chars: int) -> str:
        text = str(value or "").strip()
        if len(text) <= max_chars:
            return text
        return f"{text[: max(0, max_chars - 1)]}..."

    def _compact_sequence(self, value: Any, max_items: int, max_item_chars: int) -> list[str]:
        if not isinstance(value, list):
            return []
        compact: list[str] = []
        for item in value:
            if len(compact) >= max_items:
                break
            if isinstance(item, dict):
                name = (
                    item.get("display_name")
                    or item.get("institution")
                    or item.get("name")
                    or item.get("raw_affiliation_string")
                    or ""
                )
                text = self._truncate_text(name, max_item_chars)
            else:
                text = self._truncate_text(item, max_item_chars)
            if text:
                compact.append(text)
        return compact

    def _author_ctx(self, author: AuthorRecord) -> dict[str, Any]:
        return {
            "author_id": author.author_id,
            "display_name": self._truncate_text(author.display_name, 96),
            "orcid": author.orcid,
            "institution_name": self._truncate_text(author.institution_name, 96),
            "last_known_institutions": self._compact_sequence(author.last_known_institutions, max_items=3, max_item_chars=96),
        }

    def _build_prompt(self, author: AuthorRecord) -> str:
        return (
            "Use web_search to find relevant author pages for this exact researcher.\\n"
            "You must anchor identity by ORCID first, then verify with display_name and institution.\\n"
            "Prioritize result pages from researchgate.net and academia.edu, then orcid.org and official academic pages.\\n"
            "Return page URLs only (do not return direct image URLs); image extraction is done by code.\\n"
            "Return one compact JSON object only. Keep reason <= 12 words.\\n"
            'Schema: {"profile_pages":[{"site":"","profile_url":"","reason":"","confidence":0.0}],"image_candidates":[],"filtered_candidates":[],"failure_reason":""}\\n'
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

    def _is_http_url(self, value: str) -> bool:
        if not value.startswith(("http://", "https://")):
            return False
        parsed = urlparse(value)
        return bool(parsed.scheme and parsed.netloc)

    def _site_from_url(self, url: str) -> str:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host

    def _collect_profile_pages_from_object(self, obj: Any) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        raw_pages = obj.get("profile_pages") if isinstance(obj, dict) else None
        if not isinstance(raw_pages, list):
            return pages
        for item in raw_pages:
            if not isinstance(item, dict):
                continue
            profile_url = str(item.get("profile_url") or item.get("url") or "").strip()
            if not self._is_http_url(profile_url):
                continue
            confidence_raw = item.get("confidence")
            confidence: float | None
            if isinstance(confidence_raw, (int, float)):
                confidence = float(confidence_raw)
            else:
                confidence = None
            if confidence is not None and confidence < self._min_confidence:
                continue
            pages.append(
                {
                    "site": str(item.get("site") or self._site_from_url(profile_url)).strip(),
                    "profile_url": profile_url,
                    "reason": str(item.get("reason") or "qwen_profile_output").strip(),
                    "confidence": confidence if confidence is not None else self._min_confidence,
                }
            )
        return pages

    def _extract_urls_from_any(self, obj: Any) -> list[str]:
        urls: list[str] = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in {"url", "profile_url", "link", "source_url"} and isinstance(value, str):
                    urls.extend(match.group(0) for match in _URL_PATTERN.finditer(value))
                else:
                    urls.extend(self._extract_urls_from_any(value))
        elif isinstance(obj, list):
            for item in obj:
                urls.extend(self._extract_urls_from_any(item))
        elif isinstance(obj, str):
            urls.extend(match.group(0) for match in _URL_PATTERN.finditer(obj))
        return urls

    def _collect_profile_pages_from_tool_calls(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        output = payload.get("output")
        if not isinstance(output, list):
            return []
        pages: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if "web_search" not in str(item.get("type") or ""):
                continue
            action = item.get("action")
            if isinstance(action, dict):
                sources = action.get("sources")
                if isinstance(sources, list):
                    for source in sources:
                        if not isinstance(source, dict):
                            continue
                        source_url = str(source.get("url") or "").strip()
                        if not self._is_http_url(source_url):
                            continue
                        pages.append(
                            {
                                "site": self._site_from_url(source_url),
                                "profile_url": source_url,
                                "reason": "web_search_action_source",
                                "confidence": self._min_confidence,
                            }
                        )
            raw_output = item.get("output")
            parsed_output: Any = raw_output
            if isinstance(raw_output, str):
                try:
                    parsed_output = json.loads(raw_output)
                except Exception:
                    parsed_output = raw_output
            if isinstance(action, dict):
                parsed_output = {"action": action, "output": parsed_output}
            urls = self._extract_urls_from_any(parsed_output)
            for url in urls:
                if not self._is_http_url(url):
                    continue
                pages.append(
                    {
                        "site": self._site_from_url(url),
                        "profile_url": url,
                        "reason": "web_search_tool_call",
                        "confidence": self._min_confidence,
                    }
                )
        return pages

    def _collect_profile_pages_from_text(self, response_text: str | None) -> list[dict[str, Any]]:
        if not response_text:
            return []
        pages: list[dict[str, Any]] = []
        for match in _URL_PATTERN.finditer(response_text):
            url = match.group(0).rstrip(".,;)")
            if not self._is_http_url(url):
                continue
            pages.append(
                {
                    "site": self._site_from_url(url),
                    "profile_url": url,
                    "reason": "response_text_fallback",
                    "confidence": self._min_confidence,
                }
            )
        return pages

    def _sanitize_profile_pages(self, pages: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in pages:
            profile_url = str(row.get("profile_url") or "").strip()
            if not self._is_http_url(profile_url):
                continue
            key = profile_url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            confidence_raw = row.get("confidence")
            confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else self._min_confidence
            deduped.append(
                {
                    "site": str(row.get("site") or self._site_from_url(profile_url)).strip(),
                    "profile_url": profile_url,
                    "reason": str(row.get("reason") or "profile_page").strip(),
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
            if len(deduped) >= max_count:
                break
        return deduped

    def _collect_images_from_object(self, obj: Any) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        raw_rows = obj.get("image_candidates") if isinstance(obj, dict) else None
        if not isinstance(raw_rows, list):
            return rows
        for item in raw_rows:
            if not isinstance(item, dict):
                continue
            image_url = str(item.get("image_url") or item.get("url") or "").strip()
            if not self._is_http_url(image_url):
                continue
            confidence_raw = item.get("confidence")
            confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else self._min_confidence
            rows.append(
                {
                    "image_url": image_url,
                    "source_url": str(item.get("source_url") or "").strip(),
                    "title": str(item.get("title") or "").strip(),
                    "snippet": str(item.get("snippet") or "").strip(),
                    "reason": str(item.get("reason") or "qwen_image_output").strip(),
                    "source_type": "qwen_web_search_image",
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
        return rows

    def _collect_images_from_tool_calls(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        output = payload.get("output")
        if not isinstance(output, list):
            return []
        rows: list[dict[str, Any]] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if "web_search_image" not in str(item.get("type") or ""):
                continue
            raw_output = item.get("output")
            parsed_output: Any = raw_output
            if isinstance(raw_output, str):
                try:
                    parsed_output = json.loads(raw_output)
                except Exception:
                    parsed_output = raw_output
            if isinstance(parsed_output, list):
                iterable: list[Any] = parsed_output
            else:
                iterable = [parsed_output]
            for row in iterable:
                if not isinstance(row, dict):
                    continue
                image_url = str(row.get("url") or row.get("image_url") or "").strip()
                if not self._is_http_url(image_url):
                    continue
                source_url = str(row.get("source_url") or row.get("page_url") or "").strip()
                rows.append(
                    {
                        "image_url": image_url,
                        "source_url": source_url,
                        "title": str(row.get("title") or "").strip(),
                        "snippet": str(row.get("snippet") or "").strip(),
                        "reason": "web_search_image_tool_call",
                        "source_type": "qwen_web_search_image",
                        "confidence": self._min_confidence,
                    }
                )
        return rows

    def _collect_images_from_text(self, response_text: str | None) -> list[dict[str, Any]]:
        if not response_text:
            return []
        rows: list[dict[str, Any]] = []
        for match in _URL_PATTERN.finditer(response_text):
            image_url = match.group(0).rstrip(".,;)")
            if not self._is_http_url(image_url):
                continue
            lower = image_url.lower()
            if not any(lower.endswith(ext) or f"{ext}?" in lower for ext in (".jpg", ".jpeg", ".png", ".webp")):
                continue
            rows.append(
                {
                    "image_url": image_url,
                    "source_url": "",
                    "title": "",
                    "snippet": "",
                    "reason": "response_text_image_fallback",
                    "source_type": "qwen_web_search_image",
                    "confidence": self._min_confidence,
                }
            )
        return rows

    def _sanitize_image_candidates(self, rows: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in rows:
            image_url = str(row.get("image_url") or "").strip()
            if not self._is_http_url(image_url):
                continue
            key = image_url.rstrip("/")
            if key in seen:
                continue
            seen.add(key)
            confidence_raw = row.get("confidence")
            confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else self._min_confidence
            deduped.append(
                {
                    "image_url": image_url,
                    "source_url": str(row.get("source_url") or "").strip(),
                    "title": str(row.get("title") or "").strip(),
                    "snippet": str(row.get("snippet") or "").strip(),
                    "reason": str(row.get("reason") or "image_candidate").strip(),
                    "source_type": str(row.get("source_type") or "qwen_web_search_image").strip(),
                    "confidence": max(0.0, min(1.0, confidence)),
                }
            )
            if len(deduped) >= max_count:
                break
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
            max_output_tokens=payload.get("max_output_tokens"),
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

    def _extract_usage_total_tokens(self, payload: dict[str, Any]) -> int | None:
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return None
        value = usage.get("total_tokens")
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, float):
            return int(value) if value >= 0 else None
        return None

    def search_author(self, author: AuthorRecord) -> QwenSearchResult:
        if not self._api_key:
            return QwenSearchResult([], [], [], failure_reason="qwen_api_key_missing")
        if not author.orcid:
            return QwenSearchResult(
                [],
                [],
                [],
                failure_reason="qwen_profile_search_no_orcid",
                abandon_reason_log="orcid missing for profile search",
            )

        payload = {
            "model": self._model,
            "input": [{"role": "user", "content": [{"type": "input_text", "text": self._build_prompt(author)}]}],
            "temperature": 0,
            "max_output_tokens": self._max_output_tokens,
            "response_format": self._build_response_format(),
            "tools": ([{"type": "web_search"}] if self._enable_web_search else []),
        }
        logger.info("qwen_web_search_profile_started author_id=%s model=%s", author.author_id, self._model)
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
        usage_total_tokens = self._extract_usage_total_tokens(data)

        response_obj = self._extract_json_object(response_text or "") if response_text else None
        profile_pages_from_text: list[dict[str, Any]] = []
        images_from_text: list[dict[str, Any]] = []
        invalid_output = False
        if response_text and response_obj is None:
            invalid_output = True
        if isinstance(response_obj, dict):
            profile_pages_from_text = self._collect_profile_pages_from_object(response_obj)
            images_from_text = self._collect_images_from_object(response_obj)
            if not profile_pages_from_text and not isinstance(response_obj.get("profile_pages"), list):
                invalid_output = True

        profile_pages_from_tool = self._collect_profile_pages_from_tool_calls(data)
        profile_pages_from_fallback = self._collect_profile_pages_from_text(response_text)
        images_from_tool = self._collect_images_from_tool_calls(data)
        images_from_fallback = self._collect_images_from_text(response_text)
        deduped_pages = self._sanitize_profile_pages(
            profile_pages_from_text + profile_pages_from_tool + profile_pages_from_fallback,
            max_count=min(self._max_candidates, 5),
        )
        deduped_images = self._sanitize_image_candidates(
            images_from_text + images_from_tool + images_from_fallback,
            max_count=self._max_candidates,
        )

        raw_content = json.dumps(data, ensure_ascii=False)[:4000]
        response_text_short = (response_text or "")[:4000] or None
        if not deduped_pages and not deduped_images:
            reason = "qwen_profile_search_invalid_output" if invalid_output else "qwen_web_search_no_profile_pages"
            return QwenSearchResult(
                profile_pages=[],
                image_candidates=[],
                filtered_candidates=[],
                failure_reason=reason,
                raw_content=raw_content,
                response_text=response_text_short,
                response_format_mode=response_mode,
                abandon_reason_log="web_search returned no usable profile pages",
                usage_total_tokens=usage_total_tokens,
            )

        logger.info("qwen_web_search_profile_finished author_id=%s profile_pages=%s", author.author_id, len(deduped_pages))
        return QwenSearchResult(
            profile_pages=deduped_pages,
            image_candidates=deduped_images,
            filtered_candidates=deduped_images,
            failure_reason=None,
            raw_content=raw_content,
            response_text=response_text_short,
            response_format_mode=response_mode,
            usage_total_tokens=usage_total_tokens,
        )
