from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord
from avatar_pipeline.qwen_tools import QwenToolsClient


@dataclass(slots=True)
class SearchCandidate:
    image_url: str
    source_url: str
    title: str
    snippet: str
    mime: str
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    is_valid_image: bool | None = None
    invalid_reason: str | None = None


@dataclass(slots=True)
class SearchOutcome:
    profile_pages: list[dict[str, Any]]
    image_candidates: list[dict[str, Any]]
    filtered_candidates: list[dict[str, Any]]
    candidates: list[SearchCandidate]
    failure_reason: str | None = None
    reason_tags: list[str] | None = None
    raw_content: str | None = None
    response_text: str | None = None
    abandon_reason_log: str | None = None


@dataclass(slots=True)
class CachedImage:
    content: bytes
    mime: str


def _guess_mime_from_url(url: str) -> str:
    lower = url.lower()
    if ".jpg" in lower or ".jpeg" in lower:
        return "image/jpeg"
    if ".png" in lower:
        return "image/png"
    if ".webp" in lower:
        return "image/webp"
    return "image/jpeg"


class WebSearchClient:
    def __init__(
        self,
        http: HttpClient,
        max_candidates: int,
        qwen_api_key: str | None,
        qwen_base_url: str,
        qwen_response_path: str,
        qwen_model: str,
        qwen_timeout_seconds: int,
        qwen_min_call_interval_seconds: float,
        qwen_enable_web_search: bool,
        qwen_min_confidence: float,
    ) -> None:
        self._http = http
        self._max_candidates = max(1, max_candidates)
        self._image_cache: dict[str, CachedImage] = {}
        self._qwen_tools = QwenToolsClient(
            http=http,
            api_key=qwen_api_key,
            base_url=qwen_base_url,
            model=qwen_model,
            timeout_seconds=qwen_timeout_seconds,
            min_call_interval_seconds=qwen_min_call_interval_seconds,
            enable_web_search=qwen_enable_web_search,
            min_confidence=qwen_min_confidence,
            response_path=qwen_response_path,
        )
        self._last_search_diagnostics: dict[str, Any] = {"provider_mode": "qwen", "reason_tags": []}

    def provider_mode(self) -> str:
        return "qwen"

    def last_search_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_search_diagnostics)

    def search_author(self, author: AuthorRecord) -> SearchOutcome:
        result = self._qwen_tools.search_author(author)
        rows = result.filtered_candidates or result.image_candidates
        candidates: list[SearchCandidate] = []
        for row in rows[: self._max_candidates]:
            image_url = str(row.get("image_url") or "").strip()
            if not image_url.startswith("http"):
                continue
            candidates.append(
                SearchCandidate(
                    image_url=image_url,
                    source_url=str(row.get("source_url") or image_url).strip(),
                    title=str(row.get("title") or "").strip(),
                    snippet=str(row.get("snippet") or "").strip(),
                    mime=_guess_mime_from_url(image_url),
                )
            )
        failure_reason = result.failure_reason
        if not candidates and failure_reason is None:
            failure_reason = "qwen_web_search_image_no_candidates"
        reason_tags = [failure_reason] if failure_reason else []
        self._last_search_diagnostics = {
            "provider_mode": "qwen",
            "reason_tags": reason_tags,
            "image_candidates_count": len(result.image_candidates),
            "filtered_candidates_count": len(rows),
            "kept_count": len(candidates),
        }
        return SearchOutcome(
            profile_pages=[],
            image_candidates=result.image_candidates,
            filtered_candidates=rows,
            candidates=candidates,
            failure_reason=failure_reason,
            reason_tags=reason_tags,
            raw_content=result.raw_content,
            response_text=result.response_text,
            abandon_reason_log=result.abandon_reason_log,
        )

    def _normalize_image_url(self, image_url: str) -> str:
        parsed = urlparse(image_url)
        query = parse_qs(parsed.query)
        drop_keys = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "cache",
            "cb",
            "v",
            "version",
            "width",
            "height",
            "w",
            "h",
            "size",
        }
        cleaned_query = {key: values for key, values in query.items() if key.lower() not in drop_keys}
        query_str = urlencode(cleaned_query, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", query_str, ""))

    def _parse_png_size(self, content: bytes) -> tuple[int, int] | None:
        if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return int.from_bytes(content[16:20], "big"), int.from_bytes(content[20:24], "big")

    def _parse_jpeg_size(self, content: bytes) -> tuple[int, int] | None:
        if len(content) < 4 or content[:2] != b"\xff\xd8":
            return None
        i = 2
        while i + 9 < len(content):
            if content[i] != 0xFF:
                i += 1
                continue
            marker = content[i + 1]
            i += 2
            if marker in (0xD8, 0xD9):
                continue
            if i + 2 > len(content):
                break
            seg_len = int.from_bytes(content[i:i + 2], "big")
            if seg_len < 2 or i + seg_len > len(content):
                break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                return int.from_bytes(content[i + 5:i + 7], "big"), int.from_bytes(content[i + 3:i + 5], "big")
            i += seg_len
        return None

    def _parse_webp_size(self, content: bytes) -> tuple[int, int] | None:
        if len(content) < 30 or content[0:4] != b"RIFF" or content[8:12] != b"WEBP":
            return None
        if content[12:16] == b"VP8X":
            return 1 + int.from_bytes(content[24:27], "little"), 1 + int.from_bytes(content[27:30], "little")
        return None

    def _parse_image_size(self, content: bytes, mime: str) -> tuple[int | None, int | None]:
        parser_by_mime = {
            "image/png": self._parse_png_size,
            "image/jpeg": self._parse_jpeg_size,
            "image/webp": self._parse_webp_size,
        }
        parser = parser_by_mime.get(mime)
        if parser is None:
            return None, None
        parsed = parser(content)
        if parsed is None:
            return None, None
        return parsed

    def download_image(self, url: str) -> tuple[bytes, str]:
        cache_key = self._normalize_image_url(url)
        cached = self._image_cache.get(cache_key)
        if cached is not None:
            return cached.content, cached.mime
        resp = self._http.request("GET", url, stream=False)
        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        mime = content_type if content_type else _guess_mime_from_url(url)
        content = resp.content
        self._image_cache[cache_key] = CachedImage(content=content, mime=mime)
        return content, mime

    def enrich_candidate_image_metadata(self, candidate: SearchCandidate, allowed_mime: set[str]) -> SearchCandidate:
        try:
            content, mime = self.download_image(candidate.image_url)
            candidate.mime = mime or candidate.mime
            candidate.size_bytes = len(content)
            candidate.width, candidate.height = self._parse_image_size(content, candidate.mime)
            if candidate.size_bytes <= 0:
                candidate.is_valid_image = False
                candidate.invalid_reason = "empty_image_bytes"
            elif candidate.mime not in allowed_mime:
                candidate.is_valid_image = False
                candidate.invalid_reason = f"invalid_image_mime:{candidate.mime}"
            elif candidate.width is None or candidate.height is None:
                candidate.is_valid_image = False
                candidate.invalid_reason = "image_dimension_unknown"
            else:
                candidate.is_valid_image = True
                candidate.invalid_reason = None
        except Exception:
            candidate.is_valid_image = False
            candidate.invalid_reason = "image_metadata_fetch_failed"
        return candidate

    def content_sha256(self, url: str) -> str:
        content, _ = self.download_image(url)
        return hashlib.sha256(content).hexdigest()

