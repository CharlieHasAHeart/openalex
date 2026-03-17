from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord
from avatar_pipeline.profile_image_extractor import ProfileImageExtractor
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
    linked_profile_url: str | None = None
    linked_profile_domain: str | None = None
    score: float | None = None
    source_type: str | None = None
    alt_text: str | None = None


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
    usage_total_tokens: int | None = None


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
        qwen_max_output_tokens: int,
        qwen_sdk_max_retries: int,
        profile_page_fetch_timeout_seconds: int,
        profile_page_max_count: int,
        profile_image_max_per_page: int,
        profile_image_min_score: float,
    ) -> None:
        self._http = http
        self._max_candidates = max(1, max_candidates)
        self._profile_page_fetch_timeout_seconds = max(3, int(profile_page_fetch_timeout_seconds))
        self._profile_page_max_count = max(1, int(profile_page_max_count))
        self._profile_image_min_score = float(profile_image_min_score)
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
            max_candidates=self._profile_page_max_count,
            max_output_tokens=qwen_max_output_tokens,
            sdk_max_retries=qwen_sdk_max_retries,
            response_path=qwen_response_path,
        )
        self._profile_extractor = ProfileImageExtractor(
            profile_image_max_per_page=profile_image_max_per_page,
            profile_image_min_score=profile_image_min_score,
        )
        self._last_search_diagnostics: dict[str, Any] = {"provider_mode": "qwen", "reason_tags": []}

    def provider_mode(self) -> str:
        return "qwen"

    def last_search_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_search_diagnostics)

    def _is_http_url(self, url: str) -> bool:
        parsed = urlparse(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _normalize_domain(self, url: str) -> str:
        host = urlparse(url).netloc.lower().strip()
        return host[4:] if host.startswith("www.") else host

    def _clean_profile_pages(self, pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in pages:
            profile_url = str(row.get("profile_url") or row.get("url") or "").strip()
            if not self._is_http_url(profile_url):
                continue
            dedupe_key = profile_url.rstrip("/")
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            cleaned.append(
                {
                    "site": str(row.get("site") or self._normalize_domain(profile_url)).strip(),
                    "profile_url": profile_url,
                    "reason": str(row.get("reason") or "profile_page").strip(),
                    "confidence": row.get("confidence"),
                }
            )
            if len(cleaned) >= self._profile_page_max_count:
                break
        return cleaned

    def _fetch_profile_page_html(self, profile_url: str) -> str | None:
        try:
            resp = self._http.request("GET", profile_url, timeout=self._profile_page_fetch_timeout_seconds, stream=False)
        except Exception:
            return None
        content_type = (resp.headers.get("Content-Type") or "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type and content_type:
            return None
        try:
            resp.encoding = resp.encoding or resp.apparent_encoding or "utf-8"
        except Exception:
            pass
        return resp.text[:400_000]

    def _dedupe_image_candidates(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best_by_url: dict[str, dict[str, Any]] = {}
        for row in rows:
            image_url = str(row.get("image_url") or "").strip()
            if not self._is_http_url(image_url):
                continue
            current = best_by_url.get(image_url)
            if current is None or float(row.get("score") or 0.0) > float(current.get("score") or 0.0):
                best_by_url[image_url] = row
        merged = list(best_by_url.values())
        merged.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        return merged

    def _to_search_candidate(self, row: dict[str, Any]) -> SearchCandidate:
        image_url = str(row.get("image_url") or "").strip()
        return SearchCandidate(
            image_url=image_url,
            source_url=str(row.get("source_url") or row.get("linked_profile_url") or image_url).strip(),
            title=str(row.get("title") or "").strip(),
            snippet=str(row.get("snippet") or "").strip(),
            mime=_guess_mime_from_url(image_url),
            width=row.get("declared_width") if isinstance(row.get("declared_width"), int) else None,
            height=row.get("declared_height") if isinstance(row.get("declared_height"), int) else None,
            linked_profile_url=str(row.get("linked_profile_url") or "").strip() or None,
            linked_profile_domain=str(row.get("linked_profile_domain") or "").strip() or None,
            score=float(row.get("score")) if isinstance(row.get("score"), (int, float)) else None,
            source_type=str(row.get("source_type") or "").strip() or None,
            alt_text=str(row.get("alt_text") or "").strip() or None,
        )

    def _from_qwen_image_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        image_url = str(row.get("image_url") or "").strip()
        if not self._is_http_url(image_url):
            return None
        source_url = str(row.get("source_url") or "").strip()
        linked_profile_url = source_url if self._is_http_url(source_url) else None
        confidence_raw = row.get("confidence")
        confidence = float(confidence_raw) if isinstance(confidence_raw, (int, float)) else 0.0
        return {
            "image_url": image_url,
            "source_url": source_url or image_url,
            "title": str(row.get("title") or "").strip(),
            "snippet": str(row.get("snippet") or "").strip(),
            "linked_profile_url": linked_profile_url,
            "linked_profile_domain": self._normalize_domain(linked_profile_url) if linked_profile_url else self._normalize_domain(image_url),
            "source_type": str(row.get("source_type") or "qwen_web_search_image").strip(),
            "alt_text": str(row.get("alt_text") or "").strip(),
            "score": round(1.5 + confidence * 2.0, 3),
            "reason": str(row.get("reason") or "qwen_web_search_image").strip(),
            "declared_width": None,
            "declared_height": None,
        }

    def search_author(self, author: AuthorRecord) -> SearchOutcome:
        result = self._qwen_tools.search_author(author)
        cleaned_profile_pages = self._clean_profile_pages(result.profile_pages)
        qwen_image_rows = [row for row in (self._from_qwen_image_row(item) for item in result.image_candidates) if row is not None]

        if result.failure_reason and not cleaned_profile_pages and not qwen_image_rows:
            failure_reason = result.failure_reason
            reason_tags = [failure_reason] if failure_reason else []
            self._last_search_diagnostics = {
                "provider_mode": "qwen",
                "reason_tags": reason_tags,
                "profile_pages_count": len(cleaned_profile_pages),
                "image_candidates_count": len(qwen_image_rows),
                "filtered_candidates_count": len(qwen_image_rows[: self._max_candidates]),
                "kept_count": len(qwen_image_rows[: self._max_candidates]),
            }
            return SearchOutcome(
                profile_pages=cleaned_profile_pages,
                image_candidates=qwen_image_rows,
                filtered_candidates=qwen_image_rows[: self._max_candidates],
                candidates=[self._to_search_candidate(row) for row in qwen_image_rows[: self._max_candidates]],
                failure_reason=failure_reason,
                reason_tags=reason_tags,
                raw_content=result.raw_content,
                response_text=result.response_text,
                abandon_reason_log=result.abandon_reason_log,
                usage_total_tokens=result.usage_total_tokens,
            )

        if not cleaned_profile_pages and not qwen_image_rows:
            failure_reason = result.failure_reason or "qwen_web_search_no_profile_pages"
            reason_tags = [failure_reason]
            self._last_search_diagnostics = {
                "provider_mode": "qwen",
                "reason_tags": reason_tags,
                "profile_pages_count": 0,
                "image_candidates_count": 0,
                "filtered_candidates_count": 0,
                "kept_count": 0,
            }
            return SearchOutcome(
                profile_pages=[],
                image_candidates=[],
                filtered_candidates=[],
                candidates=[],
                failure_reason=failure_reason,
                reason_tags=reason_tags,
                raw_content=result.raw_content,
                response_text=result.response_text,
                abandon_reason_log=result.abandon_reason_log,
                usage_total_tokens=result.usage_total_tokens,
            )

        extracted_candidates: list[dict[str, Any]] = []
        for page in cleaned_profile_pages:
            profile_url = str(page.get("profile_url") or "").strip()
            if not profile_url:
                continue
            html_text = self._fetch_profile_page_html(profile_url)
            if not html_text:
                continue
            extracted_candidates.extend(self._profile_extractor.extract(author, profile_url, html_text))
        extracted_candidates.extend(qwen_image_rows)

        deduped_candidates = self._dedupe_image_candidates(extracted_candidates)
        filtered_rows = [
            row for row in deduped_candidates if float(row.get("score") or 0.0) >= self._profile_image_min_score
        ]
        filtered_rows = filtered_rows[: self._max_candidates]
        candidates = [self._to_search_candidate(row) for row in filtered_rows]

        failure_reason = result.failure_reason
        if not candidates and failure_reason is None:
            failure_reason = "profile_pages_no_image_candidates"
        if candidates:
            failure_reason = None
        reason_tags = [failure_reason] if failure_reason else []
        self._last_search_diagnostics = {
            "provider_mode": "qwen",
            "reason_tags": reason_tags,
            "profile_pages_count": len(cleaned_profile_pages),
            "image_candidates_count": len(deduped_candidates),
            "filtered_candidates_count": len(filtered_rows),
            "kept_count": len(candidates),
        }
        return SearchOutcome(
            profile_pages=cleaned_profile_pages,
            image_candidates=deduped_candidates,
            filtered_candidates=filtered_rows,
            candidates=candidates,
            failure_reason=failure_reason,
            reason_tags=reason_tags,
            raw_content=result.raw_content,
            response_text=result.response_text,
            abandon_reason_log=result.abandon_reason_log,
            usage_total_tokens=result.usage_total_tokens,
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
            parsed_width, parsed_height = self._parse_image_size(content, candidate.mime)
            candidate.width = parsed_width or candidate.width
            candidate.height = parsed_height or candidate.height
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
