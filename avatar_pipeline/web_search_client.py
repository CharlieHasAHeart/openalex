from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from html import unescape
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord
from avatar_pipeline.qwen_tools import QwenToolsClient

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchCandidate:
    image_url: str
    source_url: str
    title: str
    snippet: str
    mime: str
    page_h1: str | None = None
    page_meta_description: str | None = None
    image_alt: str | None = None
    nearby_text: str | None = None
    source_domain: str | None = None
    page_title: str | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    is_portrait: bool | None = None
    is_valid_image: bool | None = None
    invalid_reason: str | None = None
    pre_rank_score: float | None = None
    source_type: str | None = None
    discovery_score: float | None = None
    discovery_evidence: str | None = None
    normalized_image_url: str | None = None
    image_fingerprint: str | None = None
    merged_count: int | None = None
    supporting_source_types: list[str] | None = None
    supporting_source_domains: list[str] | None = None
    content_deduped: bool | None = None
    cluster_evidence_summary: str | None = None


@dataclass(slots=True)
class CandidateCluster:
    canonical: SearchCandidate
    members: list[SearchCandidate]


@dataclass(slots=True)
class SearchOutcome:
    profile_pages: list[dict[str, Any]]
    image_candidates: list[dict[str, Any]]
    filtered_candidates: list[dict[str, Any]]
    candidates: list[SearchCandidate]
    failure_reason: str | None = None
    reason_tags: list[str] | None = None


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
        qwen_enable_web_search: bool,
        qwen_enable_t2i_search: bool,
        qwen_min_confidence: float,
    ) -> None:
        self._http = http
        self._max_candidates = max(1, max_candidates)
        self._min_confidence = max(0.0, min(1.0, qwen_min_confidence))
        self._qwen_tools = QwenToolsClient(
            http=http,
            api_key=qwen_api_key,
            base_url=qwen_base_url,
            model=qwen_model,
            timeout_seconds=qwen_timeout_seconds,
            enable_web_search=qwen_enable_web_search,
            enable_t2i_search=qwen_enable_t2i_search,
            min_confidence=qwen_min_confidence,
            response_path=qwen_response_path,
        )
        self._last_search_diagnostics: dict[str, Any] = {"provider_mode": "qwen", "reason_tags": []}

    def provider_mode(self) -> str:
        return "qwen"

    def last_search_diagnostics(self) -> dict[str, Any]:
        return dict(self._last_search_diagnostics)

    def _normalize_source_type(self, raw: str) -> str:
        value = (raw or "").strip().lower()
        if value in {
            "institution_profile",
            "institution_directory",
            "lab_people_page",
            "conference_bio",
            "generic_search_result",
            "t2i_result",
            "qwen_verified_profile_image",
        }:
            return value
        if any(token in value for token in ("faculty", "profile", "staff", "person")):
            return "institution_profile"
        if any(token in value for token in ("people", "directory", "member", "team")):
            return "institution_directory"
        if "conference" in value or "speaker" in value or "bio" in value:
            return "conference_bio"
        if "t2i" in value:
            return "t2i_result"
        return "generic_search_result"

    def _name_tokens(self, name: str) -> list[str]:
        return [token for token in re.split(r"\s+", name.lower()) if token and len(token) > 1]

    def _name_match_strength(self, text: str, display_name: str) -> float:
        lower = text.lower()
        name = display_name.strip().lower()
        if not name:
            return 0.0
        if name in lower:
            return 1.0
        tokens = self._name_tokens(name)
        if len(tokens) >= 2 and all(token in lower for token in tokens[:2]):
            return 0.7
        if any(token in lower for token in tokens):
            return 0.35
        return 0.0

    def _build_candidate(self, author: AuthorRecord, item: dict[str, Any]) -> SearchCandidate | None:
        image_url = str(item.get("image_url") or "").strip()
        source_url = str(item.get("source_url") or "").strip()
        if not image_url.startswith("http") or not source_url.startswith("http"):
            return None
        try:
            confidence = float(item.get("confidence"))
        except Exception:
            return None
        if confidence < self._min_confidence:
            return None
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        source_type = self._normalize_source_type(str(item.get("source_type") or ""))
        source_domain = urlparse(source_url).hostname
        reason = str(item.get("reason") or "").strip()
        discovery_evidence = f"qwen_filtered(conf={confidence:.2f})"
        if reason:
            discovery_evidence += f"; reason={reason}"
        return SearchCandidate(
            image_url=image_url,
            source_url=source_url,
            title=title,
            snippet=snippet,
            mime=_guess_mime_from_url(image_url),
            source_domain=source_domain,
            source_type=source_type,
            discovery_score=2.0 + min(confidence, 1.0),
            discovery_evidence=discovery_evidence,
            page_title=title or None,
            image_alt=self._clean_text(str(item.get("image_alt") or "")),
            nearby_text=self._clean_text(str(item.get("nearby_text") or "")),
        )

    def search_author(self, author: AuthorRecord) -> SearchOutcome:
        logger.info("qwen_provider_prompt_triggered author_id=%s model=%s", author.author_id, self._qwen_tools.model)
        result = self._qwen_tools.search_author(author, max_candidates=self._max_candidates)
        candidates: list[SearchCandidate] = []
        reason_tags: set[str] = set()
        if result.failure_reason:
            reason_tags.add(result.failure_reason)
        if not result.profile_pages:
            reason_tags.add("qwen_no_profile_pages")
        if not result.image_candidates:
            reason_tags.add("qwen_no_image_candidates")
        if not result.filtered_candidates:
            reason_tags.add("qwen_empty_filtered_candidates")
        for row in result.filtered_candidates:
            if not isinstance(row, dict):
                continue
            candidate = self._build_candidate(author, row)
            if candidate is not None:
                candidates.append(candidate)
            if len(candidates) >= self._max_candidates:
                break
        self._last_search_diagnostics = {
            "provider_mode": "qwen",
            "reason_tags": sorted(reason_tags),
            "profile_pages_count": len(result.profile_pages),
            "image_candidates_count": len(result.image_candidates),
            "filtered_candidates_count": len(result.filtered_candidates),
            "kept_count": len(candidates),
            "failure_reason": result.failure_reason,
        }
        logger.info(
            "websearch_provider_summary author_id=%s provider_mode=qwen final_candidates=%s reasons=%s",
            author.author_id,
            len(candidates),
            ",".join(sorted(reason_tags)) if reason_tags else "ok",
        )
        return SearchOutcome(
            profile_pages=result.profile_pages,
            image_candidates=result.image_candidates,
            filtered_candidates=result.filtered_candidates,
            candidates=candidates,
            failure_reason=result.failure_reason,
            reason_tags=sorted(reason_tags),
        )

    def _clean_text(self, value: str | None) -> str | None:
        if not value:
            return None
        text = re.sub(r"<[^>]+>", " ", value)
        text = " ".join(unescape(text).split())
        return text or None

    def _extract_page_context(self, html: str, image_url: str) -> dict[str, str | None]:
        page_title = None
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if title_match:
            page_title = self._clean_text(title_match.group(1))
        page_h1 = None
        h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", html, re.IGNORECASE | re.DOTALL)
        if h1_match:
            page_h1 = self._clean_text(h1_match.group(1))
        meta_description = None
        meta_match = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']', html, re.IGNORECASE | re.DOTALL)
        if meta_match:
            meta_description = self._clean_text(meta_match.group(1))

        image_alt = None
        nearby_text = None
        chosen_match = None
        for img_match in re.finditer(r"<img[^>]+>", html, re.IGNORECASE | re.DOTALL):
            tag = img_match.group(0)
            src_match = re.search(r'src=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if not src_match:
                continue
            src = unescape(src_match.group(1)).strip().lower()
            if src and (src in image_url.lower() or image_url.lower() in src):
                chosen_match = img_match
                alt_match = re.search(r'alt=["\']([^"\']*)["\']', tag, re.IGNORECASE)
                if alt_match:
                    image_alt = self._clean_text(alt_match.group(1))
                break
        if chosen_match:
            start = max(chosen_match.start() - 250, 0)
            end = min(chosen_match.end() + 250, len(html))
            nearby_text = self._clean_text(html[start:end])
        return {
            "page_title": page_title,
            "page_h1": page_h1,
            "page_meta_description": meta_description,
            "image_alt": image_alt,
            "nearby_text": nearby_text,
        }

    def enrich_candidates_context(self, candidates: list[SearchCandidate], limit: int = 5) -> list[SearchCandidate]:
        for candidate in candidates[: max(0, limit)]:
            try:
                page = self._http.request("GET", candidate.source_url)
                context = self._extract_page_context(page.text, candidate.image_url)
                candidate.page_title = context.get("page_title") or candidate.title
                candidate.page_h1 = context.get("page_h1")
                candidate.page_meta_description = context.get("page_meta_description")
                candidate.image_alt = context.get("image_alt") or candidate.image_alt
                candidate.nearby_text = context.get("nearby_text") or candidate.nearby_text
            except Exception:
                candidate.page_title = candidate.page_title or candidate.title
        return candidates

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

    def enrich_candidates_image_metadata(
        self,
        candidates: list[SearchCandidate],
        limit: int = 5,
        allowed_mime: set[str] | None = None,
        min_edge_px: int = 200,
    ) -> list[SearchCandidate]:
        allowed = allowed_mime or {"image/jpeg", "image/png", "image/webp"}
        for candidate in candidates[: max(0, limit)]:
            try:
                content, mime = self.download_image(candidate.image_url)
                candidate.mime = mime or candidate.mime
                candidate.size_bytes = len(content)
                candidate.image_fingerprint = hashlib.sha256(content).hexdigest()
                candidate.width, candidate.height = self._parse_image_size(content, candidate.mime)
                if candidate.width and candidate.height:
                    candidate.is_portrait = candidate.height >= candidate.width
                if candidate.size_bytes <= 0:
                    candidate.is_valid_image = False
                    candidate.invalid_reason = "empty_image_bytes"
                elif candidate.mime not in allowed:
                    candidate.is_valid_image = False
                    candidate.invalid_reason = f"invalid_image_mime:{candidate.mime}"
                elif candidate.width is None or candidate.height is None:
                    candidate.is_valid_image = False
                    candidate.invalid_reason = "image_dimension_unknown"
                elif min(candidate.width, candidate.height) < min_edge_px:
                    candidate.is_valid_image = False
                    candidate.invalid_reason = "invalid_image_too_small"
                elif max(candidate.width / candidate.height, candidate.height / candidate.width) > 4.5:
                    candidate.is_valid_image = False
                    candidate.invalid_reason = "invalid_image_extreme_aspect_ratio"
                else:
                    candidate.is_valid_image = True
                    candidate.invalid_reason = None
            except Exception:
                candidate.is_valid_image = False
                candidate.invalid_reason = "image_metadata_fetch_failed"
        return candidates

    def _normalize_image_url(self, image_url: str) -> str:
        parsed = urlparse(image_url)
        query = parse_qs(parsed.query)
        drop_keys = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "cache", "cb", "v", "version", "width", "height", "w", "h", "size"}
        cleaned_query = {key: values for key, values in query.items() if key.lower() not in drop_keys}
        query_str = urlencode(cleaned_query, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", query_str, ""))

    def _file_stem_key(self, image_url: str) -> str:
        path = urlparse(image_url).path.lower()
        filename = path.rsplit("/", 1)[-1]
        filename = re.sub(r"[-_]?\d{2,4}x\d{2,4}", "", filename)
        filename = re.sub(r"[^a-z0-9.]+", "", filename)
        return filename

    def _candidate_strength(self, candidate: SearchCandidate) -> float:
        discovery = float(candidate.discovery_score or 0.0)
        rank = float(candidate.pre_rank_score or 0.0)
        validity = 0.5 if candidate.is_valid_image else 0.0
        size_bonus = 0.0
        if candidate.width and candidate.height:
            size_bonus = min((candidate.width * candidate.height) / 500000.0, 3.0)
        return discovery + rank + validity + size_bonus

    def _choose_canonical(self, members: list[SearchCandidate]) -> SearchCandidate:
        if not members:
            raise ValueError("members must not be empty")
        return max(members, key=self._candidate_strength)

    def cluster_candidates(self, candidates: list[SearchCandidate], fingerprint_top_n: int = 8) -> list[SearchCandidate]:
        if not candidates:
            return []
        grouped: dict[str, list[SearchCandidate]] = {}
        for candidate in candidates:
            normalized = self._normalize_image_url(candidate.image_url)
            candidate.normalized_image_url = normalized
            page_host = urlparse(candidate.source_url).netloc.lower()
            stem = self._file_stem_key(candidate.image_url)
            grouped.setdefault(f"{page_host}|{stem or normalized}", []).append(candidate)

        clusters = [CandidateCluster(canonical=self._choose_canonical(members), members=members) for members in grouped.values()]
        clusters.sort(key=lambda cluster: self._candidate_strength(cluster.canonical), reverse=True)

        merged: list[CandidateCluster] = []
        fp_map: dict[str, CandidateCluster] = {}
        for index, cluster in enumerate(clusters):
            if index >= max(1, min(fingerprint_top_n, len(clusters))):
                merged.append(cluster)
                continue
            canonical = cluster.canonical
            try:
                content, _ = self.download_image(canonical.image_url)
                canonical.image_fingerprint = hashlib.sha256(content).hexdigest()
            except Exception:
                canonical.image_fingerprint = None
            if not canonical.image_fingerprint:
                merged.append(cluster)
                continue
            existing = fp_map.get(canonical.image_fingerprint)
            if existing is None:
                fp_map[canonical.image_fingerprint] = cluster
                merged.append(cluster)
            else:
                existing.members.extend(cluster.members)
                existing.canonical = self._choose_canonical([existing.canonical, cluster.canonical])

        finalized: list[SearchCandidate] = []
        for cluster in merged:
            canonical = self._choose_canonical(cluster.members)
            domains = sorted({item.source_domain for item in cluster.members if item.source_domain})[:6]
            source_types = sorted({item.source_type for item in cluster.members if item.source_type})[:6]
            canonical.merged_count = len(cluster.members)
            canonical.supporting_source_domains = domains
            canonical.supporting_source_types = source_types
            canonical.content_deduped = any(item.image_fingerprint for item in cluster.members if item is not canonical)
            canonical.cluster_evidence_summary = f"merged={len(cluster.members)}, sources={len(domains)}, types={','.join(source_types[:3]) or 'unknown'}"
            finalized.append(canonical)
        return finalized

    def download_image(self, url: str) -> tuple[bytes, str]:
        resp = self._http.request("GET", url, stream=False)
        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        mime = content_type if content_type else _guess_mime_from_url(url)
        return resp.content, mime
