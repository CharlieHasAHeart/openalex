from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import parse_qs, quote_plus, urljoin, urlparse

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord


@dataclass(slots=True)
class SearchCandidate:
    image_url: str
    source_url: str
    title: str
    snippet: str
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


def _extract_uddg(href: str) -> str:
    parsed = urlparse(href)
    query = parse_qs(parsed.query)
    url = query.get("uddg", [""])[0]
    if url:
        return unescape(url)
    return href


class WebSearchClient:
    def __init__(self, http: HttpClient, max_results: int = 8) -> None:
        self._http = http
        self._max_results = max(1, max_results)

    def _build_query(self, author: AuthorRecord) -> str:
        parts = [author.display_name.strip(), "profile photo"]
        if author.orcid:
            parts.append(author.orcid)
        if author.institution_name:
            parts.append(author.institution_name.strip())
        return " ".join([p for p in parts if p])

    def _parse_search_results(self, html: str) -> list[tuple[str, str, str]]:
        pattern = re.compile(
            r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL,
        )
        items: list[tuple[str, str, str]] = []
        for match in pattern.finditer(html):
            href = unescape(match.group(1))
            title = re.sub(r"<[^>]+>", " ", match.group(2))
            snippet = re.sub(r"<[^>]+>", " ", match.group(3))
            items.append((_extract_uddg(href).strip(), " ".join(title.split()), " ".join(snippet.split())))
            if len(items) >= self._max_results:
                break
        return items

    def _extract_image_candidates(self, source_url: str, page_html: str) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        meta_patterns = [
            r'<meta[^>]+property="og:image"[^>]+content="([^"]+)"',
            r'<meta[^>]+name="twitter:image"[^>]+content="([^"]+)"',
        ]
        for pattern in meta_patterns:
            match = re.search(pattern, page_html, re.IGNORECASE)
            if not match:
                continue
            image_url = urljoin(source_url, unescape(match.group(1)).strip())
            mime = _guess_mime_from_url(image_url)
            candidates.append(SearchCandidate(image_url=image_url, source_url=source_url, title="", snippet="", mime=mime))

        img_pattern = re.compile(r'<img[^>]+src="([^"]+)"', re.IGNORECASE)
        for match in img_pattern.finditer(page_html):
            image_url = urljoin(source_url, unescape(match.group(1)).strip())
            if image_url.startswith("data:"):
                continue
            mime = _guess_mime_from_url(image_url)
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            candidates.append(SearchCandidate(image_url=image_url, source_url=source_url, title="", snippet="", mime=mime))
            if len(candidates) >= 5:
                break
        return candidates

    def search_image_candidates(self, author: AuthorRecord) -> list[SearchCandidate]:
        query = quote_plus(self._build_query(author))
        search_url = f"https://duckduckgo.com/html/?q={query}"
        resp = self._http.request("GET", search_url)
        search_hits = self._parse_search_results(resp.text)

        results: list[SearchCandidate] = []
        for source_url, title, snippet in search_hits:
            try:
                page = self._http.request("GET", source_url)
            except Exception:
                continue
            for item in self._extract_image_candidates(source_url, page.text):
                item.title = title
                item.snippet = snippet
                results.append(item)
                if len(results) >= self._max_results:
                    return results
        return results

    def download_image(self, url: str) -> tuple[bytes, str]:
        resp = self._http.request("GET", url, stream=False)
        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        mime = content_type if content_type else _guess_mime_from_url(url)
        return resp.content, mime
