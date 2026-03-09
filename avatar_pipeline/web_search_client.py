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
    page_h1: str | None = None
    page_meta_description: str | None = None
    image_alt: str | None = None
    nearby_text: str | None = None
    name_match_score: float | None = None
    institution_match_score: float | None = None
    source_trust_score: float | None = None
    pre_rank_score: float | None = None
    source_domain: str | None = None
    page_title: str | None = None
    width: int | None = None
    height: int | None = None
    size_bytes: int | None = None
    face_count: int | None = None
    is_portrait: bool | None = None
    is_valid_image: bool | None = None
    invalid_reason: str | None = None
    image_precheck_score: float | None = None
    source_type: str | None = None
    discovery_score: float | None = None
    discovery_evidence: str | None = None
    page_image_role: str | None = None
    page_image_position_score: float | None = None
    name_proximity_score: float | None = None
    context_block_type: str | None = None
    structure_evidence: str | None = None


@dataclass(slots=True)
class ProfilePageCandidate:
    page_url: str
    source_domain: str | None
    source_type: str
    discovery_score: float | None = None
    discovery_evidence: str | None = None
    title: str | None = None
    snippet: str | None = None


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

    def _search_pages(self, query: str, max_results: int | None = None) -> list[tuple[str, str, str]]:
        limit = max_results if max_results is not None else self._max_results
        q = quote_plus(query)
        search_url = f"https://duckduckgo.com/html/?q={q}"
        resp = self._http.request("GET", search_url)
        return self._parse_search_results(resp.text)[: max(1, limit)]

    def _classify_source_type(self, url: str) -> str:
        lower = url.lower()
        if "orcid.org" in lower:
            return "orcid_profile"
        if any(token in lower for token in ("/faculty/", "/profile/", "/staff/", "/person/")):
            return "institution_profile"
        if any(token in lower for token in ("/people/", "/directory/")):
            return "institution_directory"
        if any(token in lower for token in ("/lab/", "/group/", "/team/")):
            return "lab_people_page"
        return "generic_search_result"

    def _name_tokens(self, name: str) -> list[str]:
        return [tok for tok in re.split(r"\s+", name.lower()) if tok and len(tok) > 1]

    def _name_match_strength(self, text: str, display_name: str) -> float:
        lower = text.lower()
        name = display_name.strip().lower()
        if not name:
            return 0.0
        if name in lower:
            return 1.0
        tokens = self._name_tokens(name)
        if len(tokens) >= 2 and all(tok in lower for tok in tokens[:2]):
            return 0.7
        if any(tok in lower for tok in tokens):
            return 0.35
        return 0.0

    def _is_noise_url(self, link: str) -> bool:
        lower = link.lower()
        if lower.startswith(("mailto:", "javascript:")):
            return True
        noisy = (
            "login",
            "auth",
            "signin",
            "signup",
            "twitter.com",
            "x.com",
            "facebook.com",
            "instagram.com",
            "linkedin.com",
            "doi.org",
            "journal",
            "publisher",
            "citation",
        )
        return any(token in lower for token in noisy)

    def _score_profile_page(self, page: ProfilePageCandidate) -> float:
        score = float(page.discovery_score or 0.0)
        domain = (page.source_domain or "").lower()
        path = urlparse(page.page_url).path.lower()
        if domain.endswith(".edu") or ".edu." in domain:
            score += 1.2
        if "orcid.org" in domain:
            score += 1.6
        if any(token in path for token in ("/faculty/", "/profile/", "/staff/", "/person/")):
            score += 1.2
        if any(token in path for token in ("/people/", "/directory/")):
            score += 0.8
        if any(token in path for token in ("/lab/", "/group/", "/team/")):
            score += 0.5
        if page.source_type == "generic_search_result":
            score -= 0.2
        return score

    def _dedupe_profile_pages(self, pages: list[ProfilePageCandidate]) -> list[ProfilePageCandidate]:
        deduped: dict[str, ProfilePageCandidate] = {}
        for item in pages:
            parsed = urlparse(item.page_url)
            key = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
            existing = deduped.get(key)
            if existing is None:
                deduped[key] = item
                continue
            if (item.discovery_score or 0.0) > (existing.discovery_score or 0.0):
                deduped[key] = item
        result = list(deduped.values())
        result.sort(key=self._score_profile_page, reverse=True)
        return result

    def discover_profile_pages(self, author: AuthorRecord) -> list[ProfilePageCandidate]:
        pages: list[ProfilePageCandidate] = []

        if author.orcid:
            orcid_url = f"https://orcid.org/{author.orcid}"
            pages.append(
                ProfilePageCandidate(
                    page_url=orcid_url,
                    source_domain="orcid.org",
                    source_type="orcid_profile",
                    discovery_score=3.0,
                    discovery_evidence="orcid profile url",
                )
            )
            try:
                orcid_resp = self._http.request("GET", orcid_url)
                for href in re.findall(r'href=["\']([^"\']+)["\']', orcid_resp.text, re.IGNORECASE):
                    link = unescape(href).strip()
                    if not link.startswith("http"):
                        continue
                    if "orcid.org" in link:
                        continue
                    if self._is_noise_url(link):
                        continue
                    source_type = self._classify_source_type(link)
                    if source_type == "generic_search_result":
                        if not (
                            ".edu" in link.lower()
                            or any(k in link.lower() for k in ("university", "institute", "research", "/profile/", "/people/", "/faculty/", "/staff/"))
                        ):
                            continue
                    pages.append(
                        ProfilePageCandidate(
                            page_url=link,
                            source_domain=urlparse(link).hostname,
                            source_type=source_type,
                            discovery_score=5.0 if source_type != "generic_search_result" else 3.2,
                            discovery_evidence="found via orcid external link",
                        )
                    )
            except Exception:
                pass

        if author.institution_name:
            institution_queries = [
                f'"{author.display_name}" "{author.institution_name}" faculty',
                f'"{author.display_name}" "{author.institution_name}" profile',
                f'"{author.display_name}" "{author.institution_name}" people',
            ]
            for query in institution_queries:
                try:
                    for url, title, snippet in self._search_pages(query, max_results=5):
                        source_type = self._classify_source_type(url)
                        if source_type == "generic_search_result":
                            continue
                        pages.append(
                            ProfilePageCandidate(
                                page_url=url,
                                source_domain=urlparse(url).hostname,
                                source_type=source_type,
                                discovery_score=3.2 if source_type == "institution_profile" else 2.6,
                                discovery_evidence=f"institution-oriented query: {query}",
                                title=title,
                                snippet=snippet,
                            )
                        )
                except Exception:
                    continue

        # Generic fallback path must always remain available.
        generic_query = self._build_query(author)
        try:
            for url, title, snippet in self._search_pages(generic_query, max_results=self._max_results):
                pages.append(
                    ProfilePageCandidate(
                        page_url=url,
                        source_domain=urlparse(url).hostname,
                        source_type="generic_search_result",
                        discovery_score=1.0,
                        discovery_evidence="generic fallback search",
                        title=title,
                        snippet=snippet,
                    )
                )
        except Exception:
            pass

        deduped = self._dedupe_profile_pages(pages)
        return deduped[: max(self._max_results, 5)]

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
            candidates.append(
                SearchCandidate(
                    image_url=image_url,
                    source_url=source_url,
                    title="",
                    snippet="",
                    mime=mime,
                    source_domain=urlparse(source_url).hostname,
                )
            )

        img_pattern = re.compile(r'<img[^>]+src="([^"]+)"', re.IGNORECASE)
        for match in img_pattern.finditer(page_html):
            image_url = urljoin(source_url, unescape(match.group(1)).strip())
            if image_url.startswith("data:"):
                continue
            mime = _guess_mime_from_url(image_url)
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            candidates.append(
                SearchCandidate(
                    image_url=image_url,
                    source_url=source_url,
                    title="",
                    snippet="",
                    mime=mime,
                    source_domain=urlparse(source_url).hostname,
                )
            )
            if len(candidates) >= 5:
                break
        return candidates

    def _extract_images_from_block(
        self,
        block_html: str,
        source_url: str,
        role: str,
        block_type: str,
        position_score: float,
        name_proximity: float,
        structure_evidence: str,
    ) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        for match in re.finditer(r"<img[^>]+>", block_html, re.IGNORECASE | re.DOTALL):
            tag = match.group(0)
            src_match = re.search(r'src=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if not src_match:
                continue
            raw_src = unescape(src_match.group(1)).strip()
            if not raw_src or raw_src.startswith("data:"):
                continue
            lower_src = raw_src.lower()
            if any(token in lower_src for token in ("logo", "banner", "sprite", "icon", "favicon", "social")):
                role_value = "decorative_or_logo"
            else:
                role_value = role
            image_url = urljoin(source_url, raw_src)
            mime = _guess_mime_from_url(image_url)
            if mime not in {"image/jpeg", "image/png", "image/webp"}:
                continue
            alt_match = re.search(r'alt=["\']([^"\']*)["\']', tag, re.IGNORECASE)
            alt_text = self._clean_text(alt_match.group(1)) if alt_match else None
            candidates.append(
                SearchCandidate(
                    image_url=image_url,
                    source_url=source_url,
                    title="",
                    snippet="",
                    mime=mime,
                    source_domain=urlparse(source_url).hostname,
                    image_alt=alt_text,
                    page_image_role=role_value,
                    page_image_position_score=position_score,
                    name_proximity_score=name_proximity,
                    context_block_type=block_type,
                    structure_evidence=structure_evidence,
                )
            )
        return candidates

    def _structured_extract_image_candidates(
        self,
        source_url: str,
        page_html: str,
        author: AuthorRecord,
    ) -> list[SearchCandidate]:
        lowered_html = page_html.lower()
        display_name = author.display_name or ""
        path = urlparse(source_url).path.lower()
        is_profile_like = any(token in (path + " " + lowered_html[:2000]) for token in ("faculty", "profile", "staff", "person", "researcher"))
        is_people_like = any(token in (path + " " + lowered_html[:2000]) for token in ("people", "directory", "team", "group", "members"))

        collected: list[SearchCandidate] = []

        # 1) Name-matched card/block extraction.
        block_pattern = re.compile(r"<(div|section|article|li)[^>]*>(.*?)</\\1>", re.IGNORECASE | re.DOTALL)
        keyword = re.compile(r"(people|person|profile|faculty|staff|member|researcher|team|group)", re.IGNORECASE)
        for block_match in block_pattern.finditer(page_html):
            block = block_match.group(0)
            if len(block) > 12000:
                continue
            if not keyword.search(block):
                continue
            name_score = self._name_match_strength(block, display_name)
            if name_score < 0.35:
                continue
            block_type = "people_card" if is_people_like else ("faculty_card" if is_profile_like else "name_matched_block")
            collected.extend(
                self._extract_images_from_block(
                    block,
                    source_url,
                    role="card_headshot",
                    block_type=block_type,
                    position_score=0.85,
                    name_proximity=name_score,
                    structure_evidence="name matched card-like block",
                )
            )
            if len(collected) >= 6:
                break

        # 2) Profile header / top-of-page extraction.
        if len(collected) < 4 and is_profile_like:
            top_slice = page_html[: max(1200, len(page_html) // 3)]
            collected.extend(
                self._extract_images_from_block(
                    top_slice,
                    source_url,
                    role="profile_headshot",
                    block_type="profile_header",
                    position_score=0.9,
                    name_proximity=self._name_match_strength(top_slice, display_name),
                    structure_evidence="profile-like page top/header region",
                )
            )

        # 3) Fallback generic extraction (must remain available).
        if len(collected) < 3:
            generic = self._extract_image_candidates(source_url, page_html)
            for item in generic:
                item.page_image_role = item.page_image_role or "generic_page_image"
                item.page_image_position_score = item.page_image_position_score if item.page_image_position_score is not None else 0.3
                item.name_proximity_score = item.name_proximity_score if item.name_proximity_score is not None else self._name_match_strength(page_html[:2000], display_name)
                item.context_block_type = item.context_block_type or "generic_page"
                item.structure_evidence = item.structure_evidence or "generic image extraction fallback"
            collected.extend(generic)

        # Deduplicate by normalized URL and keep strongest structure score.
        dedup: dict[str, SearchCandidate] = {}
        for item in collected:
            key = item.image_url.split("#")[0]
            existing = dedup.get(key)
            if existing is None:
                dedup[key] = item
                continue
            old_score = float(existing.page_image_position_score or 0.0) + float(existing.name_proximity_score or 0.0)
            new_score = float(item.page_image_position_score or 0.0) + float(item.name_proximity_score or 0.0)
            if new_score > old_score:
                dedup[key] = item

        ranked = list(dedup.values())
        ranked.sort(
            key=lambda c: float(c.page_image_position_score or 0.0) + float(c.name_proximity_score or 0.0),
            reverse=True,
        )
        return ranked[:8]

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
        meta_match = re.search(
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\']([^"\']+)["\']',
            html,
            re.IGNORECASE | re.DOTALL,
        )
        if meta_match:
            meta_description = self._clean_text(meta_match.group(1))

        image_alt = None
        nearby_text = None
        img_iter = re.finditer(r"<img[^>]+>", html, re.IGNORECASE | re.DOTALL)
        image_url_lower = image_url.lower()
        chosen_match = None
        for img_match in img_iter:
            tag = img_match.group(0)
            src_match = re.search(r'src=["\']([^"\']+)["\']', tag, re.IGNORECASE)
            if not src_match:
                continue
            src = unescape(src_match.group(1)).strip().lower()
            if src and (src in image_url_lower or image_url_lower in src):
                chosen_match = img_match
                alt_match = re.search(r'alt=["\']([^"\']*)["\']', tag, re.IGNORECASE)
                if alt_match:
                    image_alt = self._clean_text(alt_match.group(1))
                break

        if chosen_match:
            start = max(chosen_match.start() - 250, 0)
            end = min(chosen_match.end() + 250, len(html))
            nearby_text = self._clean_text(html[start:end])

        if image_alt is None:
            first_alt = re.search(r'<img[^>]+alt=["\']([^"\']*)["\']', html, re.IGNORECASE | re.DOTALL)
            if first_alt:
                image_alt = self._clean_text(first_alt.group(1))

        if nearby_text is None:
            p_match = re.search(r"<p[^>]*>(.*?)</p>", html, re.IGNORECASE | re.DOTALL)
            if p_match:
                nearby_text = self._clean_text(p_match.group(1))

        return {
            "page_title": page_title,
            "page_h1": page_h1,
            "page_meta_description": meta_description,
            "image_alt": image_alt,
            "nearby_text": nearby_text,
        }

    def enrich_candidates_context(self, candidates: list[SearchCandidate], limit: int = 5) -> list[SearchCandidate]:
        if not candidates:
            return candidates
        max_items = max(0, limit)
        for candidate in candidates[:max_items]:
            try:
                page = self._http.request("GET", candidate.source_url)
                context = self._extract_page_context(page.text, candidate.image_url)
                candidate.page_title = context.get("page_title") or candidate.title
                candidate.page_h1 = context.get("page_h1")
                candidate.page_meta_description = context.get("page_meta_description")
                candidate.image_alt = context.get("image_alt")
                candidate.nearby_text = context.get("nearby_text")
            except Exception:
                candidate.page_title = candidate.page_title or candidate.title
        return candidates

    def _parse_png_size(self, content: bytes) -> tuple[int, int] | None:
        if len(content) < 24 or content[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        width = int.from_bytes(content[16:20], "big")
        height = int.from_bytes(content[20:24], "big")
        return width, height

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
                if i + 7 > len(content):
                    break
                height = int.from_bytes(content[i + 3:i + 5], "big")
                width = int.from_bytes(content[i + 5:i + 7], "big")
                return width, height
            i += seg_len
        return None

    def _parse_webp_size(self, content: bytes) -> tuple[int, int] | None:
        if len(content) < 30 or content[0:4] != b"RIFF" or content[8:12] != b"WEBP":
            return None
        chunk = content[12:16]
        if chunk == b"VP8X" and len(content) >= 30:
            width = 1 + int.from_bytes(content[24:27], "little")
            height = 1 + int.from_bytes(content[27:30], "little")
            return width, height
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
        size = parser(content)
        if size is None:
            return None, None
        return size

    def enrich_candidates_image_metadata(
        self,
        candidates: list[SearchCandidate],
        limit: int = 5,
        allowed_mime: set[str] | None = None,
        min_edge_px: int = 200,
    ) -> list[SearchCandidate]:
        if not candidates:
            return candidates
        max_items = max(0, limit)
        allowed = allowed_mime or {"image/jpeg", "image/png", "image/webp"}
        for candidate in candidates[:max_items]:
            try:
                content, mime = self.download_image(candidate.image_url)
                candidate.mime = mime or candidate.mime
                candidate.size_bytes = len(content)
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
                elif candidate.width > 0 and candidate.height > 0 and max(candidate.width / candidate.height, candidate.height / candidate.width) > 4.5:
                    candidate.is_valid_image = False
                    candidate.invalid_reason = "invalid_image_extreme_aspect_ratio"
                else:
                    candidate.is_valid_image = True
                    candidate.invalid_reason = None
            except Exception:
                candidate.is_valid_image = False
                candidate.invalid_reason = "image_metadata_fetch_failed"
        return candidates

    def search_image_candidates(self, author: AuthorRecord) -> list[SearchCandidate]:
        profile_pages = self.discover_profile_pages(author)
        results: list[SearchCandidate] = []
        for page in profile_pages:
            try:
                page_resp = self._http.request("GET", page.page_url)
            except Exception:
                continue
            for item in self._structured_extract_image_candidates(page.page_url, page_resp.text, author):
                item.title = page.title or ""
                item.snippet = page.snippet or ""
                item.page_title = page.title or item.page_title
                item.source_type = page.source_type
                item.discovery_score = self._score_profile_page(page)
                item.discovery_evidence = page.discovery_evidence
                results.append(item)
                if len(results) >= self._max_results:
                    return results
        return results

    def download_image(self, url: str) -> tuple[bytes, str]:
        resp = self._http.request("GET", url, stream=False)
        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        mime = content_type if content_type else _guess_mime_from_url(url)
        return resp.content, mime
