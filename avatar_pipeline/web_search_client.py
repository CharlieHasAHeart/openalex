from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from html import unescape
from urllib.parse import parse_qs, quote_plus, unquote, urlencode, urljoin, urlparse, urlunparse

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord

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
        return unescape(unquote(url))
    if href.startswith("/l/?"):
        return ""
    return href


class WebSearchClient:
    def __init__(
        self,
        http: HttpClient,
        max_results: int = 8,
        person_page_query_max: int = 7,
        person_page_per_query_results: int = 5,
        person_page_max_fetch: int = 12,
        profile_image_score_threshold: float = 0.35,
    ) -> None:
        self._http = http
        self._max_results = max(1, max_results)
        self._person_page_query_max = max(1, person_page_query_max)
        self._person_page_per_query_results = max(1, person_page_per_query_results)
        self._person_page_max_fetch = max(1, person_page_max_fetch)
        self._profile_image_score_threshold = max(0.0, profile_image_score_threshold)

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
        parsed = self._parse_search_results(resp.text)
        logger.debug("websearch_ddg_parsed query=%r parsed_results=%s", query, len(parsed))
        return parsed[: max(1, limit)]

    def _classify_source_type(self, url: str) -> str:
        lower = url.lower()
        if "orcid.org" in lower:
            return "orcid_profile"
        if any(token in lower for token in ("scholar.google.", "researchgate.net", "academia.edu")):
            return "academic_social"
        if any(token in lower for token in ("/news", "/event", "/article", "/press", "/blog/")):
            return "news_or_article"
        if any(token in lower for token in (".pdf", "/publication", "/publications", "/paper", "/proceedings")):
            return "publication_page"
        if any(token in lower for token in ("/faculty/", "/profile/", "/staff/", "/person/")):
            return "institution_profile"
        if any(token in lower for token in ("/people/", "/directory/", "/member/", "/bio/")):
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
            "youtube.com",
            "youtu.be",
            "doi.org",
            "journal",
            "publisher",
            "citation",
            "/news",
            "/event",
            "/events",
            "/article",
            "/press",
            "/login",
            "/signin",
            "/signup",
            ".pdf",
            "/paper",
            "/proceedings",
            "/archive",
        )
        return any(token in lower for token in noisy)

    def _score_profile_page(self, page: ProfilePageCandidate, author: AuthorRecord | None = None) -> float:
        score = float(page.discovery_score or 0.0)
        domain = (page.source_domain or "").lower()
        path = urlparse(page.page_url).path.lower()
        text_blob = " ".join([page.title or "", page.snippet or "", page.page_url]).lower()
        institution_name = (author.institution_name or "").lower() if author else ""
        display_name = (author.display_name or "").lower() if author else ""
        if domain.endswith(".edu") or ".edu." in domain:
            score += 1.2
        if "orcid.org" in domain:
            score += 1.6
        if any(token in path for token in ("/faculty/", "/profile/", "/staff/", "/person/", "/member/", "/bio/")):
            score += 1.2
        if any(token in path for token in ("/people/", "/directory/", "/team/")):
            score += 0.8
        if any(token in path for token in ("/lab/", "/group/", "/researcher", "/speaker")):
            score += 0.5
        if institution_name:
            inst_tokens = [tok for tok in re.split(r"[^a-z0-9]+", institution_name) if len(tok) >= 4]
            if any(tok in domain for tok in inst_tokens[:5]):
                score += 0.9
        if display_name and display_name in text_blob:
            score += 1.1
        if any(token in text_blob for token in ("professor", "faculty", "researcher", "lab", "university", "institute", "department")):
            score += 0.6
        if "found via orcid external link" in (page.discovery_evidence or "").lower():
            score += 0.8
        if any(token in text_blob for token in ("news", "event", "article", "publication", "paper", "proceedings", "archive", "login")):
            score -= 1.6
        if any(token in domain for token in ("twitter.com", "x.com", "facebook.com", "instagram.com", "linkedin.com")):
            score -= 1.8
        if page.source_type in {"academic_social", "publication_page", "news_or_article"}:
            score -= 1.0
        if page.source_type == "generic_search_result":
            score -= 0.2
        return score

    def _normalize_image_url(self, image_url: str) -> str:
        parsed = urlparse(image_url)
        query = parse_qs(parsed.query)
        drop_keys = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "cache", "cb", "v", "version", "width", "height", "w", "h", "size"}
        cleaned_query: dict[str, list[str]] = {}
        for key, values in query.items():
            if key.lower() in drop_keys:
                continue
            cleaned_query[key] = values
        query_str = urlencode(cleaned_query, doseq=True)
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", query_str, ""))

    def _file_stem_key(self, image_url: str) -> str:
        path = urlparse(image_url).path.lower()
        filename = path.rsplit("/", 1)[-1]
        filename = re.sub(r"[-_]?\\d{2,4}x\\d{2,4}", "", filename)
        filename = re.sub(r"[^a-z0-9.]+", "", filename)
        return filename

    def _candidate_strength(self, candidate: SearchCandidate) -> float:
        structure = float(candidate.page_image_position_score or 0.0) + float(candidate.name_proximity_score or 0.0)
        discovery = float(candidate.discovery_score or 0.0)
        size_bonus = 0.0
        if candidate.width and candidate.height:
            size_bonus = min((candidate.width * candidate.height) / 500000.0, 3.0)
        validity = 0.5 if candidate.is_valid_image else 0.0
        return structure + discovery + size_bonus + validity

    def _choose_canonical(self, members: list[SearchCandidate]) -> SearchCandidate:
        if not members:
            raise ValueError("members must not be empty")
        return max(members, key=self._candidate_strength)

    def cluster_candidates(self, candidates: list[SearchCandidate], fingerprint_top_n: int = 8) -> list[SearchCandidate]:
        if not candidates:
            return candidates
        # 1) URL-level + same-page-size-variant grouping.
        grouped: dict[str, list[SearchCandidate]] = {}
        for c in candidates:
            normalized = self._normalize_image_url(c.image_url)
            c.normalized_image_url = normalized
            page_host = urlparse(c.source_url).netloc.lower()
            stem = self._file_stem_key(c.image_url)
            group_key = f"{page_host}|{stem or normalized}"
            grouped.setdefault(group_key, []).append(c)

        clusters: list[CandidateCluster] = []
        for members in grouped.values():
            canonical = self._choose_canonical(members)
            clusters.append(CandidateCluster(canonical=canonical, members=members))

        # 2) Limited content-level dedupe for high-priority candidates.
        clusters.sort(key=lambda cc: self._candidate_strength(cc.canonical), reverse=True)
        top = clusters[: max(1, min(fingerprint_top_n, len(clusters)))]
        fp_map: dict[str, CandidateCluster] = {}
        merged_by_fp: list[CandidateCluster] = []
        for cluster in top:
            c = cluster.canonical
            try:
                content, _ = self.download_image(c.image_url)
                c.image_fingerprint = hashlib.sha256(content).hexdigest()
            except Exception:
                c.image_fingerprint = None
            fp = c.image_fingerprint
            if not fp:
                merged_by_fp.append(cluster)
                continue
            existing = fp_map.get(fp)
            if existing is None:
                fp_map[fp] = cluster
                merged_by_fp.append(cluster)
            else:
                existing.members.extend(cluster.members)
                # keep stronger canonical
                existing.canonical = self._choose_canonical([existing.canonical, cluster.canonical])

        if len(clusters) > len(top):
            merged_by_fp.extend(clusters[len(top):])

        # 3) finalize canonical cluster metadata
        finalized: list[SearchCandidate] = []
        for cluster in merged_by_fp:
            canonical = self._choose_canonical(cluster.members)
            domains = sorted({(m.source_domain or "") for m in cluster.members if m.source_domain})[:6]
            source_types = sorted({(m.source_type or "") for m in cluster.members if m.source_type})[:6]
            canonical.merged_count = len(cluster.members)
            canonical.supporting_source_domains = domains
            canonical.supporting_source_types = source_types
            canonical.content_deduped = any(m.image_fingerprint for m in cluster.members if m is not canonical)
            canonical.cluster_evidence_summary = (
                f"merged={len(cluster.members)}, sources={len(domains)}, types={','.join(source_types[:3]) or 'unknown'}"
            )
            finalized.append(canonical)
        return finalized

    def _dedupe_profile_pages(self, pages: list[ProfilePageCandidate], author: AuthorRecord | None = None) -> list[ProfilePageCandidate]:
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
        result.sort(key=lambda p: self._score_profile_page(p, author=author), reverse=True)
        return result

    def _extract_institution_domains(self, author: AuthorRecord) -> list[str]:
        domains: set[str] = set()
        profile = author.profile if isinstance(author.profile, dict) else {}
        stack: list[object] = [profile]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                for key, value in node.items():
                    if isinstance(value, (dict, list)):
                        stack.append(value)
                    elif isinstance(value, str):
                        key_lower = key.lower()
                        if key_lower in {"homepage_url", "website", "url", "homepage"} or value.startswith("http"):
                            parsed = urlparse(value)
                            if parsed.hostname:
                                domains.add(parsed.hostname.lower())
            elif isinstance(node, list):
                stack.extend(node)
        return sorted(domains)

    def _build_discovery_queries(self, author: AuthorRecord) -> list[tuple[str, str]]:
        name = (author.display_name or "").strip()
        institution = (author.institution_name or "").strip()
        if not name:
            return []

        queries: list[tuple[str, str]] = []
        if institution:
            for suffix in ("faculty", "profile", "people", "staff", "researcher"):
                queries.append((f'"{name}" "{institution}" {suffix}', f"institution_query:{suffix}"))
        else:
            for suffix in ("faculty", "profile", "people", "staff", "researcher"):
                queries.append((f'"{name}" {suffix}', f"name_only_query:{suffix}"))

        if author.orcid:
            queries.append((f'"{name}" orcid', "orcid_query"))
        queries.append((f'"{name}" "profile photo"', "fallback_profile_photo"))

        for domain in self._extract_institution_domains(author)[:2]:
            queries.insert(0, (f'site:{domain} "{name}" profile', f"domain_hint_query:{domain}"))
            queries.insert(1, (f'site:{domain} "{name}" faculty', f"domain_hint_query:{domain}"))

        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for query, evidence in queries:
            key = query.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append((query, evidence))
        return unique[: self._person_page_query_max]

    def discover_profile_pages(self, author: AuthorRecord) -> list[ProfilePageCandidate]:
        pages: list[ProfilePageCandidate] = []
        reason_tags: set[str] = set()
        queries = self._build_discovery_queries(author)
        logger.info(
            "websearch_discovery_queries author_id=%s display_name=%r queries=%s",
            author.author_id,
            author.display_name,
            [q for q, _ in queries],
        )

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
                found_external = 0
                for href in re.findall(r'href=["\']([^"\']+)["\']', orcid_resp.text, re.IGNORECASE):
                    link = unescape(href).strip()
                    if not link.startswith("http") or "orcid.org" in link or self._is_noise_url(link):
                        continue
                    source_type = self._classify_source_type(link)
                    if source_type == "generic_search_result" and not (
                        ".edu" in link.lower()
                        or any(k in link.lower() for k in ("university", "institute", "research", "/profile/", "/people/", "/faculty/", "/staff/"))
                    ):
                        continue
                    found_external += 1
                    pages.append(
                        ProfilePageCandidate(
                            page_url=link,
                            source_domain=urlparse(link).hostname,
                            source_type=source_type,
                            discovery_score=5.0 if source_type != "generic_search_result" else 3.2,
                            discovery_evidence="found via orcid external link",
                        )
                    )
                logger.info(
                    "websearch_orcid_external_links author_id=%s external_candidates=%s",
                    author.author_id,
                    found_external,
                )
            except Exception as exc:
                reason_tags.add("page_fetch_failed")
                logger.debug(
                    "websearch_orcid_fetch_failed author_id=%s orcid=%s error_message=%s",
                    author.author_id,
                    author.orcid,
                    str(exc),
                )

        # Generic fallback path must always remain available.
        if not queries:
            queries = [(self._build_query(author), "legacy_fallback_query")]
        for query, query_source in queries:
            try:
                search_rows = self._search_pages(query, max_results=self._person_page_per_query_results)
                logger.info(
                    "websearch_query_result_count author_id=%s query_source=%s results=%s query=%r",
                    author.author_id,
                    query_source,
                    len(search_rows),
                    query,
                )
                if not search_rows:
                    reason_tags.add("no_search_results")
            except Exception as exc:
                reason_tags.add("parse_failed")
                logger.debug(
                    "websearch_query_failed author_id=%s query_source=%s query=%r error_message=%s",
                    author.author_id,
                    query_source,
                    query,
                    str(exc),
                )
                continue
            for url, title, snippet in search_rows:
                if self._is_noise_url(url):
                    continue
                source_type = self._classify_source_type(url)
                if source_type in {"news_or_article", "publication_page"}:
                    continue
                score = 1.4 if source_type == "institution_profile" else 1.1
                if source_type == "orcid_profile":
                    score = 2.8
                pages.append(
                    ProfilePageCandidate(
                        page_url=url,
                        source_domain=urlparse(url).hostname,
                        source_type=source_type,
                        discovery_score=score,
                        discovery_evidence=f"{query_source}; query={query}",
                        title=title,
                        snippet=snippet,
                    )
                )

        deduped = self._dedupe_profile_pages(pages, author=author)
        kept = deduped[: max(self._person_page_max_fetch, self._max_results, 5)]
        logger.info(
            "websearch_discover_profile_pages author_id=%s total_raw=%s kept=%s reasons=%s",
            author.author_id,
            len(pages),
            len(kept),
            ",".join(sorted(reason_tags)) if reason_tags else "ok",
        )
        if not kept:
            logger.info("websearch_discover_profile_pages_zero author_id=%s reason=no_profile_pages", author.author_id)
        return kept

    def _parse_search_results(self, html: str) -> list[tuple[str, str, str]]:
        items: list[tuple[str, str, str]] = []
        seen_urls: set[str] = set()

        block_pattern = re.compile(r'<div[^>]*class=["\'][^"\']*result[^"\']*["\'][^>]*>(.*?)</div>', re.IGNORECASE | re.DOTALL)
        candidate_blocks = [m.group(1) for m in block_pattern.finditer(html)]
        if not candidate_blocks:
            candidate_blocks = [html]

        anchor_pattern = re.compile(r"<a([^>]+)href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
        snippet_pattern = re.compile(
            r'<(?:a|div|span)[^>]*class=["\'][^"\']*(?:result__snippet|snippet|result-snippet)[^"\']*["\'][^>]*>(.*?)</(?:a|div|span)>',
            re.IGNORECASE | re.DOTALL,
        )
        for block in candidate_blocks:
            snippet_match = snippet_pattern.search(block)
            snippet = self._clean_text(snippet_match.group(1)) if snippet_match else ""
            for anchor_match in anchor_pattern.finditer(block):
                attrs = anchor_match.group(1).lower()
                href = unescape(anchor_match.group(2)).strip()
                title = self._clean_text(anchor_match.group(3)) or ""
                if "result__a" not in attrs and "result-link" not in attrs and "uddg=" not in href:
                    continue
                resolved = _extract_uddg(href).strip()
                if not resolved or not resolved.startswith("http"):
                    continue
                if resolved in seen_urls:
                    continue
                seen_urls.add(resolved)
                items.append((resolved, title, snippet))
                break

        if not items:
            # fallback: keep urls even when title/snippet parsing fails
            for href in re.findall(r'href=["\']([^"\']+)["\']', html, re.IGNORECASE):
                resolved = _extract_uddg(unescape(href).strip())
                if not resolved.startswith("http") or resolved in seen_urls:
                    continue
                if "duckduckgo.com" in resolved.lower():
                    continue
                seen_urls.add(resolved)
                items.append((resolved, "", ""))
                if len(items) >= self._max_results:
                    break
            if not items:
                logger.debug("websearch_ddg_parse_failed reason=no_result_links")
        return items

    def _extract_image_candidates(self, source_url: str, page_html: str) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        img_pattern = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"']", re.IGNORECASE)
        for match in img_pattern.finditer(page_html):
            image_url = urljoin(source_url, unescape(match.group(1)).strip())
            if image_url.startswith("data:"):
                continue
            if self._is_decorative_image(image_url):
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

    def _append_structure_evidence(self, candidate: SearchCandidate, marker: str) -> None:
        current = candidate.structure_evidence or ""
        if marker in current:
            return
        candidate.structure_evidence = marker if not current else f"{current}; {marker}"

    def _is_decorative_image(self, text: str) -> bool:
        lower = text.lower()
        return any(token in lower for token in ("logo", "banner", "hero", "icon", "sprite", "default", "placeholder", "favicon", "social"))

    def _is_avatarish_image(self, text: str) -> bool:
        lower = text.lower()
        return any(token in lower for token in ("headshot", "portrait", "profile", "avatar", "people", "faculty", "staff", "member"))

    def _extract_jsonld_person_images(self, source_url: str, page_html: str, author: AuthorRecord) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        script_pattern = re.compile(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', re.IGNORECASE | re.DOTALL)
        display_name = (author.display_name or "").strip()

        def _iter_nodes(node: object):
            if isinstance(node, dict):
                yield node
                for value in node.values():
                    yield from _iter_nodes(value)
            elif isinstance(node, list):
                for item in node:
                    yield from _iter_nodes(item)

        def _is_person_node(node: dict) -> bool:
            typ = node.get("@type")
            if isinstance(typ, str) and "person" in typ.lower():
                return True
            if isinstance(typ, list) and any(isinstance(t, str) and "person" in t.lower() for t in typ):
                return True
            return False

        def _extract_image_urls(value: object) -> list[str]:
            urls: list[str] = []
            if isinstance(value, str):
                urls.append(value)
            elif isinstance(value, dict):
                for key in ("url", "contentUrl", "@id", "image"):
                    raw = value.get(key)
                    if isinstance(raw, str) and raw.strip():
                        urls.append(raw)
            elif isinstance(value, list):
                for item in value:
                    urls.extend(_extract_image_urls(item))
            return urls

        for match in script_pattern.finditer(page_html):
            raw_json = match.group(1).strip()
            if not raw_json:
                continue
            try:
                payload = json.loads(unescape(raw_json))
            except Exception:
                continue
            for node in _iter_nodes(payload):
                if not isinstance(node, dict):
                    continue
                if not _is_person_node(node):
                    continue
                person_name = str(node.get("name") or "").strip()
                person_name_score = self._name_match_strength(person_name, display_name)
                person_image = node.get("image")
                if person_image is None:
                    continue
                for raw_url in _extract_image_urls(person_image):
                    image_url = urljoin(source_url, unescape(raw_url).strip())
                    if not image_url.startswith("http"):
                        continue
                    mime = _guess_mime_from_url(image_url)
                    item = SearchCandidate(
                        image_url=image_url,
                        source_url=source_url,
                        title="",
                        snippet="",
                        mime=mime,
                        source_domain=urlparse(source_url).hostname,
                        image_alt=person_name or None,
                        page_image_role="jsonld_person_image",
                        page_image_position_score=1.45,
                        name_proximity_score=max(0.55, person_name_score),
                        context_block_type="jsonld_person",
                        structure_evidence="jsonld_person_image",
                    )
                    candidates.append(item)
        return candidates

    def _extract_og_image_candidates(self, source_url: str, page_html: str) -> list[SearchCandidate]:
        candidates: list[SearchCandidate] = []
        meta_pattern = re.compile(
            r'<meta[^>]+(?:property|name)=["\']([^"\']+)["\'][^>]+content=["\']([^"\']+)["\'][^>]*>',
            re.IGNORECASE,
        )
        og_map: dict[str, str] = {}
        for key, value in meta_pattern.findall(page_html):
            lower_key = key.strip().lower()
            og_map[lower_key] = unescape(value).strip()
        image_url = og_map.get("og:image") or og_map.get("twitter:image")
        if not image_url:
            return candidates
        resolved = urljoin(source_url, image_url)
        mime = og_map.get("og:image:type") or _guess_mime_from_url(resolved)
        width = None
        height = None
        try:
            width = int(og_map.get("og:image:width", ""))
        except Exception:
            width = None
        try:
            height = int(og_map.get("og:image:height", ""))
        except Exception:
            height = None
        evidence_parts = ["og:image"]
        if og_map.get("og:image:alt"):
            evidence_parts.append("og:image:alt")
        if width and height:
            evidence_parts.append("og:image:size")
        if og_map.get("og:image:type"):
            evidence_parts.append("og:image:type")
        candidates.append(
            SearchCandidate(
                image_url=resolved,
                source_url=source_url,
                title="",
                snippet="",
                mime=mime,
                source_domain=urlparse(source_url).hostname,
                image_alt=self._clean_text(og_map.get("og:image:alt")),
                width=width,
                height=height,
                page_image_role="og_image",
                page_image_position_score=1.15,
                name_proximity_score=0.25,
                context_block_type="open_graph",
                structure_evidence=", ".join(evidence_parts),
            )
        )
        return candidates

    def _refine_structured_candidate(self, candidate: SearchCandidate, author: AuthorRecord) -> SearchCandidate:
        display_name = author.display_name or ""
        institution_name = author.institution_name or ""
        context_text = " ".join(
            [
                candidate.image_alt or "",
                candidate.nearby_text or "",
                candidate.structure_evidence or "",
                candidate.page_image_role or "",
                candidate.context_block_type or "",
                candidate.image_url,
            ]
        )
        name_score = self._name_match_strength(context_text, display_name)
        candidate.name_proximity_score = max(float(candidate.name_proximity_score or 0.0), name_score)
        institution_score = self._name_match_strength(context_text, institution_name) if institution_name else 0.0
        candidate.institution_match_score = max(float(candidate.institution_match_score or 0.0), institution_score)

        structure_score = float(candidate.page_image_position_score or 0.0)
        if self._is_avatarish_image(context_text):
            structure_score += 0.45
            self._append_structure_evidence(candidate, "avatarish_context")
            if candidate.page_image_role in {None, "generic_page_image"}:
                candidate.page_image_role = "likely_headshot"
        if self._is_decorative_image(context_text):
            structure_score -= 0.9
            self._append_structure_evidence(candidate, "decorative_context")
            candidate.page_image_role = "decorative_or_logo"
        if candidate.context_block_type in {"profile_header", "people_card", "faculty_card"}:
            structure_score += 0.35
        if candidate.context_block_type == "jsonld_person":
            structure_score += 0.7
        if candidate.context_block_type == "open_graph":
            structure_score += 0.3

        candidate.page_image_position_score = structure_score
        candidate.image_precheck_score = structure_score + float(candidate.name_proximity_score or 0.0) + 0.6 * float(candidate.institution_match_score or 0.0)
        return candidate

    def _structured_extract_image_candidates_with_stats(
        self,
        source_url: str,
        page_html: str,
        author: AuthorRecord,
    ) -> tuple[list[SearchCandidate], dict[str, int]]:
        lowered_html = page_html.lower()
        display_name = author.display_name or ""
        path = urlparse(source_url).path.lower()
        is_profile_like = any(token in (path + " " + lowered_html[:2500]) for token in ("faculty", "profile", "staff", "person", "researcher", "bio"))
        is_people_like = any(token in (path + " " + lowered_html[:2500]) for token in ("people", "directory", "team", "group", "member", "speaker"))
        stats = {"jsonld": 0, "og": 0, "blocks": 0, "fallback": 0, "kept": 0}

        collected: list[SearchCandidate] = []

        jsonld_candidates = self._extract_jsonld_person_images(source_url, page_html, author)
        stats["jsonld"] = len(jsonld_candidates)
        collected.extend(jsonld_candidates)

        og_candidates = self._extract_og_image_candidates(source_url, page_html)
        stats["og"] = len(og_candidates)
        collected.extend(og_candidates)

        block_pattern = re.compile(r"<(div|section|article|li)[^>]*>(.*?)</\\1>", re.IGNORECASE | re.DOTALL)
        keyword = re.compile(r"(profile|faculty|staff|people|member|bio|researcher|speaker|person|team|group|directory)", re.IGNORECASE)
        block_count = 0
        for block_match in block_pattern.finditer(page_html):
            block = block_match.group(0)
            if len(block) > 15000 or not keyword.search(block):
                continue
            name_score = self._name_match_strength(block, display_name)
            if name_score < 0.25:
                continue
            block_type = "people_card" if is_people_like else ("faculty_card" if is_profile_like else "name_matched_block")
            block_items = self._extract_images_from_block(
                block,
                source_url,
                role="card_headshot",
                block_type=block_type,
                position_score=0.85,
                name_proximity=name_score,
                structure_evidence="name matched card-like block",
            )
            block_count += len(block_items)
            collected.extend(block_items)
            if len(collected) >= 12:
                break
        stats["blocks"] = block_count

        if len(collected) < 8 and is_profile_like:
            top_slice = page_html[: max(1800, len(page_html) // 3)]
            top_items = self._extract_images_from_block(
                top_slice,
                source_url,
                role="profile_headshot",
                block_type="profile_header",
                position_score=0.95,
                name_proximity=self._name_match_strength(top_slice, display_name),
                structure_evidence="profile-like page top/header region",
            )
            stats["blocks"] += len(top_items)
            collected.extend(top_items)

        generic = self._extract_image_candidates(source_url, page_html)
        stats["fallback"] = len(generic)
        for item in generic:
            item.page_image_role = item.page_image_role or "generic_page_image"
            item.page_image_position_score = item.page_image_position_score if item.page_image_position_score is not None else 0.25
            item.name_proximity_score = item.name_proximity_score if item.name_proximity_score is not None else self._name_match_strength(page_html[:2500], display_name)
            item.context_block_type = item.context_block_type or "generic_page"
            item.structure_evidence = item.structure_evidence or "generic image extraction fallback"
        collected.extend(generic)

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

        refined: list[SearchCandidate] = []
        for item in dedup.values():
            self._refine_structured_candidate(item, author)
            if (item.image_precheck_score or 0.0) < self._profile_image_score_threshold:
                continue
            if item.page_image_role == "decorative_or_logo":
                continue
            refined.append(item)

        refined.sort(
            key=lambda c: (float(c.image_precheck_score or 0.0), float(c.page_image_position_score or 0.0) + float(c.name_proximity_score or 0.0)),
            reverse=True,
        )
        limited = refined[:8]
        stats["kept"] = len(limited)
        return limited, stats

    def _structured_extract_image_candidates(
        self,
        source_url: str,
        page_html: str,
        author: AuthorRecord,
    ) -> list[SearchCandidate]:
        candidates, _ = self._structured_extract_image_candidates_with_stats(source_url, page_html, author)
        return candidates

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
                candidate.image_fingerprint = hashlib.sha256(content).hexdigest() if content else None
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
        if not profile_pages:
            logger.info("websearch_search_image_candidates author_id=%s reason=no_profile_pages candidates=0", author.author_id)
            return []
        results: list[SearchCandidate] = []
        reason_tags: set[str] = set()
        parse_failures = 0
        for page in profile_pages:
            try:
                page_resp = self._http.request("GET", page.page_url)
            except Exception as exc:
                reason_tags.add("page_fetch_failed")
                logger.debug(
                    "websearch_page_fetch_failed author_id=%s page_url=%s error_message=%s",
                    author.author_id,
                    page.page_url,
                    str(exc),
                )
                continue
            try:
                extracted, stats = self._structured_extract_image_candidates_with_stats(page.page_url, page_resp.text, author)
            except Exception as exc:
                parse_failures += 1
                reason_tags.add("parse_failed")
                logger.debug(
                    "websearch_page_parse_failed author_id=%s page_url=%s error_message=%s",
                    author.author_id,
                    page.page_url,
                    str(exc),
                )
                continue
            logger.info(
                "websearch_page_extract_stats author_id=%s page_url=%s jsonld=%s og=%s blocks=%s fallback=%s kept=%s",
                author.author_id,
                page.page_url,
                stats.get("jsonld", 0),
                stats.get("og", 0),
                stats.get("blocks", 0),
                stats.get("fallback", 0),
                stats.get("kept", 0),
            )
            if not extracted:
                reason_tags.add("no_structured_images")
            for item in extracted:
                item.title = page.title or ""
                item.snippet = page.snippet or ""
                item.page_title = page.title or item.page_title
                item.source_type = page.source_type
                item.discovery_score = self._score_profile_page(page, author=author)
                item.discovery_evidence = page.discovery_evidence
                results.append(item)
                if len(results) >= self._max_results:
                    logger.info(
                        "websearch_search_image_candidates author_id=%s candidates=%s reason=ok_early_stop",
                        author.author_id,
                        len(results),
                    )
                    return results
        if parse_failures and parse_failures == len(profile_pages):
            reason_tags.add("parse_failed")
        if not results and not reason_tags:
            reason_tags.add("all_images_filtered")
        logger.info(
            "websearch_search_image_candidates author_id=%s profile_pages=%s candidates=%s reasons=%s",
            author.author_id,
            len(profile_pages),
            len(results),
            ",".join(sorted(reason_tags)) if reason_tags else "ok",
        )
        return results

    def download_image(self, url: str) -> tuple[bytes, str]:
        resp = self._http.request("GET", url, stream=False)
        content_type = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        mime = content_type if content_type else _guess_mime_from_url(url)
        return resp.content, mime
