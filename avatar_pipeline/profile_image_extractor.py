from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

from avatar_pipeline.models import AuthorRecord


_POSITIVE_IMAGE_HINTS = {
    "avatar",
    "profile",
    "headshot",
    "portrait",
    "person",
    "faculty",
    "team",
    "member",
    "people",
}
_NEGATIVE_IMAGE_HINTS = {
    "logo",
    "banner",
    "icon",
    "sprite",
    "default",
    "placeholder",
    "header",
    "footer",
}
_SOCIAL_SHARE_HINTS = {
    "ogp",
    "opengraph",
    "open-graph",
    "social",
    "share",
    "thumbnail",
    "thumb",
}
_NON_PORTRAIT_HINTS = {
    "banner",
    "hero",
    "cover",
    "bnr",
    "news",
    "event",
    "paper",
    "publication",
    "figure",
    "diagram",
    "graph",
    "chart",
    "project",
}
_SOCIAL_MEDIUM_DOMAINS = {
    "scholar.google.com",
    "semanticscholar.org",
    "dblp.org",
}
_SOCIAL_HIGH_DOMAINS = {
    "researchgate.net",
    "academia.edu",
}


class _ProfileHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.meta_images: list[dict[str, str]] = []
        self.image_rows: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k.lower(): (v or "") for k, v in attrs}
        if tag.lower() == "title":
            self.in_title = True
            return
        if tag.lower() == "meta":
            key = (attr_map.get("property") or attr_map.get("name") or "").lower()
            content = (attr_map.get("content") or "").strip()
            if key in {"og:image", "twitter:image", "twitter:image:src"} and content:
                self.meta_images.append({"url": content, "tag": key})
            return
        if tag.lower() == "img":
            self.image_rows.append(
                {
                    "src": attr_map.get("src", "").strip(),
                    "srcset": attr_map.get("srcset", "").strip(),
                    "alt": attr_map.get("alt", "").strip(),
                    "class": attr_map.get("class", "").strip(),
                    "id": attr_map.get("id", "").strip(),
                    "width": attr_map.get("width", "").strip(),
                    "height": attr_map.get("height", "").strip(),
                }
            )

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        if len(self.text_parts) < 2000:
            self.text_parts.append(text)


def _safe_int(text: str) -> int | None:
    if not text:
        return None
    match = re.search(r"\d+", text)
    if not match:
        return None
    try:
        value = int(match.group(0))
    except ValueError:
        return None
    return value if value > 0 else None


def _parse_srcset(srcset: str, base_url: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for chunk in srcset.split(","):
        part = chunk.strip()
        if not part:
            continue
        bits = part.split()
        candidate_url = bits[0].strip()
        width = None
        if len(bits) > 1 and bits[1].endswith("w"):
            width = _safe_int(bits[1][:-1])
        rows.append(
            {
                "url": urljoin(base_url, candidate_url),
                "declared_width": width,
            }
        )
    return rows


def _is_likely_list_or_search_page(url: str, page_text: str) -> bool:
    lower_url = url.lower()
    if any(token in lower_url for token in ["/search", "?q=", "/people", "/directory", "/staff", "/members"]):
        return True
    lower_text = page_text.lower()
    return any(token in lower_text for token in ["search results", "showing results", "directory", "all members", "faculty list"])


def _normalize_domain(url: str) -> str:
    host = urlparse(url).netloc.lower().strip()
    return host[4:] if host.startswith("www.") else host


def _infer_source_type(domain: str) -> tuple[str, float]:
    if domain in _SOCIAL_HIGH_DOMAINS or any(domain.endswith(f".{d}") for d in _SOCIAL_HIGH_DOMAINS):
        return "academic_social_profile", 2.8
    if domain.endswith("orcid.org"):
        return "orcid", 1.8
    if domain in _SOCIAL_MEDIUM_DOMAINS or any(domain.endswith(f".{d}") for d in _SOCIAL_MEDIUM_DOMAINS):
        return "academic_index", 1.2
    if domain.endswith(".edu") or ".ac." in domain:
        return "institution_profile", 2.4
    if any(token in domain for token in ["university", "college", "institute", "faculty", "lab", "school"]):
        return "institution_profile", 2.2
    return "academic_homepage", 2.0


def _score_image_row(
    *,
    author: AuthorRecord,
    page_url: str,
    page_title: str,
    page_text: str,
    image_url: str,
    alt_text: str,
    class_text: str,
    id_text: str,
    width: int | None,
    height: int | None,
    profile_domain: str,
    source_type: str,
    base_score: float,
    list_or_search_page: bool,
    declared_width: int | None = None,
) -> float:
    score = base_score
    display_name = (author.display_name or "").strip().lower()
    institution_name = (author.institution_name or "").strip().lower()
    orcid = (author.orcid or "").strip().lower()
    title_lower = page_title.lower()
    text_lower = page_text.lower()

    if display_name and display_name in title_lower:
        score += 2.0
    elif display_name and display_name in text_lower:
        score += 1.0

    if orcid and orcid in text_lower:
        score += 2.5
    if institution_name and institution_name in text_lower:
        score += 1.0

    attrs_blob = " ".join([alt_text, class_text, id_text, image_url]).lower()
    positive_hits = sum(1 for kw in _POSITIVE_IMAGE_HINTS if kw in attrs_blob)
    negative_hits = sum(1 for kw in _NEGATIVE_IMAGE_HINTS if kw in attrs_blob)
    score += min(1.5, positive_hits * 0.5)
    score -= min(2.5, negative_hits * 0.8)

    if any(token in image_url.lower() for token in ["/logo", "logo.", "favicon", "sprite", "/icons/"]):
        score -= 2.2
    if any(token in image_url.lower() for token in _SOCIAL_SHARE_HINTS):
        score -= 2.4
    if any(token in image_url.lower() for token in _NON_PORTRAIT_HINTS):
        score -= 1.2
    if image_url.lower().endswith(".svg"):
        score -= 2.0
    if "og:image" in class_text.lower() or "twitter:image" in class_text.lower():
        score -= 1.6

    min_edge = min(width, height) if width and height else None
    if min_edge is not None:
        if min_edge < 80:
            score -= 1.6
        elif min_edge < 120:
            score -= 0.8
        elif min_edge >= 320:
            score += 0.8
        elif min_edge >= 180:
            score += 0.4
    elif declared_width is not None:
        if declared_width >= 320:
            score += 0.5
        elif declared_width < 120:
            score -= 0.5

    image_domain = _normalize_domain(image_url)
    if image_domain and image_domain == profile_domain:
        score += 0.3

    # Prefer image files that directly align with the author's name.
    image_path = urlparse(image_url).path.rsplit("/", 1)[-1].lower()
    base_name = image_path.rsplit(".", 1)[0]
    normalized_base = re.sub(r"[^a-z0-9]+", " ", base_name).strip()
    name_tokens = [token for token in re.sub(r"[^a-z0-9]+", " ", display_name).split() if token]
    if normalized_base and name_tokens:
        if normalized_base in name_tokens:
            score += 1.5
        elif any(token and token in normalized_base for token in name_tokens):
            score += 0.5

    if list_or_search_page:
        score -= 2.0

    return score


class ProfileImageExtractor:
    def __init__(self, profile_image_max_per_page: int, profile_image_min_score: float) -> None:
        self._profile_image_max_per_page = max(1, int(profile_image_max_per_page))
        # Min score filtering is applied in WebSearchClient for global ranking.
        _ = profile_image_min_score

    def extract(self, author: AuthorRecord, profile_page_url: str, html_text: str) -> list[dict[str, Any]]:
        parser = _ProfileHTMLParser()
        try:
            parser.feed(html_text)
        except Exception:
            return []

        page_title = unescape(" ".join(parser.title_parts)).strip()
        page_text = unescape(" ".join(parser.text_parts)).strip()
        profile_domain = _normalize_domain(profile_page_url)
        source_type, domain_base_score = _infer_source_type(profile_domain)
        list_or_search_page = _is_likely_list_or_search_page(profile_page_url, page_text)

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        for meta in parser.meta_images:
            image_url = urljoin(profile_page_url, meta.get("url", "").strip())
            if not image_url.startswith(("http://", "https://")) or image_url in seen:
                continue
            seen.add(image_url)
            score = _score_image_row(
                author=author,
                page_url=profile_page_url,
                page_title=page_title,
                page_text=page_text,
                image_url=image_url,
                alt_text="",
                class_text=meta.get("tag", ""),
                id_text="",
                width=None,
                height=None,
                profile_domain=profile_domain,
                source_type=source_type,
                base_score=domain_base_score + 0.6,
                list_or_search_page=list_or_search_page,
            )
            rows.append(
                {
                    "image_url": image_url,
                    "source_url": profile_page_url,
                    "title": page_title,
                    "snippet": "",
                    "linked_profile_url": profile_page_url,
                    "linked_profile_domain": profile_domain,
                    "source_type": source_type,
                    "alt_text": "",
                    "score": round(score, 3),
                    "reason": f"{meta.get('tag', 'meta_image')}_from_profile_page",
                    "declared_width": None,
                    "declared_height": None,
                }
            )

        for img in parser.image_rows:
            src = img.get("src", "").strip()
            srcset = img.get("srcset", "").strip()
            alt_text = img.get("alt", "").strip()
            class_text = img.get("class", "").strip()
            id_text = img.get("id", "").strip()
            declared_width = _safe_int(img.get("width", ""))
            declared_height = _safe_int(img.get("height", ""))

            candidate_urls: list[dict[str, Any]] = []
            if src:
                candidate_urls.append({"url": urljoin(profile_page_url, src), "declared_width": declared_width})
            if srcset:
                srcset_rows = _parse_srcset(srcset, profile_page_url)
                srcset_rows.sort(key=lambda row: row.get("declared_width") or 0, reverse=True)
                candidate_urls.extend(srcset_rows[:2])

            for item in candidate_urls:
                image_url = item.get("url") or ""
                if not image_url.startswith(("http://", "https://")) or image_url in seen:
                    continue
                seen.add(image_url)
                score = _score_image_row(
                    author=author,
                    page_url=profile_page_url,
                    page_title=page_title,
                    page_text=page_text,
                    image_url=image_url,
                    alt_text=alt_text,
                    class_text=class_text,
                    id_text=id_text,
                    width=declared_width,
                    height=declared_height,
                    profile_domain=profile_domain,
                    source_type=source_type,
                    base_score=domain_base_score,
                    list_or_search_page=list_or_search_page,
                    declared_width=item.get("declared_width"),
                )
                rows.append(
                    {
                        "image_url": image_url,
                        "source_url": profile_page_url,
                        "title": page_title,
                        "snippet": "",
                        "linked_profile_url": profile_page_url,
                        "linked_profile_domain": profile_domain,
                        "source_type": source_type,
                        "alt_text": alt_text,
                        "score": round(score, 3),
                        "reason": "img_tag_from_profile_page",
                        "declared_width": declared_width,
                        "declared_height": declared_height,
                    }
                )

        rows.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        return rows[: self._profile_image_max_per_page]
