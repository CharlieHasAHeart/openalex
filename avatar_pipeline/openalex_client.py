from __future__ import annotations

from datetime import date
from typing import Iterator
from urllib.parse import urlparse

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import AuthorRecord


class OpenAlexClient:
    def __init__(self, base_url: str, http: HttpClient, mailto: str | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._mailto = mailto

    def get_author(self, author_id: str) -> AuthorRecord:
        normalized_id = self._normalize_author_id(author_id)
        path = f"{self._base_url}/authors/{normalized_id}"
        params: dict[str, str] = {}
        if self._mailto:
            params["mailto"] = self._mailto
        data = self._http.request("GET", path, params=params).json()
        return AuthorRecord.from_openalex(data)

    def _normalize_author_id(self, author_id: str) -> str:
        value = author_id.strip().rstrip("/")
        if value.startswith("http://") or value.startswith("https://"):
            parsed = urlparse(value)
            tail = parsed.path.rsplit("/", 1)[-1].strip()
            if tail:
                return tail
        return value

    def iter_authors_with_orcid(
        self,
        limit: int,
        per_page: int = 50,
        cursor: str = "*",
    ) -> Iterator[AuthorRecord]:
        yielded = 0
        next_cursor = cursor

        while yielded < limit and next_cursor:
            params = {
                "filter": "has_orcid:true",
                "select": "id,display_name,orcid,last_known_institution,x_concepts",
                "per-page": str(per_page),
                "cursor": next_cursor,
            }
            if self._mailto:
                params["mailto"] = self._mailto

            url = f"{self._base_url}/authors"
            payload = self._http.request("GET", url, params=params).json()
            results = payload.get("results") or []
            meta = payload.get("meta") or {}
            next_cursor = meta.get("next_cursor")

            for item in results:
                if yielded >= limit:
                    break
                author = AuthorRecord.from_openalex(item)
                if author.orcid:
                    yield author
                    yielded += 1

    def iter_top_works(
        self,
        top_n: int,
        top_offset: int,
        from_publication_date: date,
        to_publication_date: date,
        per_page: int = 200,
        cursor: str = "*",
    ) -> Iterator[dict]:
        yielded = 0
        scanned = 0
        next_cursor = cursor
        url = f"{self._base_url}/works"
        filter_expr = (
            f"from_publication_date:{from_publication_date.isoformat()},"
            f"to_publication_date:{to_publication_date.isoformat()}"
        )

        while yielded < top_n and next_cursor:
            params = {
                "filter": filter_expr,
                "sort": "cited_by_count:desc",
                "select": "id,cited_by_count,authorships",
                "per-page": str(per_page),
                "cursor": next_cursor,
            }
            if self._mailto:
                params["mailto"] = self._mailto

            payload = self._http.request("GET", url, params=params).json()
            results = payload.get("results") or []
            meta = payload.get("meta") or {}
            next_cursor = meta.get("next_cursor")

            for item in results:
                if yielded >= top_n:
                    break
                scanned += 1
                if scanned <= max(top_offset, 0):
                    continue
                yielded += 1
                yield item
