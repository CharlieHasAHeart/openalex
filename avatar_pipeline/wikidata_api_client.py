from __future__ import annotations

from avatar_pipeline.http import HttpClient


class WikidataApiClient:
    def __init__(self, api_url: str, http: HttpClient) -> None:
        self._api_url = api_url
        self._http = http

    def search_entities(self, name: str, limit: int = 5) -> list[dict]:
        params = {
            "action": "wbsearchentities",
            "format": "json",
            "language": "en",
            "type": "item",
            "search": name,
            "limit": str(limit),
        }
        data = self._http.request("GET", self._api_url, params=params).json()
        return data.get("search") or []
