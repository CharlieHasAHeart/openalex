from __future__ import annotations

from urllib.parse import unquote, urlparse

from avatar_pipeline.http import HttpClient


class WdqsClient:
    def __init__(self, endpoint: str, http: HttpClient) -> None:
        self._endpoint = endpoint
        self._http = http

    def find_qid_by_orcid(self, orcid: str) -> tuple[str | None, str | None]:
        query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item WHERE {{
  ?item wdt:P496 "{orcid}".
}}
LIMIT 3
"""
        payload = self._sparql(query)
        bindings = payload.get("results", {}).get("bindings", [])
        if len(bindings) != 1:
            if len(bindings) == 0:
                return None, "no_match"
            return None, "qid_not_unique"

        item_uri = bindings[0]["item"]["value"]
        qid = item_uri.rsplit("/", 1)[-1]
        return qid, None

    def get_p18_image_by_qid(self, qid: str) -> str | None:
        query = f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?file WHERE {{
  wd:{qid} wdt:P18 ?file .
}}
LIMIT 1
"""
        payload = self._sparql(query)
        bindings = payload.get("results", {}).get("bindings", [])
        if not bindings:
            return None

        file_uri = bindings[0]["file"]["value"]
        path = urlparse(file_uri).path
        filename = path.rsplit("/", 1)[-1]
        return unquote(filename) if filename else None

    def _sparql(self, query: str) -> dict:
        headers = {"Accept": "application/sparql-results+json"}
        params = {"query": query}
        response = self._http.request("GET", self._endpoint, params=params, headers=headers)
        return response.json()
