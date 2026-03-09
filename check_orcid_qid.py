#!/usr/bin/env python3
from __future__ import annotations

import requests

WDQS = "https://query.wikidata.org/sparql"
HEADERS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "openalex-avatar-pipeline/1.0 (contact: you@example.com)",
}
ORCID = "0000-0002-4768-4492"
QID = "Q20090537"


def run_query(query: str) -> dict:
    resp = requests.get(WDQS, params={"query": query}, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    q1 = f'SELECT ?item WHERE {{ ?item wdt:P496 "{ORCID}". }}'
    r1 = run_query(q1)
    q1_items = [b["item"]["value"] for b in r1.get("results", {}).get("bindings", [])]

    q2 = (
        "SELECT ?orcid ?img WHERE { "
        f"wd:{QID} wdt:P496 ?orcid . "
        f"OPTIONAL {{ wd:{QID} wdt:P18 ?img . }} "
        "}"
    )
    r2 = run_query(q2)
    q2_rows = r2.get("results", {}).get("bindings", [])

    print(f"ORCID={ORCID}")
    print(f"ORCID->QID matches ({len(q1_items)}):")
    for item in q1_items:
        print(f"  - {item}")

    print(f"\nQID={QID} P496/P18 rows ({len(q2_rows)}):")
    for row in q2_rows:
        orcid = row.get("orcid", {}).get("value")
        image = row.get("img", {}).get("value")
        print(f"  - orcid={orcid} image={image}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
