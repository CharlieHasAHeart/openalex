#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import date

import requests


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def shift_years_day(value: date, years: int) -> date:
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year + years)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count OpenAlex works in a publication-date window")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--base-url", default=os.getenv("OPENALEX_BASE_URL", "https://api.openalex.org"))
    parser.add_argument("--window-years", type=int, default=5)
    parser.add_argument("--window-start-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--window-end-date", default=None, help="YYYY-MM-DD")
    parser.add_argument("--mailto", default=os.getenv("OPENALEX_MAILTO"))
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.dotenv)

    window_end = date.fromisoformat(args.window_end_date) if args.window_end_date else date.today()
    if args.window_start_date:
        window_start = date.fromisoformat(args.window_start_date)
    else:
        window_start = shift_years_day(window_end, -args.window_years)

    base_url = args.base_url.rstrip("/")
    url = f"{base_url}/works"
    filter_expr = (
        f"from_publication_date:{window_start.isoformat()},"
        f"to_publication_date:{window_end.isoformat()}"
    )
    params = {
        "filter": filter_expr,
        "per-page": "1",
    }
    if args.mailto:
        params["mailto"] = args.mailto

    headers = {"User-Agent": "openalex-avatar-pipeline/1.0 (count-works-script)"}
    resp = requests.get(url, params=params, headers=headers, timeout=args.timeout)
    resp.raise_for_status()
    payload = resp.json()
    count = int((payload.get("meta") or {}).get("count") or 0)

    print(f"window_start={window_start.isoformat()}")
    print(f"window_end={window_end.isoformat()}")
    print(f"works_count={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
