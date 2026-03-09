#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from typing import Iterable

import requests

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def subtract_years(value: dt.date, years: int) -> dt.date:
    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(month=2, day=28, year=value.year - years)


def build_params(start_date: dt.date, cursor: str, per_page: int, email: str | None) -> dict[str, str | int]:
    params: dict[str, str | int] = {
        "filter": f"from_publication_date:{start_date.isoformat()}",
        "select": "id,authorships",
        "per-page": per_page,
        "cursor": cursor,
    }
    if email:
        params["mailto"] = email
    return params


def iter_works(start_date: dt.date, per_page: int, email: str | None, sleep_seconds: float) -> Iterable[dict]:
    cursor = "*"
    session = requests.Session()
    session.headers.update({"User-Agent": "openalex-author-counter/1.0"})

    while True:
        params = build_params(start_date=start_date, cursor=cursor, per_page=per_page, email=email)
        response = session.get(OPENALEX_WORKS_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        for work in payload.get("results", []):
            yield work

        meta = payload.get("meta", {})
        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            break
        cursor = next_cursor
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def count_authors(works: Iterable[dict]) -> tuple[int, int, int]:
    work_count = 0
    authorship_total = 0
    unique_authors: set[str] = set()

    for work in works:
        work_count += 1
        authorships = work.get("authorships", [])
        authorship_total += len(authorships)
        for authorship in authorships:
            author = authorship.get("author") or {}
            author_id = author.get("id")
            if author_id:
                unique_authors.add(author_id)

    return work_count, authorship_total, len(unique_authors)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="统计 OpenAlex 近五年论文作者数量（作者人次 + 去重作者数）。"
    )
    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="统计最近 N 年（默认 5）。",
    )
    parser.add_argument(
        "--per-page",
        type=int,
        default=200,
        help="每页拉取条数（默认 200，OpenAlex 上限通常为 200）。",
    )
    parser.add_argument(
        "--mailto",
        default=None,
        help="可选：你的邮箱（OpenAlex 推荐，便于请求识别）。",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="每页请求之间休眠秒数（默认 0）。",
    )
    args = parser.parse_args()

    if args.years <= 0:
        print("--years 必须为正整数", file=sys.stderr)
        return 2
    if args.per_page <= 0:
        print("--per-page 必须为正整数", file=sys.stderr)
        return 2

    today = dt.date.today()
    start_date = subtract_years(today, args.years)

    try:
        work_count, authorship_total, unique_author_total = count_authors(
            iter_works(
                start_date=start_date,
                per_page=args.per_page,
                email=args.mailto,
                sleep_seconds=args.sleep,
            )
        )
    except requests.HTTPError as exc:
        print(f"OpenAlex API 请求失败: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"网络请求异常: {exc}", file=sys.stderr)
        return 1

    print(f"统计区间起始日期: {start_date.isoformat()}")
    print(f"论文总数: {work_count}")
    print(f"作者人次总和(不去重): {authorship_total}")
    print(f"作者去重总数: {unique_author_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
