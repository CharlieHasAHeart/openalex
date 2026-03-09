from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4


@dataclass(slots=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


@dataclass(slots=True)
class StandaloneConfig:
    pghost: str
    pgport: int
    pgdatabase: str
    pguser: str
    pgpassword: str
    pgsslmode: str | None
    aliyun_oss_access_key_id: str
    aliyun_oss_access_key_secret: str
    aliyun_oss_bucket: str
    aliyun_oss_endpoint: str
    aliyun_oss_key_prefix: str
    openalex_base_url: str
    wikidata_api_url: str
    wdqs_endpoint: str
    openalex_mailto: str | None


def load_dotenv(path: str = ".env", override: bool = False) -> None:
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
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = value


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Missing required env: {name}")
    return value or ""


def _normalize_oss_endpoint(endpoint: str, bucket: str) -> str:
    value = endpoint.strip()
    if value.startswith("https://"):
        value = value[len("https://") :]
    elif value.startswith("http://"):
        value = value[len("http://") :]
    value = value.strip("/")
    bucket_prefix = f"{bucket}."
    if value.startswith(bucket_prefix):
        value = value[len(bucket_prefix) :]
    return value


def load_config() -> StandaloneConfig:
    bucket = _env("ALIYUN_OSS_BUCKET", required=True)
    endpoint = _normalize_oss_endpoint(_env("ALIYUN_OSS_ENDPOINT", required=True), bucket)
    return StandaloneConfig(
        pghost=_env("PGHOST", required=True),
        pgport=int(_env("PGPORT", "5432")),
        pgdatabase=_env("PGDATABASE", required=True),
        pguser=_env("PGUSER", required=True),
        pgpassword=_env("PGPASSWORD", required=True),
        pgsslmode=os.getenv("PGSSLMODE"),
        aliyun_oss_access_key_id=_env("ALIYUN_OSS_ACCESS_KEY_ID", required=True),
        aliyun_oss_access_key_secret=_env("ALIYUN_OSS_ACCESS_KEY_SECRET", required=True),
        aliyun_oss_bucket=bucket,
        aliyun_oss_endpoint=endpoint,
        aliyun_oss_key_prefix=_env("ALIYUN_OSS_KEY_PREFIX", "openalex"),
        openalex_base_url=_env("OPENALEX_BASE_URL", "https://api.openalex.org"),
        wikidata_api_url=_env("WIKIDATA_API_URL", "https://www.wikidata.org/w/api.php"),
        wdqs_endpoint=_env("WDQS_ENDPOINT", "https://query.wikidata.org/sparql"),
        openalex_mailto=os.getenv("OPENALEX_MAILTO"),
    )


def check_database(cfg: StandaloneConfig, timeout: int) -> CheckResult:
    try:
        import psycopg
    except Exception as exc:
        return CheckResult("database", False, f"missing dependency psycopg: {exc}")

    conninfo = (
        f"host={cfg.pghost} port={cfg.pgport} dbname={cfg.pgdatabase} "
        f"user={cfg.pguser} password={cfg.pgpassword}"
        + (f" sslmode={cfg.pgsslmode}" if cfg.pgsslmode else "")
        + f" connect_timeout={timeout}"
    )
    try:
        with psycopg.connect(conninfo) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
        return CheckResult("database", bool(row and row[0] == 1), "SELECT 1 ok")
    except Exception as exc:
        return CheckResult("database", False, str(exc))


def check_openalex(cfg: StandaloneConfig, timeout: int) -> CheckResult:
    try:
        import requests
    except Exception as exc:
        return CheckResult("openalex", False, f"missing dependency requests: {exc}")

    params = {"per-page": "1", "select": "id,cited_by_count"}
    if cfg.openalex_mailto:
        params["mailto"] = cfg.openalex_mailto
    url = f"{cfg.openalex_base_url.rstrip('/')}/works"
    headers = {"User-Agent": "openalex-avatar-pipeline/1.0 (contact: ops@example.com)"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        results = (resp.json() or {}).get("results") or []
        if not results:
            return CheckResult("openalex", False, "empty results")
        return CheckResult("openalex", True, f"http {resp.status_code}, results={len(results)}")
    except Exception as exc:
        return CheckResult("openalex", False, str(exc))


def check_wikidata_api(cfg: StandaloneConfig, timeout: int) -> CheckResult:
    try:
        import requests
    except Exception as exc:
        return CheckResult("wikidata_api", False, f"missing dependency requests: {exc}")

    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "type": "item",
        "search": "Albert Einstein",
        "limit": "1",
    }
    headers = {"User-Agent": "openalex-avatar-pipeline/1.0 (contact: ops@example.com)"}
    try:
        resp = requests.get(cfg.wikidata_api_url, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        hits = (resp.json() or {}).get("search") or []
        if not hits:
            return CheckResult("wikidata_api", False, "no search hits")
        return CheckResult("wikidata_api", True, f"http {resp.status_code}, qid={hits[0].get('id')}")
    except Exception as exc:
        return CheckResult("wikidata_api", False, str(exc))


def check_wdqs(cfg: StandaloneConfig, timeout: int) -> CheckResult:
    try:
        import requests
    except Exception as exc:
        return CheckResult("wikidata_wdqs", False, f"missing dependency requests: {exc}")

    params = {"query": "SELECT ?item WHERE { ?item wdt:P31 wd:Q5 . } LIMIT 1"}
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "openalex-avatar-pipeline/1.0 (contact: ops@example.com)",
    }
    try:
        resp = requests.get(cfg.wdqs_endpoint, params=params, headers=headers, timeout=timeout)
        resp.raise_for_status()
        bindings = (((resp.json() or {}).get("results") or {}).get("bindings")) or []
        if not bindings:
            return CheckResult("wikidata_wdqs", False, "empty bindings")
        return CheckResult("wikidata_wdqs", True, f"http {resp.status_code}, bindings={len(bindings)}")
    except Exception as exc:
        return CheckResult("wikidata_wdqs", False, str(exc))


def check_oss(cfg: StandaloneConfig, timeout: int) -> CheckResult:
    try:
        import oss2

        auth = oss2.Auth(cfg.aliyun_oss_access_key_id, cfg.aliyun_oss_access_key_secret)
        bucket = oss2.Bucket(auth, cfg.aliyun_oss_endpoint, cfg.aliyun_oss_bucket, connect_timeout=timeout)
        list_result = bucket.list_objects(max_keys=1)
        count = len(list_result.object_list or [])

        prefix = cfg.aliyun_oss_key_prefix.strip("/")
        if not prefix:
            prefix = "openalex"
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        object_key = f"{prefix}/_connectivity_checks/{now}_{uuid4().hex}.txt"

        payload = b"connectivity_check_ok\n"
        bucket.put_object(object_key, payload, headers={"Content-Type": "text/plain"})
        bucket.delete_object(object_key)
        return CheckResult(
            "oss",
            True,
            f"list/put/delete ok, objects_returned={count}, temp_key={object_key}",
        )
    except Exception as exc:
        return CheckResult("oss", False, str(exc))


def run_checks(cfg: StandaloneConfig, timeout: int) -> list[CheckResult]:
    checks: list[Callable[[StandaloneConfig, int], CheckResult]] = [
        check_database,
        check_openalex,
        check_wikidata_api,
        check_wdqs,
        check_oss,
    ]
    return [check(cfg, timeout) for check in checks]


def main() -> int:
    parser = argparse.ArgumentParser(description="Connectivity checks for avatar pipeline dependencies")
    parser.add_argument("--dotenv", default=".env", help="Path to .env file")
    parser.add_argument("--timeout", type=int, default=10, help="Network/DB timeout seconds")
    args = parser.parse_args()

    load_dotenv(args.dotenv)
    cfg = load_config()
    results = run_checks(cfg, args.timeout)

    print("connectivity_check_results:")
    for item in results:
        status = "OK" if item.ok else "FAIL"
        print(f"- {item.name}: {status} | {item.detail}")

    failures = [item for item in results if not item.ok]
    if failures:
        print(f"summary: {len(failures)} failed, {len(results) - len(failures)} passed")
        return 1
    print(f"summary: all {len(results)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
