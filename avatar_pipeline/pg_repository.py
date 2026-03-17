from __future__ import annotations

import time
from contextlib import AbstractContextManager
from typing import Any, Callable, TypeVar

import psycopg
from psycopg.rows import dict_row

from avatar_pipeline.models import AuthorRecord, PipelineResult

T = TypeVar("T")


class PgRepository(AbstractContextManager["PgRepository"]):
    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        sslmode: str | None = None,
    ) -> None:
        self._conninfo = (
            f"host={host} port={port} dbname={database} user={user} password={password}"
            + (f" sslmode={sslmode}" if sslmode else "")
        )
        self._conn = self._connect()

    def _connect(self) -> Any:
        conn = psycopg.connect(self._conninfo, row_factory=dict_row)
        conn.autocommit = True
        return conn

    def reconnect(self) -> None:
        self.close()
        self._conn = self._connect()

    def _run_with_reconnect(self, operation: Callable[[], T], max_attempts: int = 3) -> T:
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return operation()
            except psycopg.OperationalError as exc:
                last_exc = exc
                if attempt >= max_attempts:
                    raise
                self.reconnect()
                time.sleep(min(2 ** (attempt - 1), 3))
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("unreachable")

    def close(self) -> None:
        conn = getattr(self, "_conn", None)
        if conn is None or conn.closed:
            return
        conn.close()

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _row_to_author_record(self, row: dict[str, Any]) -> AuthorRecord | None:
        raw_author_id = row.get("author_id")
        if raw_author_id is None:
            return None
        author_id = str(raw_author_id).strip()
        if not author_id:
            return None

        return AuthorRecord(
            author_id=author_id,
            display_name=str(row.get("display_name") or "").strip(),
            orcid_url=str(row.get("orcid") or "").strip() or None,
            institution_name=str(row.get("institution_name") or "").strip() or None,
        )

    def _fetch_author_records(self, sql: str, params: tuple[Any, ...]) -> list[AuthorRecord]:
        def _op() -> list[dict[str, Any]]:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall() or []

        rows = self._run_with_reconnect(_op)
        records: list[AuthorRecord] = []
        for row in rows:
            record = self._row_to_author_record(row)
            if record is not None:
                records.append(record)
        return records

    def list_author_records(self, limit: int | None = None, offset: int = 0) -> list[AuthorRecord]:
        sql = """
        SELECT
            aa.id AS author_id,
            aa.orcid AS orcid,
            aa.display_name,
            alk.institution_name
        FROM public.authors_analysis aa
        LEFT JOIN public.author_last_known_institution alk
            ON alk.author_id = aa.id
        ORDER BY aa.id
        """
        params: list[Any] = []
        if limit is not None and limit > 0:
            sql += " LIMIT %s"
            params.append(limit)
        if offset > 0:
            sql += " OFFSET %s"
            params.append(offset)
        return self._fetch_author_records(sql, tuple(params))

    def list_author_records_by_ids(self, author_ids: list[str]) -> list[AuthorRecord]:
        normalized_ids = [item.strip() for item in author_ids if item and item.strip()]
        if not normalized_ids:
            return []
        sql = """
        SELECT
            aa.id AS author_id,
            aa.orcid AS orcid,
            aa.display_name,
            alk.institution_name
        FROM public.authors_analysis aa
        LEFT JOIN public.author_last_known_institution alk
            ON alk.author_id = aa.id
        WHERE aa.id::text = ANY(%s)
        ORDER BY aa.id
        """
        return self._fetch_author_records(sql, (normalized_ids,))

    def get_author_avatar_record(self, author_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT
            author_id,
            content_sha256,
            oss_object_key,
            oss_url,
            updated_at,
            commons_file
        FROM public.authors_avatars
        WHERE author_id::text = %s
        """

        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, (author_id,))
                return cur.fetchone()

        return self._run_with_reconnect(_op)

    def upsert_result(self, result: PipelineResult) -> None:
        if result.status != "ok":
            return
        sql = """
        INSERT INTO public.authors_avatars (
            author_id,
            oss_object_key,
            oss_url,
            commons_file,
            content_sha256,
            updated_at
        )
        VALUES (
            %(author_id)s,
            %(oss_object_key)s,
            %(oss_url)s,
            %(commons_file)s,
            %(content_sha256)s,
            NOW()
        )
        ON CONFLICT (author_id) DO UPDATE SET
            commons_file = COALESCE(EXCLUDED.commons_file, public.authors_avatars.commons_file),
            oss_object_key = EXCLUDED.oss_object_key,
            oss_url = EXCLUDED.oss_url,
            content_sha256 = EXCLUDED.content_sha256,
            updated_at = NOW()
        """
        params = {
            "author_id": result.author_id,
            "oss_object_key": result.oss_object_key,
            "oss_url": result.oss_url,
            "commons_file": result.commons_file,
            "content_sha256": result.content_sha256,
        }

        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return None

        self._run_with_reconnect(_op)
