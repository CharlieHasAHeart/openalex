from __future__ import annotations

from contextlib import AbstractContextManager
import time
from typing import Any

import psycopg
from psycopg.rows import dict_row

from avatar_pipeline.models import AuthorRecord, PipelineResult


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
        self._authors_ids_table_and_id_col: tuple[str, str] | None = None
        self._authors_analysis_table_and_id_col: tuple[str, str] | None = None

    def _connect(self):
        conn = psycopg.connect(self._conninfo, row_factory=dict_row)
        conn.autocommit = True
        return conn

    def reconnect(self) -> None:
        self.close()
        self._conn = self._connect()

    def _run_with_reconnect(self, func, max_attempts: int = 3):
        last_exc: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                return func()
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
        if conn is None:
            return
        if conn.closed:
            return
        conn.close()

    def __exit__(self, *args: Any) -> None:
        self.close()

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
        WHERE author_id = %s
        """
        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, (author_id,))
                return cur.fetchone()
        return self._run_with_reconnect(_op)

    def get_existing_by_author_id(self, author_id: str) -> dict[str, Any] | None:
        return self.get_author_avatar_record(author_id)

    def get_avatar_state(self, author_id: str) -> dict[str, Any] | None:
        record = self.get_author_avatar_record(author_id)
        if record is None:
            return None
        return {
            "author_id": record.get("author_id"),
            "status": None,
            "updated_at": record.get("updated_at"),
        }

    def list_invalid_image_too_small_author_ids(self, limit: int | None = None) -> list[str]:
        sql = """
        SELECT author_id
        FROM openalex.authors_avatars
        WHERE status = 'invalid_image'
          AND error_message IN ('invalid_image_too_small', 'invalid_image_too_small_bytes')
        ORDER BY updated_at DESC
        """
        params: tuple[Any, ...] = ()
        if limit is not None and limit > 0:
            sql += " LIMIT %s"
            params = (limit,)
        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                return cur.fetchall() or []
        rows = self._run_with_reconnect(_op)
        return [row["author_id"] for row in rows]

    def list_author_ids_from_authors(self, limit: int | None = None, offset: int = 0) -> list[str]:
        params: list[Any] = []
        safe_offset = max(offset, 0)
        if limit is not None and limit > 0:
            params.append(limit)
        if safe_offset > 0:
            params.append(safe_offset)

        def _build_sql(table_name: str) -> str:
            sql = f"SELECT id FROM {table_name} ORDER BY id"
            if limit is not None and limit > 0:
                sql += " LIMIT %s"
            if safe_offset > 0:
                sql += " OFFSET %s"
            return sql

        def _op():
            with self._conn.cursor() as cur:
                try:
                    cur.execute(_build_sql("authors"), tuple(params))
                except psycopg.errors.UndefinedTable:
                    cur.execute(_build_sql("openalex.authors"), tuple(params))
                return cur.fetchall() or []
        rows = self._run_with_reconnect(_op)
        return [str(row["id"]).strip() for row in rows if row.get("id") is not None]

    def _resolve_authors_ids_table_and_id_col(self) -> tuple[str, str]:
        if self._authors_ids_table_and_id_col is not None:
            return self._authors_ids_table_and_id_col
        candidates = ("authors_ids", "openalex.authors_ids")
        for table in candidates:
            try:
                def _op():
                    with self._conn.cursor() as cur:
                        cur.execute(f"SELECT * FROM {table} LIMIT 0")
                        return [desc.name for desc in (cur.description or [])]
                cols = self._run_with_reconnect(_op)
                if not cols:
                    continue
                id_col = "author_id" if "author_id" in cols else ("id" if "id" in cols else "")
                if not id_col:
                    raise RuntimeError(f"{table} must contain column author_id or id")
                if "orcid" not in cols:
                    raise RuntimeError(f"{table} must contain column orcid")
                self._authors_ids_table_and_id_col = (table, id_col)
                return self._authors_ids_table_and_id_col
            except psycopg.errors.UndefinedTable:
                continue
        raise RuntimeError("Missing table authors_ids (or openalex.authors_ids)")

    def _resolve_authors_analysis_table_and_id_col(self) -> tuple[str, str]:
        if self._authors_analysis_table_and_id_col is not None:
            return self._authors_analysis_table_and_id_col
        candidates = ("authors_analysis", "openalex.authors_analysis")
        for table in candidates:
            try:
                def _op():
                    with self._conn.cursor() as cur:
                        cur.execute(f"SELECT * FROM {table} LIMIT 0")
                        return [desc.name for desc in (cur.description or [])]
                cols = self._run_with_reconnect(_op)
                if not cols:
                    continue
                if "orcid" not in cols:
                    raise RuntimeError(f"{table} must contain column orcid")
                id_col = "author_id" if "author_id" in cols else ("id" if "id" in cols else "")
                if not id_col:
                    raise RuntimeError(f"{table} must contain column author_id or id")
                self._authors_analysis_table_and_id_col = (table, id_col)
                return self._authors_analysis_table_and_id_col
            except psycopg.errors.UndefinedTable:
                continue
        raise RuntimeError("Missing table authors_analysis (or openalex.authors_analysis)")

    def _row_to_author_record(self, row: dict[str, Any], id_col: str) -> AuthorRecord | None:
        raw_id = row.get("author_id") or row.get(id_col)
        if raw_id is None:
            return None
        orcid = row.get("orcid")
        display_name = (row.get("display_name") or "").strip()
        lki = row.get("last_known_institutions")
        institution_name: str | None = None
        if isinstance(lki, list) and lki:
            first = lki[0]
            if isinstance(first, dict):
                institution_name = (first.get("display_name") or "").strip() or None
        return AuthorRecord(
            author_id=str(raw_id).strip(),
            display_name=display_name,
            orcid_url=str(orcid).strip() if orcid else None,
            institution_name=institution_name,
            profile=row,
        )

    def list_author_records_from_authors_analysis(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[AuthorRecord]:
        table, id_col = self._resolve_authors_analysis_table_and_id_col()
        safe_offset = max(offset, 0)
        sql = f"""
        SELECT
            {id_col} AS author_id,
            orcid,
            display_name,
            works_count,
            cited_by_count,
            affiliations,
            last_known_institutions
        FROM {table}
        WHERE COALESCE(NULLIF(TRIM(orcid), ''), NULL) IS NOT NULL
        ORDER BY {id_col}
        """
        params: list[Any] = []
        if limit is not None and limit > 0:
            sql += " LIMIT %s"
            params.append(limit)
        if safe_offset > 0:
            sql += " OFFSET %s"
            params.append(safe_offset)

        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return cur.fetchall() or []
        rows = self._run_with_reconnect(_op)
        records: list[AuthorRecord] = []
        for row in rows:
            rec = self._row_to_author_record(row, id_col=id_col)
            if rec is not None:
                records.append(rec)
        return records

    def list_author_records_from_authors_analysis_by_ids(self, author_ids: list[str]) -> list[AuthorRecord]:
        normalized_ids = [item.strip() for item in author_ids if item and item.strip()]
        if not normalized_ids:
            return []
        table, id_col = self._resolve_authors_analysis_table_and_id_col()
        sql = f"""
        SELECT
            {id_col} AS author_id,
            orcid,
            display_name,
            works_count,
            cited_by_count,
            affiliations,
            last_known_institutions
        FROM {table}
        WHERE {id_col}::text = ANY(%s)
          AND COALESCE(NULLIF(TRIM(orcid), ''), NULL) IS NOT NULL
        """
        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, (normalized_ids,))
                return cur.fetchall() or []
        rows = self._run_with_reconnect(_op)
        records: list[AuthorRecord] = []
        for row in rows:
            rec = self._row_to_author_record(row, id_col=id_col)
            if rec is not None:
                records.append(rec)
        return records

    def list_author_orcid_pairs_from_authors_ids(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[tuple[str, str | None]]:
        table, id_col = self._resolve_authors_ids_table_and_id_col()
        safe_offset = max(offset, 0)
        sql = f"SELECT {id_col} AS author_id, orcid FROM {table} ORDER BY {id_col}"
        params: list[Any] = []
        if limit is not None and limit > 0:
            sql += " LIMIT %s"
            params.append(limit)
        if safe_offset > 0:
            sql += " OFFSET %s"
            params.append(safe_offset)

        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return cur.fetchall() or []
        rows = self._run_with_reconnect(_op)

        pairs: list[tuple[str, str | None]] = []
        for row in rows:
            author_id = row.get("author_id")
            if author_id is None:
                continue
            orcid = row.get("orcid")
            pairs.append((str(author_id).strip(), str(orcid).strip() if orcid is not None else None))
        return pairs

    def list_author_orcid_pairs_by_author_ids(self, author_ids: list[str]) -> list[tuple[str, str | None]]:
        normalized_ids = [item.strip() for item in author_ids if item and item.strip()]
        if not normalized_ids:
            return []

        table, id_col = self._resolve_authors_ids_table_and_id_col()
        sql = f"SELECT {id_col} AS author_id, orcid FROM {table} WHERE {id_col} = ANY(%s)"
        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, (normalized_ids,))
                return cur.fetchall() or []
        rows = self._run_with_reconnect(_op)

        pairs: list[tuple[str, str | None]] = []
        for row in rows:
            author_id = row.get("author_id")
            if author_id is None:
                continue
            orcid = row.get("orcid")
            pairs.append((str(author_id).strip(), str(orcid).strip() if orcid is not None else None))
        return pairs

    def upsert_result(self, result: PipelineResult) -> None:
        if result.status != "ok":
            return
        params = {
            "author_id": result.author_id,
            "oss_object_key": result.oss_object_key,
            "oss_url": result.oss_url,
            "commons_file": result.commons_file,
            "content_sha256": result.content_sha256,
        }
        def _op():
            with self._conn.cursor() as cur:
                sql_ok = """
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
                cur.execute(sql_ok, params)
                return None

        self._run_with_reconnect(_op)
