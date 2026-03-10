from __future__ import annotations

import time
from contextlib import AbstractContextManager
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
        if conn is None or conn.closed:
            return
        conn.close()

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _resolve_authors_analysis_table_and_id_col(self) -> tuple[str, str]:
        if self._authors_analysis_table_and_id_col is not None:
            return self._authors_analysis_table_and_id_col
        for table in ("authors_analysis", "openalex.authors_analysis", "public.authors_analysis"):
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
                self._authors_analysis_table_and_id_col = (table, id_col)
                return self._authors_analysis_table_and_id_col
            except psycopg.errors.UndefinedTable:
                continue
        raise RuntimeError("Missing table authors_analysis (or openalex.authors_analysis / public.authors_analysis)")

    def _row_to_author_record(self, row: dict[str, Any], id_col: str) -> AuthorRecord | None:
        raw_id = row.get("author_id") or row.get(id_col)
        if raw_id is None:
            return None
        institution_name: str | None = None
        last_known_institutions = row.get("last_known_institutions")
        if isinstance(last_known_institutions, list) and last_known_institutions:
            first = last_known_institutions[0]
            if isinstance(first, dict):
                institution_name = (first.get("display_name") or "").strip() or None
        elif isinstance(last_known_institutions, dict):
            institution_name = (last_known_institutions.get("display_name") or "").strip() or None

        concept_names: list[str] = []
        raw_concepts = row.get("x_concepts") or row.get("concepts") or []
        if isinstance(raw_concepts, list):
            for item in raw_concepts:
                if isinstance(item, dict):
                    name = str(item.get("display_name") or "").strip()
                    if name:
                        concept_names.append(name)

        return AuthorRecord(
            author_id=str(raw_id).strip(),
            display_name=str(row.get("display_name") or "").strip(),
            orcid_url=str(row.get("orcid")).strip() if row.get("orcid") else None,
            institution_name=institution_name,
            concept_names=concept_names,
            profile=row,
        )

    def list_author_records_from_authors_analysis(self, limit: int | None = None, offset: int = 0) -> list[AuthorRecord]:
        table, id_col = self._resolve_authors_analysis_table_and_id_col()
        sql = f"""
        SELECT
            {id_col} AS author_id,
            orcid,
            display_name,
            x_concepts,
            concepts,
            last_known_institutions,
            affiliations
        FROM {table}
        ORDER BY {id_col}
        """
        params: list[Any] = []
        if limit is not None and limit > 0:
            sql += " LIMIT %s"
            params.append(limit)
        if offset > 0:
            sql += " OFFSET %s"
            params.append(offset)

        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, tuple(params))
                return cur.fetchall() or []

        rows = self._run_with_reconnect(_op)
        records: list[AuthorRecord] = []
        for row in rows:
            record = self._row_to_author_record(row, id_col=id_col)
            if record is not None:
                records.append(record)
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
            x_concepts,
            concepts,
            last_known_institutions,
            affiliations
        FROM {table}
        WHERE {id_col}::text = ANY(%s)
        """

        def _op():
            with self._conn.cursor() as cur:
                cur.execute(sql, (normalized_ids,))
                return cur.fetchall() or []

        rows = self._run_with_reconnect(_op)
        records: list[AuthorRecord] = []
        for row in rows:
            record = self._row_to_author_record(row, id_col=id_col)
            if record is not None:
                records.append(record)
        return records

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
