from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

import psycopg
from psycopg.rows import dict_row

from avatar_pipeline.models import PipelineResult


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
        conninfo = (
            f"host={host} port={port} dbname={database} user={user} password={password}"
            + (f" sslmode={sslmode}" if sslmode else "")
        )
        self._conn = psycopg.connect(conninfo, row_factory=dict_row)
        self._conn.autocommit = True

    def __exit__(self, *args: Any) -> None:
        self._conn.close()

    def get_existing_by_author_id(self, author_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT author_id, content_sha256, oss_object_key, oss_url, status, updated_at
        FROM openalex.authors_avatars
        WHERE author_id = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (author_id,))
            return cur.fetchone()

    def get_avatar_state(self, author_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT author_id, status, updated_at
        FROM openalex.authors_avatars
        WHERE author_id = %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (author_id,))
            return cur.fetchone()

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
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall() or []
        return [row["author_id"] for row in rows]

    def upsert_result(self, result: PipelineResult) -> None:
        params = {
            "author_id": result.author_id,
            "oss_object_key": result.oss_object_key,
            "oss_url": result.oss_url,
            "status": result.status,
            "error_message": result.error_message,
            "wikidata_qid": result.wikidata_qid,
            "commons_file": result.commons_file,
            "content_sha256": result.content_sha256,
        }
        with self._conn.cursor() as cur:
            if result.status == "ok":
                sql_ok = """
                INSERT INTO openalex.authors_avatars (
                    author_id,
                    oss_object_key,
                    oss_url,
                    status,
                    error_message,
                    wikidata_qid,
                    commons_file,
                    content_sha256,
                    updated_at
                )
                VALUES (
                    %(author_id)s,
                    %(oss_object_key)s,
                    %(oss_url)s,
                    %(status)s,
                    %(error_message)s,
                    %(wikidata_qid)s,
                    %(commons_file)s,
                    %(content_sha256)s,
                    NOW()
                )
                ON CONFLICT (author_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    error_message = EXCLUDED.error_message,
                    wikidata_qid = COALESCE(EXCLUDED.wikidata_qid, openalex.authors_avatars.wikidata_qid),
                    commons_file = COALESCE(EXCLUDED.commons_file, openalex.authors_avatars.commons_file),
                    oss_object_key = EXCLUDED.oss_object_key,
                    oss_url = EXCLUDED.oss_url,
                    content_sha256 = EXCLUDED.content_sha256,
                    updated_at = NOW()
                """
                cur.execute(sql_ok, params)
                return

            # Failure status: only update existing row to avoid NOT NULL violations on insert-only rows.
            sql_non_ok = """
            UPDATE openalex.authors_avatars
            SET
                status = %(status)s,
                error_message = %(error_message)s,
                wikidata_qid = COALESCE(%(wikidata_qid)s, openalex.authors_avatars.wikidata_qid),
                commons_file = COALESCE(%(commons_file)s, openalex.authors_avatars.commons_file),
                updated_at = NOW()
            WHERE author_id = %(author_id)s
            """
            cur.execute(sql_non_ok, params)
