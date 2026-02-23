from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AuthorRecord:
    author_id: str
    display_name: str
    orcid_url: str | None
    institution_name: str | None = None
    concept_names: list[str] | None = None

    @property
    def orcid(self) -> str | None:
        if not self.orcid_url:
            return None
        value = self.orcid_url.strip().rstrip("/")
        if not value:
            return None
        return value.split("/")[-1]

    @classmethod
    def from_openalex(cls, data: dict[str, Any]) -> "AuthorRecord":
        institution_name = None
        lki = data.get("last_known_institution") or {}
        if isinstance(lki, dict):
            institution_name = lki.get("display_name")

        concept_names: list[str] = []
        raw_concepts = data.get("x_concepts") or []
        if isinstance(raw_concepts, list):
            for item in raw_concepts:
                if not isinstance(item, dict):
                    continue
                name = item.get("display_name")
                if isinstance(name, str) and name:
                    concept_names.append(name)

        return cls(
            author_id=data["id"],
            display_name=data.get("display_name", ""),
            orcid_url=data.get("orcid"),
            institution_name=institution_name,
            concept_names=concept_names,
        )


@dataclass(slots=True)
class ImageCandidate:
    commons_file: str
    download_url: str
    mime: str
    width: int
    height: int
    size_bytes: int


@dataclass(slots=True)
class PipelineResult:
    author_id: str
    status: str
    error_message: str | None = None
    wikidata_qid: str | None = None
    commons_file: str | None = None
    content_sha256: str | None = None
    oss_object_key: str | None = None
    oss_url: str | None = None


@dataclass(slots=True)
class AuthorCandidate:
    author: AuthorRecord
    seed_work_id: str
    seed_work_cited_by_count: int
    appearance_count: int = 1
