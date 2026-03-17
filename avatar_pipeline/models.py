from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AuthorRecord:
    author_id: str
    display_name: str
    orcid_url: str | None
    institution_name: str | None = None
    institution_names: list[str] | None = None
    institution_country_codes: list[str] | None = None
    affiliations: Any = None
    last_known_institutions: Any = None
    profile: dict[str, Any] | None = None

    @property
    def orcid(self) -> str | None:
        if not self.orcid_url:
            return None
        value = self.orcid_url.strip().rstrip("/")
        if not value:
            return None
        return value.split("/")[-1]


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
    failure_reason: str | None = None
    commons_file: str | None = None
    content_sha256: str | None = None
    oss_object_key: str | None = None
    oss_url: str | None = None
    selected_candidate: dict[str, Any] | None = None
    ranked_candidates: list[dict[str, Any]] = field(default_factory=list)
    profile_pages: list[dict[str, Any]] = field(default_factory=list)
    image_candidates: list[dict[str, Any]] = field(default_factory=list)
    filtered_candidates: list[dict[str, Any]] = field(default_factory=list)
    raw_content: str | None = None
    response_text: str | None = None
    abandon_reason_log: str | None = None
    usage_total_tokens: int | None = None
