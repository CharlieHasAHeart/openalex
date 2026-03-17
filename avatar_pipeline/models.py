from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AuthorRecord:
    author_id: str
    display_name: str
    orcid_url: str | None
    institution_name: str | None = None

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
    # Debug pass-through source pages derived from tool output source URLs.
    source_pages: list[dict[str, Any]] = field(default_factory=list)
    # Tool-output candidate rows after normalization/dedupe.
    image_candidates: list[dict[str, Any]] = field(default_factory=list)
    # Final candidate rows after max-candidate filtering.
    filtered_candidates: list[dict[str, Any]] = field(default_factory=list)
    # Raw response payload snapshot for run audit.
    raw_content: str | None = None
    # Debug/audit text only; candidate selection is tool-output driven.
    response_text: str | None = None
    # Optional explanatory string from upstream early-abandon handling.
    abandon_reason_log: str | None = None
    # Response usage accounting for run audit and cost monitoring.
    usage_total_tokens: int | None = None
