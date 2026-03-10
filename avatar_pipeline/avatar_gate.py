from __future__ import annotations

from avatar_pipeline.models import ImageCandidate


def validate_image_candidate(
    candidate: ImageCandidate,
    downloaded_bytes: bytes,
    allowed_mime: set[str],
    min_edge_px: int,
) -> tuple[bool, str | None]:
    if candidate.mime not in allowed_mime:
        return False, f"invalid_image_mime:{candidate.mime}"
    if not downloaded_bytes:
        return False, "empty_image_bytes"
    if candidate.width <= 0 or candidate.height <= 0:
        return False, "image_dimension_unknown"
    return True, None


def mime_to_ext(mime: str) -> str:
    mapping = {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }
    if mime not in mapping:
        raise ValueError(f"Unsupported mime for extension mapping: {mime}")
    return mapping[mime]
