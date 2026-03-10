from __future__ import annotations

import os
from dataclasses import dataclass


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
            if key and (override or key not in os.environ):
                os.environ[key] = value


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Missing required env: {name}")
    return value or ""


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_env_any(names: tuple[str, ...], *, required: bool = False, default: str | None = None) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value != "":
            return value
    if required:
        raise ValueError(f"Missing required env: one of {', '.join(names)}")
    return default or ""


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


@dataclass(slots=True)
class PipelineConfig:
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
    aliyun_oss_public_base_url: str
    aliyun_oss_key_prefix: str
    aliyun_oss_cache_control: str | None
    allowed_mime: set[str]
    min_image_edge_px: int
    request_timeout_seconds: int
    max_retries: int
    global_qps_limit: float
    retry_base_seconds: float
    retry_max_seconds: float
    retry_jitter_ratio: float
    retry_429_min_delay_seconds: float
    qwen_api_key: str | None
    qwen_base_url: str
    qwen_response_path: str
    qwen_model: str
    qwen_enable_web_search: bool
    qwen_max_candidates: int
    qwen_min_confidence: float
    qwen_timeout_seconds: int
    qwen_min_call_interval_seconds: float

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        allowed_mime_raw = _get_env("ALLOWED_MIME", "image/jpeg,image/png,image/webp")
        allowed_mime = {item.strip() for item in allowed_mime_raw.split(",") if item.strip()}
        bucket = _get_env("ALIYUN_OSS_BUCKET", required=True)
        endpoint = _normalize_oss_endpoint(_get_env("ALIYUN_OSS_ENDPOINT", required=True), bucket)
        return cls(
            pghost=_get_env("PGHOST", required=True),
            pgport=int(_get_env("PGPORT", "5432")),
            pgdatabase=_get_env("PGDATABASE", required=True),
            pguser=_get_env("PGUSER", required=True),
            pgpassword=_get_env("PGPASSWORD", required=True),
            pgsslmode=os.getenv("PGSSLMODE"),
            aliyun_oss_access_key_id=_get_env("ALIYUN_OSS_ACCESS_KEY_ID", required=True),
            aliyun_oss_access_key_secret=_get_env("ALIYUN_OSS_ACCESS_KEY_SECRET", required=True),
            aliyun_oss_bucket=bucket,
            aliyun_oss_endpoint=endpoint,
            aliyun_oss_public_base_url=_get_env("ALIYUN_OSS_PUBLIC_BASE_URL", required=True),
            aliyun_oss_key_prefix=_get_env("ALIYUN_OSS_KEY_PREFIX", "openalex"),
            aliyun_oss_cache_control=os.getenv("ALIYUN_OSS_CACHE_CONTROL"),
            allowed_mime=allowed_mime,
            min_image_edge_px=int(_get_env("MIN_IMAGE_EDGE_PX", "96")),
            request_timeout_seconds=int(_get_env("REQUEST_TIMEOUT_SECONDS", "20")),
            max_retries=int(_get_env("MAX_RETRIES", "3")),
            global_qps_limit=float(_get_env("GLOBAL_QPS_LIMIT", "2")),
            retry_base_seconds=float(_get_env("RETRY_BASE_SECONDS", "1.5")),
            retry_max_seconds=float(_get_env("RETRY_MAX_SECONDS", "60")),
            retry_jitter_ratio=float(_get_env("RETRY_JITTER_RATIO", "0.25")),
            retry_429_min_delay_seconds=float(_get_env("RETRY_429_MIN_DELAY_SECONDS", "8")),
            qwen_api_key=_get_env_any(("LLM_API_KEY", "QWEN_API_KEY"), required=True),
            qwen_base_url=_get_env_any(("LLM_BASE_URL", "QWEN_BASE_URL"), required=True),
            qwen_response_path=_get_env("QWEN_RESPONSE_PATH", "/responses"),
            qwen_model=_get_env_any(("LLM_MODEL", "QWEN_MODEL"), required=True),
            qwen_enable_web_search=_bool_env("QWEN_ENABLE_WEB_SEARCH", True),
            qwen_max_candidates=int(_get_env("QWEN_MAX_CANDIDATES", "8")),
            qwen_min_confidence=float(_get_env("QWEN_MIN_CONFIDENCE", "0.55")),
            qwen_timeout_seconds=int(_get_env("QWEN_TIMEOUT_SECONDS", "120")),
            qwen_min_call_interval_seconds=float(_get_env("QWEN_MIN_CALL_INTERVAL_SECONDS", "0")),
        )
