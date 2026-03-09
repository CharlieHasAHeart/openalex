from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_LOCAL_PROXY = "http://127.0.0.1:7890"


def _has_env_value(name: str) -> bool:
    value = os.getenv(name)
    return value is not None and value.strip() != ""


def _apply_default_proxy_env() -> None:
    http_set = _has_env_value("http_proxy") or _has_env_value("HTTP_PROXY")
    https_set = _has_env_value("https_proxy") or _has_env_value("HTTPS_PROXY")

    if not http_set:
        os.environ["http_proxy"] = DEFAULT_LOCAL_PROXY
        os.environ["HTTP_PROXY"] = DEFAULT_LOCAL_PROXY
    if not https_set:
        os.environ["https_proxy"] = DEFAULT_LOCAL_PROXY
        os.environ["HTTPS_PROXY"] = DEFAULT_LOCAL_PROXY


def load_dotenv(path: str = ".env", override: bool = False) -> None:
    if not os.path.exists(path):
        _apply_default_proxy_env()
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
    _apply_default_proxy_env()


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
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


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


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
    avatar_thumb_width: int
    allowed_mime: set[str]
    min_image_edge_px: int
    request_timeout_seconds: int
    max_retries: int
    global_qps_limit: float
    retry_base_seconds: float
    retry_max_seconds: float
    retry_jitter_ratio: float
    retry_429_min_delay_seconds: float
    openalex_base_url: str
    wdqs_endpoint: str
    wikidata_api_url: str
    commons_api_url: str
    openalex_mailto: str | None
    allow_name_fallback: bool
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str
    llm_timeout_seconds: int
    websearch_max_results: int
    search_provider: str
    qwen_base_url: str
    qwen_response_path: str
    qwen_model: str
    qwen_enable_web_search: bool
    qwen_enable_t2i_search: bool
    qwen_max_candidates: int
    qwen_min_confidence: float
    qwen_timeout_seconds: int
    person_page_query_max: int
    person_page_per_query_results: int
    person_page_max_fetch: int
    profile_image_score_threshold: float
    refresh_ok_days: int
    refresh_no_image_days: int
    refresh_error_days: int
    refresh_ambiguous_days: int
    refresh_no_match_days: int

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        _apply_default_proxy_env()
        allowed_mime_raw = _get_env("ALLOWED_MIME", "image/jpeg,image/png,image/webp")
        allowed_mime = {item.strip() for item in allowed_mime_raw.split(",") if item.strip()}
        bucket = _get_env("ALIYUN_OSS_BUCKET", required=True)
        endpoint = _normalize_oss_endpoint(
            _get_env("ALIYUN_OSS_ENDPOINT", required=True),
            bucket,
        )

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
            avatar_thumb_width=int(_get_env("AVATAR_THUMB_WIDTH", "512")),
            allowed_mime=allowed_mime,
            min_image_edge_px=int(_get_env("MIN_IMAGE_EDGE_PX", "200")),
            request_timeout_seconds=int(_get_env("REQUEST_TIMEOUT_SECONDS", "20")),
            max_retries=int(_get_env("MAX_RETRIES", "3")),
            global_qps_limit=float(_get_env("GLOBAL_QPS_LIMIT", "2")),
            retry_base_seconds=float(_get_env("RETRY_BASE_SECONDS", "1.5")),
            retry_max_seconds=float(_get_env("RETRY_MAX_SECONDS", "60")),
            retry_jitter_ratio=float(_get_env("RETRY_JITTER_RATIO", "0.25")),
            retry_429_min_delay_seconds=float(_get_env("RETRY_429_MIN_DELAY_SECONDS", "8")),
            openalex_base_url=_get_env("OPENALEX_BASE_URL", "https://api.openalex.org"),
            wdqs_endpoint=_get_env("WDQS_ENDPOINT", "https://query.wikidata.org/sparql"),
            wikidata_api_url=_get_env("WIKIDATA_API_URL", "https://www.wikidata.org/w/api.php"),
            commons_api_url=_get_env("COMMONS_API_URL", "https://commons.wikimedia.org/w/api.php"),
            openalex_mailto=os.getenv("OPENALEX_MAILTO"),
            allow_name_fallback=_bool_env("ALLOW_NAME_FALLBACK", False),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_base_url=_get_env("LLM_BASE_URL", "https://api.openai.com"),
            llm_model=_get_env("LLM_MODEL", "gpt-4o-mini"),
            llm_timeout_seconds=int(_get_env("LLM_TIMEOUT_SECONDS", "30")),
            websearch_max_results=int(_get_env("WEBSEARCH_MAX_RESULTS", "8")),
            search_provider=_get_env("SEARCH_PROVIDER", "hybrid"),
            qwen_base_url=_get_env("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"),
            qwen_response_path=_get_env("QWEN_RESPONSE_PATH", "/responses"),
            qwen_model=_get_env("QWEN_MODEL", "qwen3.5-plus"),
            qwen_enable_web_search=_bool_env("QWEN_ENABLE_WEB_SEARCH", True),
            qwen_enable_t2i_search=_bool_env("QWEN_ENABLE_T2I_SEARCH", True),
            qwen_max_candidates=int(_get_env("QWEN_MAX_CANDIDATES", "8")),
            qwen_min_confidence=float(_get_env("QWEN_MIN_CONFIDENCE", "0.55")),
            qwen_timeout_seconds=int(_get_env("QWEN_TIMEOUT_SECONDS", "30")),
            person_page_query_max=int(_get_env("PERSON_PAGE_QUERY_MAX", "7")),
            person_page_per_query_results=int(_get_env("PERSON_PAGE_PER_QUERY_RESULTS", "5")),
            person_page_max_fetch=int(_get_env("PERSON_PAGE_MAX_FETCH", "12")),
            profile_image_score_threshold=float(_get_env("PROFILE_IMAGE_SCORE_THRESHOLD", "0.35")),
            refresh_ok_days=int(_get_env("REFRESH_OK_DAYS", "90")),
            refresh_no_image_days=int(_get_env("REFRESH_NO_IMAGE_DAYS", "30")),
            refresh_error_days=int(_get_env("REFRESH_ERROR_DAYS", "1")),
            refresh_ambiguous_days=int(_get_env("REFRESH_AMBIGUOUS_DAYS", "90")),
            refresh_no_match_days=int(_get_env("REFRESH_NO_MATCH_DAYS", "90")),
        )
