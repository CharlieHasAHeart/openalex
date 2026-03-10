from __future__ import annotations

import random
import time
from threading import Lock
from typing import Any

import requests


class RateLimiter:
    def __init__(self, qps: float) -> None:
        self._interval = 1.0 / qps if qps > 0 else 0.0
        self._last = 0.0
        self._lock = Lock()

    def wait(self) -> None:
        if self._interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = now - self._last
            if delta < self._interval:
                time.sleep(self._interval - delta)
            self._last = time.monotonic()


class HttpClient:
    def __init__(
        self,
        timeout_seconds: int,
        max_retries: int,
        rate_limiter: RateLimiter | None = None,
        retry_base_seconds: float = 1.0,
        retry_max_seconds: float = 30.0,
        retry_jitter_ratio: float = 0.2,
        retry_429_min_delay_seconds: float = 5.0,
    ) -> None:
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._rate_limiter = rate_limiter
        self._retry_base_seconds = max(retry_base_seconds, 0.1)
        self._retry_max_seconds = max(retry_max_seconds, self._retry_base_seconds)
        self._retry_jitter_ratio = max(retry_jitter_ratio, 0.0)
        self._retry_429_min_delay_seconds = max(retry_429_min_delay_seconds, 0.0)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "openalex-avatar-pipeline/1.0 (+https://openalex.org)"})

    def _sleep_before_retry(self, attempt: int, resp: requests.Response | None = None) -> None:
        delay = min(self._retry_base_seconds * (2 ** (attempt - 1)), self._retry_max_seconds)
        if resp is not None and resp.status_code == 429:
            retry_after_raw = resp.headers.get("Retry-After")
            if retry_after_raw:
                try:
                    delay = max(delay, float(retry_after_raw.strip()))
                except ValueError:
                    pass
            delay = max(delay, self._retry_429_min_delay_seconds)
        jitter = delay * self._retry_jitter_ratio * random.random()
        time.sleep(delay + jitter)

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        timeout = kwargs.pop("timeout", self._timeout)
        max_retries = int(kwargs.pop("max_retries", self._max_retries))
        max_retries = max(1, max_retries)
        for attempt in range(1, max_retries + 1):
            if self._rate_limiter:
                self._rate_limiter.wait()
            try:
                resp = self._session.request(method, url, timeout=timeout, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt == max_retries:
                        resp.raise_for_status()
                    self._sleep_before_retry(attempt, resp=resp)
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException:
                if attempt == max_retries:
                    raise
                self._sleep_before_retry(attempt)
        raise RuntimeError("Unreachable retry branch")
