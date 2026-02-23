from __future__ import annotations

import time
from typing import Any

import requests


class RateLimiter:
    def __init__(self, qps: float) -> None:
        self._interval = 1.0 / qps if qps > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self._interval <= 0:
            return
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
    ) -> None:
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._rate_limiter = rate_limiter
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "openalex-avatar-pipeline/1.0 (+https://openalex.org)",
            }
        )

    def request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        for attempt in range(1, self._max_retries + 1):
            if self._rate_limiter:
                self._rate_limiter.wait()
            try:
                resp = self._session.request(method, url, timeout=self._timeout, **kwargs)
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt == self._max_retries:
                        resp.raise_for_status()
                    time.sleep(min(2 ** (attempt - 1), 8))
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException:
                if attempt == self._max_retries:
                    raise
                time.sleep(min(2 ** (attempt - 1), 8))
        raise RuntimeError("Unreachable retry branch")
