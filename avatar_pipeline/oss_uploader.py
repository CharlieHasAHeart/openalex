from __future__ import annotations

import hashlib

import oss2

from avatar_pipeline.avatar_gate import mime_to_ext


def sha1_hex(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class OssUploader:
    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        bucket_name: str,
        endpoint: str,
        public_base_url: str,
        key_prefix: str,
        cache_control: str | None = None,
    ) -> None:
        auth = oss2.Auth(access_key_id, access_key_secret)
        self._bucket = oss2.Bucket(auth, endpoint, bucket_name)
        self._public_base_url = public_base_url.rstrip("/")
        self._cache_control = cache_control
        self._key_prefix = key_prefix.strip("/")

    def build_object_key(self, orcid: str, sha256: str, mime: str) -> str:
        orcid_key = orcid.strip().replace("/", "_")
        ext = mime_to_ext(mime)
        prefix = f"{self._key_prefix}/" if self._key_prefix else ""
        return f"{prefix}authors/{orcid_key}/{sha256}.{ext}"

    def upload(self, object_key: str, content: bytes, mime: str) -> str:
        headers = {"Content-Type": mime}
        if self._cache_control:
            headers["Cache-Control"] = self._cache_control
        self._bucket.put_object(object_key, content, headers=headers)
        return f"{self._public_base_url}/{object_key}"
