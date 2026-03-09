from __future__ import annotations

from avatar_pipeline.http import HttpClient
from avatar_pipeline.models import ImageCandidate


class CommonsClient:
    def __init__(self, api_url: str, http: HttpClient) -> None:
        self._api_url = api_url
        self._http = http

    def get_image_candidate(self, commons_file: str, thumb_width: int) -> ImageCandidate | None:
        params = {
            "action": "query",
            "format": "json",
            "prop": "imageinfo",
            "titles": f"File:{commons_file}",
            "iiprop": "url|size|mime",
            "iiurlwidth": str(thumb_width),
        }
        payload = self._http.request("GET", self._api_url, params=params).json()
        pages = (payload.get("query") or {}).get("pages") or {}
        page = next(iter(pages.values()), None)
        if not page:
            return None

        imageinfo = (page.get("imageinfo") or [None])[0]
        if not imageinfo:
            return None

        download_url = imageinfo.get("thumburl") or imageinfo.get("url")
        if not download_url:
            return None

        return ImageCandidate(
            commons_file=commons_file,
            download_url=download_url,
            mime=imageinfo.get("mime", ""),
            width=int(imageinfo.get("thumbwidth") or imageinfo.get("width") or 0),
            height=int(imageinfo.get("thumbheight") or imageinfo.get("height") or 0),
            size_bytes=int(imageinfo.get("size") or 0),
        )

    def download_image(self, url: str) -> bytes:
        response = self._http.request("GET", url, stream=False)
        return response.content
