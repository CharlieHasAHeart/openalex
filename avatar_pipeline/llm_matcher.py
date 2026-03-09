from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import requests

from avatar_pipeline.models import AuthorRecord
from avatar_pipeline.web_search_client import SearchCandidate


@dataclass(slots=True)
class MatchDecision:
    selected_index: int
    confidence: float
    reason: str


class LlmMatcher:
    def __init__(self, api_key: str | None, base_url: str, model: str, timeout_seconds: int) -> None:
        self._api_key = (api_key or "").strip()
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = max(5, timeout_seconds)

    def _heuristic_match(self, author: AuthorRecord, candidates: list[SearchCandidate]) -> MatchDecision | None:
        name = author.display_name.lower().strip()
        orcid = (author.orcid or "").lower()
        best_idx = -1
        best_score = 0
        for idx, c in enumerate(candidates):
            text = (
                f"{c.title} {c.snippet} {c.source_url} {c.image_url} "
                f"{c.page_title or ''} {c.page_h1 or ''} {c.page_meta_description or ''} "
                f"{c.image_alt or ''} {c.nearby_text or ''}"
            ).lower()
            score = 0
            if name and name in text:
                score += 2
            if orcid and orcid in text:
                score += 3
            if author.institution_name and author.institution_name.lower() in text:
                score += 1
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx < 0 or best_score < 2:
            return None
        return MatchDecision(selected_index=best_idx, confidence=min(0.6 + 0.1 * best_score, 0.95), reason="heuristic")

    def choose_best(self, author: AuthorRecord, candidates: list[SearchCandidate]) -> MatchDecision | None:
        if not candidates:
            return None
        if not self._api_key:
            return self._heuristic_match(author, candidates)

        author_ctx: dict[str, Any] = {
            "author_id": author.author_id,
            "display_name": author.display_name,
            "orcid": author.orcid,
            "institution_name": author.institution_name,
            "profile": author.profile or {},
        }
        options = [
            {
                "index": i,
                "source_domain": c.source_domain,
                "source_url": c.source_url,
                "image_url": c.image_url,
                "page_title": c.page_title or c.title,
                "page_h1": c.page_h1,
                "page_meta_description": c.page_meta_description,
                "image_alt": c.image_alt,
                "nearby_text": c.nearby_text,
                "snippet": c.snippet,
                "pre_rank_score": c.pre_rank_score,
                "name_match_score": c.name_match_score,
                "institution_match_score": c.institution_match_score,
                "source_trust_score": c.source_trust_score,
            }
            for i, c in enumerate(candidates)
        ]
        prompt = (
            "You are selecting a scholar profile image candidate based on page context evidence.\n"
            "Given the author context and structured candidate evidence, return strict JSON only:\n"
            '{"selected_index": <int or -1>, "confidence": <0..1>, "reason": "<short>"}\n'
            "Choose -1 if uncertain or likely wrong person.\n"
            "Judge whether the page context indicates this image is used as the author's profile photo.\n"
            "Prioritize name match, institution match, and official profile/faculty page signals.\n"
            f"author={json.dumps(author_ctx, ensure_ascii=False)}\n"
            f"candidate_evidence={json.dumps(options, ensure_ascii=False)}"
        )

        payload = {
            "model": self._model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        resp = requests.post(
            f"{self._base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        if not content:
            return self._heuristic_match(author, candidates)
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            return self._heuristic_match(author, candidates)
        obj = json.loads(match.group(0))
        idx = int(obj.get("selected_index", -1))
        conf = float(obj.get("confidence", 0))
        reason = str(obj.get("reason", ""))
        if idx < 0 or idx >= len(candidates):
            return None
        if conf < 0.55:
            return None
        return MatchDecision(selected_index=idx, confidence=conf, reason=reason)
