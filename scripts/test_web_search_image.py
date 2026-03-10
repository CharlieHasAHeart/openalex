#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


def _load_dotenv(path: str = ".env") -> None:
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
            if key and key not in os.environ:
                os.environ[key] = value


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = (text or "").strip()
    if not stripped:
        return None
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _collect_urls(obj: Any) -> list[str]:
    urls: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key.lower() in {"image_url", "url", "source_url"} and isinstance(value, str) and value.startswith("http"):
                    urls.append(value)
                walk(value)
            return
        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _collect_urls_from_tool_calls(payload: dict[str, Any]) -> list[str]:
    output = payload.get("output")
    if not isinstance(output, list):
        return []
    urls: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "") != "web_search_image_call":
            continue
        raw = item.get("output")
        if not raw:
            continue
        try:
            rows = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            url = str(row.get("url") or row.get("image_url") or "").strip()
            if url.startswith("http"):
                urls.append(url)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    return deduped


def _parse_json_text(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        return text


def build_prompt(
    author_id: str,
    display_name: str,
    orcid: str | None,
    affiliations: Any,
    last_known_institutions: Any,
) -> str:
    ctx = {
        "author_id": author_id,
        "display_name": display_name,
        "orcid": orcid or "",
        "affiliations": affiliations,
        "last_known_institutions": last_known_institutions,
    }
    return (
        "Find candidate portrait/headshot images for this author using web_search_image.\n"
        "Focus on this exact author identity using name, ORCID, affiliations, and last known institutions.\n"
        "Prefer official/institutional sources and pages clearly tied to this author.\n"
        "Avoid logos, banners, icons, group photos, and unrelated same-name people.\n"
        "Return exactly one JSON object, no markdown.\n"
        'Schema: {"candidates":[{"image_url":"","source_url":"","reason":""}],"failure_reason":""}\n'
        f"author={json.dumps(ctx, ensure_ascii=False)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test qwen web_search_image connectivity and image retrieval.")
    parser.add_argument("--author-id", required=True, help="Author id")
    parser.add_argument("--display-name", required=True, help="Author display name")
    parser.add_argument("--orcid", default="", help="ORCID (optional)")
    parser.add_argument("--affiliations", default="[]", help="JSON string for affiliations")
    parser.add_argument("--last-known-institutions", default="[]", help="JSON string for last_known_institutions")
    parser.add_argument("--timeout", type=int, default=90, help="HTTP timeout seconds")
    parser.add_argument("--runs-dir", default="runs", help="Output runs directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_dotenv(".env")

    model = os.getenv("LLM_MODEL")
    base_url = os.getenv("LLM_BASE_URL")
    api_key = os.getenv("LLM_API_KEY")

    if not model or not base_url or not api_key:
        raise SystemExit("Missing env vars. Require: LLM_MODEL, LLM_BASE_URL, LLM_API_KEY")

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=max(5, args.timeout))
    prompt = build_prompt(
        args.author_id,
        args.display_name,
        args.orcid,
        _parse_json_text(args.affiliations),
        _parse_json_text(args.last_known_institutions),
    )

    response = client.responses.create(
        model=model,
        input=prompt,
        tools=[{"type": "web_search_image"}],
        extra_body={"enable_thinking": False},
    )

    payload = response.model_dump() if hasattr(response, "model_dump") else {}
    output_text = str(payload.get("output_text") or "").strip()

    urls: list[str] = []
    if output_text:
        print("=== output_text ===")
        print(output_text)
        obj = _extract_json_object(output_text)
        if obj is not None:
            urls.extend(_collect_urls(obj))

    tool_urls = _collect_urls_from_tool_calls(payload)
    if tool_urls:
        print(f"\n=== tool_call_urls ({len(tool_urls)}) ===")
        for url in tool_urls[:30]:
            print(url)
        urls.extend(tool_urls)

    deduped_urls: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped_urls.append(url)

    if not deduped_urls:
        print("No candidate image URLs found from output_text or web_search_image tool call output.")
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:4000])
        return 1

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    run_id = str(uuid.uuid4())
    run_dir = Path(args.runs_dir) / run_date / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "web_search_image_test.json"
    output_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_date": run_date,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input": {
                    "author_id": args.author_id,
                    "display_name": args.display_name,
                    "orcid": args.orcid,
                    "affiliations": _parse_json_text(args.affiliations),
                    "last_known_institutions": _parse_json_text(args.last_known_institutions),
                    "model": model,
                    "base_url": base_url,
                },
                "tool_call_urls": tool_urls,
                "extracted_urls": deduped_urls,
                "raw_response": payload,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n=== extracted_urls ({len(deduped_urls)}) ===")
    for url in deduped_urls:
        print(url)
    print(f"\nSaved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
