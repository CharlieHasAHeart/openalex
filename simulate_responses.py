from __future__ import annotations

import argparse
import os

from openai import OpenAI

from avatar_pipeline.config import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Qwen /responses connectivity test")
    parser.add_argument("--env-file", default=".env", help="Path to env file")
    parser.add_argument("--input", default="你能做些什么？", help="Input text for the model")
    parser.add_argument("--model", default="", help="Override model from env")
    parser.add_argument("--base-url", default="", help="Override base_url from env")
    parser.add_argument("--api-key", default="", help="Override API key from env")
    return parser.parse_args()


def _get_env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def main() -> int:
    args = parse_args()
    load_dotenv(args.env_file)

    api_key = args.api_key.strip() or _get_env("LLM_API_KEY")
    if not api_key:
        raise SystemExit("Missing LLM_API_KEY in env")

    base_url = args.base_url.strip() or _get_env("LLM_BASE_URL")
    if not base_url:
        raise SystemExit("Missing LLM_BASE_URL in env")

    model = args.model.strip() or _get_env("LLM_MODEL")
    if not model:
        raise SystemExit("Missing LLM_MODEL in env")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    response = client.responses.create(
        model=model,
        input=args.input,
    )

    print(response.output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
