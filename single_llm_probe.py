"""Send a single sentiment payload to OpenRouter and report elapsed time.

Useful for smoke-testing custom prompts or measuring per-call latency
without running the full sentiment runner.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "x-ai/grok-4.1-fast:free"
DEFAULT_INPUT = "sentiment_llm_payload.jsonl"
DEFAULT_API_KEY_PATH = "key"
DEFAULT_REFERER = "https://github.com/code-mj12/hn-ai-sentiment"
DEFAULT_TITLE = "HN AI Sentiment Probe"


def read_api_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file {path} is empty")
    return key


def load_payloads(path: Path) -> List[Dict]:
    payloads: List[Dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payloads.append(json.loads(line))
    if not payloads:
        raise ValueError(f"No payloads found in {path}")
    return payloads


def select_payload(payloads: List[Dict], payload_id: Optional[int], payload_index: int) -> Dict:
    if payload_id is not None:
        for payload in payloads:
            if int(payload.get("id")) == payload_id:
                return payload
        raise ValueError(f"Payload id {payload_id} not found in file")
    if payload_index < 0 or payload_index >= len(payloads):
        raise IndexError(f"payload-index {payload_index} is out of range (0-{len(payloads)-1})")
    return payloads[payload_index]


def build_conversation_view(payload: Dict) -> Dict:
    ctx = payload.get("context") or {}
    depth = payload.get("comment_depth") or 0
    try:
        depth = int(depth)
    except (TypeError, ValueError):
        depth = 0
    mode = ctx.get("conversation_role")
    if not mode:
        node_type = (payload.get("node_type") or "").lower()
        mode = "story" if node_type == "story" else ("direct_reply" if depth <= 1 else "thread_reply")

    story_section = {
        "id": payload.get("root_story_id"),
        "title": ctx.get("root_title", ""),
        "author": ctx.get("root_author", ""),
        "url": ctx.get("root_url", ""),
        "text": ctx.get("root_text", ""),
    }

    parent_section = None
    parent_text = ctx.get("parent_text") or ""
    parent_id = ctx.get("parent_id")
    if mode != "story" and (parent_text or parent_id is not None):
        parent_section = {
            "id": parent_id,
            "author": ctx.get("parent_author", ""),
            "type": ctx.get("parent_type", ""),
            "text": parent_text,
        }

    target_section = {
        "id": payload.get("id"),
        "type": payload.get("node_type"),
        "author": ctx.get("node_author", ""),
        "time": ctx.get("node_time", ""),
        "text": payload.get("text", ""),
        "normalized_text": payload.get("normalized_text", ""),
        "depth": depth,
        "lineage": ctx.get("lineage_to_story", ""),
    }

    return {
        "mode": mode,
        "story": story_section,
        "parent": parent_section,
        "target": target_section,
    }


def format_user_message(payload: Dict) -> str:
    summary = {
        "id": payload.get("id"),
        "root_story_id": payload.get("root_story_id"),
        "node_type": payload.get("node_type"),
        "comment_depth": payload.get("comment_depth"),
        "model_tags": payload.get("model_tags", []),
        "aspect_hints": payload.get("aspect_hints", []),
        "context": payload.get("context", {}),
        "text": payload.get("text", ""),
        "normalized_text": payload.get("normalized_text", ""),
        "taxonomy": payload.get("taxonomy", []),
        "conversation_view": build_conversation_view(payload),
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def call_openrouter(
    token: str,
    model: str,
    messages: List[Dict],
    reasoning: bool,
    referer: str,
    title: str,
) -> Dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "HTTP-Referer": referer,
        "X-Title": title,
    }
    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": reasoning},
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(
            f"OpenRouter request failed with status {response.status_code}: {response.text}"
        ) from exc
    data = response.json()
    if not data.get("choices"):
        raise RuntimeError(f"OpenRouter response missing choices: {data}")
    return data["choices"][0]["message"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a single LLM payload to OpenRouter")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSONL payload file from sentiment_preprocess")
    parser.add_argument("--payload-id", type=int, default=None, help="Specific payload id to send")
    parser.add_argument(
        "--payload-index",
        type=int,
        default=0,
        help="Zero-based index of the payload to send when --payload-id is not provided",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model identifier")
    parser.add_argument("--api-key-path", default=DEFAULT_API_KEY_PATH, help="Path to the OpenRouter API key file")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Override the payload's system prompt (defaults to prompt stored in payload)",
    )
    parser.add_argument(
        "--no-reasoning",
        action="store_true",
        help="Disable the reasoning flag for the request",
    )
    parser.add_argument(
        "--referer",
        default=DEFAULT_REFERER,
        help="HTTP referer header value required by OpenRouter",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help="X-Title header value reported to OpenRouter",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = read_api_key(Path(args.api_key_path))
    payloads = load_payloads(Path(args.input))
    payload = select_payload(payloads, args.payload_id, args.payload_index)

    system_prompt = args.prompt or payload.get("prompt") or "You are an expert sentiment analyst."
    user_content = format_user_message(payload)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    start = time.perf_counter()
    response_message = call_openrouter(
        token,
        args.model,
        messages,
        reasoning=not args.no_reasoning,
        referer=args.referer,
        title=args.title,
    )
    elapsed = time.perf_counter() - start

    print("=== OpenRouter response ===")
    print(response_message.get("content", "<no content>"))
    if response_message.get("reasoning_details"):
        print("\nReasoning details:")
        print(response_message["reasoning_details"])
    print(f"\nElapsed time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
