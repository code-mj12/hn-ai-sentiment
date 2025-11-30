"""Send sentiment payloads to OpenRouter with reasoning-enabled double checks.

This script expects a JSONL file produced by `sentiment_preprocess.py`. For each
record it performs two calls:
  1. Initial reasoning-enabled response.
  2. Follow-up "Are you sure?" prompt that includes the model's prior
     reasoning_details so it can continue the chain of thought.

The results (initial + final responses) are saved to a JSONL file for later
analysis.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "x-ai/grok-4.1-fast:free"
DEFAULT_INPUT = "sentiment_llm_payload.jsonl"
DEFAULT_OUTPUT = "sentiment_llm_results.jsonl"
DEFAULT_LIMIT: Optional[int] = None
DEFAULT_PER_STORY_BATCH_SIZE = 50

FOLLOWUP_PROMPT = (
    "Are you sure? Think carefully, verify the JSON schema, and revise if needed before you reply."
)


def read_api_key(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"API key file not found: {path}")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError(f"API key file {path} is empty")
    return key


def load_payloads(path: Path) -> List[Dict]:
    payloads = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payloads.append(json.loads(line))
    if not payloads:
        raise ValueError(f"No payloads found in {path}")
    return payloads


def parse_story_ids(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    story_ids: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            story_ids.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid story id '{token}' in --story-ids") from None
    if not story_ids:
        return None
    return story_ids


def order_story_payloads(payloads: List[Dict], story_ids: Optional[Sequence[int]] = None) -> OrderedDict:
    grouped: "OrderedDict[int, List[Dict]]" = OrderedDict()
    for payload in payloads:
        story_id = payload.get("root_story_id")
        if story_id is None:
            continue
        try:
            story_id_int = int(story_id)
        except (TypeError, ValueError):
            continue
        if story_id_int not in grouped:
            grouped[story_id_int] = []
        grouped[story_id_int].append(payload)

    if story_ids:
        ordered = OrderedDict()
        for story_id in story_ids:
            if story_id in grouped:
                ordered[story_id] = grouped[story_id]
        return ordered

    return grouped


def chunk_payloads(items: List[Dict], size: int) -> List[List[Dict]]:
    if size <= 0 or size >= len(items):
        return [items]
    return [items[idx : idx + size] for idx in range(0, len(items), size)]


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
    }
    return json.dumps(summary, ensure_ascii=False, indent=2)


def call_openrouter(token: str, model: str, messages: List[Dict], reasoning: bool, timeout: int = 60) -> Dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": reasoning},
    }
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not data.get("choices"):
        raise RuntimeError(f"OpenRouter response missing choices: {data}")
    return data["choices"][0]["message"]


def process_payload(payload: Dict, token: str, model: str, sleep: float) -> Dict:
    system_prompt = payload.get("prompt") or "You are an expert sentiment analyst."
    user_content = format_user_message(payload)

    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    first_message = call_openrouter(token, model, base_messages, reasoning=True)

    second_messages = base_messages + [
        {
            "role": "assistant",
            "content": first_message.get("content", ""),
            "reasoning_details": first_message.get("reasoning_details"),
        },
        {"role": "user", "content": FOLLOWUP_PROMPT},
    ]

    final_message = call_openrouter(token, model, second_messages, reasoning=True)

    result = {
        "id": payload.get("id"),
        "root_story_id": payload.get("root_story_id"),
        "initial_response": first_message.get("content"),
        "initial_reasoning": first_message.get("reasoning_details"),
        "final_response": final_message.get("content"),
        "final_reasoning": final_message.get("reasoning_details"),
        "model": model,
    }

    if sleep > 0:
        time.sleep(sleep)

    return result


def write_results(results: List[Dict], path: Path, append: bool = False) -> None:
    if not results:
        return
    mode = "a" if append and path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    action = "Appended" if mode == "a" else "Stored"
    print(f"{action} {len(results)} annotated rows in {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenRouter sentiment analysis on prepared payloads")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="JSONL payload file from sentiment_preprocess")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Where to store model responses (JSONL)")
    parser.add_argument("--api-key-path", default="key", help="Path to file containing the OpenRouter API key")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenRouter model identifier")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of payloads to send overall (omit for full story coverage)",
    )
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to sleep between calls (avoid rate limits)")
    parser.add_argument(
        "--per-story-batch-size",
        type=int,
        default=DEFAULT_PER_STORY_BATCH_SIZE,
        help="Chunk each root story into batches of this size (0 = all comments at once)",
    )
    parser.add_argument(
        "--story-ids",
        default=None,
        help="Comma-separated list of root_story_id values to process (preserves provided order)",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Limit how many distinct root stories to process in this run",
    )
    parser.add_argument(
        "--append-output",
        action="store_true",
        help="Append to the output file instead of overwriting it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = read_api_key(Path(args.api_key_path))
    payloads = load_payloads(Path(args.input))
    selected_story_ids = parse_story_ids(args.story_ids)
    ordered_payloads = order_story_payloads(payloads, selected_story_ids)

    if not ordered_payloads:
        raise ValueError("No payloads matched the provided story filters")

    processed_payloads = 0
    processed_stories = 0
    story_batches: List[Dict] = []

    for story_order_index, (story_id, story_payloads) in enumerate(ordered_payloads.items()):
        if args.max_stories is not None and processed_stories >= args.max_stories:
            break
        batches = chunk_payloads(story_payloads, args.per_story_batch_size)
        batch_count = len(batches)
        for batch_index, batch_payloads in enumerate(batches):
            story_batches.append(
                {
                    "story_id": story_id,
                    "story_order_index": story_order_index,
                    "batch_index": batch_index,
                    "batch_count": batch_count,
                    "payloads": batch_payloads,
                }
            )
        processed_stories += 1

    if not story_batches:
        raise ValueError("No batches prepared; check story filters or batch size")

    limit = args.limit if (args.limit is not None and args.limit > 0) else None
    results = []
    total_batches = len(story_batches)

    for batch_number, batch in enumerate(story_batches, start=1):
        story_id = batch["story_id"]
        print(
            f"Story {story_id} batch {batch['batch_index'] + 1}/{batch['batch_count']}"
            f" ({len(batch['payloads'])} payloads) [{batch_number}/{total_batches}]"
        )
        for payload in batch["payloads"]:
            if limit is not None and processed_payloads >= limit:
                print("Reached global --limit; stopping further processing.")
                write_results(results, Path(args.output), append=args.append_output)
                return
            try:
                result = process_payload(payload, api_key, args.model, args.sleep)
                result["story_batch_index"] = batch["batch_index"]
                result["story_batch_count"] = batch["batch_count"]
                result["story_order_index"] = batch["story_order_index"]
                results.append(result)
                processed_payloads += 1
            except Exception as exc:  # noqa: BLE001
                print(f"Failed to process payload id={payload.get('id')}: {exc}")
                write_results(results, Path(args.output), append=args.append_output)
                return

    write_results(results, Path(args.output), append=args.append_output)


if __name__ == "__main__":
    main()
