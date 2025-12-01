"""Run per-story OpenRouter sentiment analysis with per-comment outputs.

This script consumes the JSONL payload emitted by ``sentiment_preprocess.py``
but sends one OpenRouter request *per story batch* instead of per comment. Each
API call contains every comment (or chunk) for the story, instructing the model
to return a JSON object with an aligned result for each payload id. The saved
results still remain per-comment granularity, while the number of API calls is
reduced by up to ~50x.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "x-ai/grok-4.1-fast:free"
DEFAULT_INPUT = "sentiment_llm_payload.jsonl"
DEFAULT_OUTPUT = "sentiment_llm_results.jsonl"
DEFAULT_LIMIT: Optional[int] = None
DEFAULT_PER_STORY_BATCH_SIZE = 50
MAX_TEXT_CHARS = 400

STORY_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "story_id": {"type": "integer"},
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "neutral", "negative", "mixed"],
                    },
                    "aspects": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "aspect": {"type": "string"},
                                "present": {"type": "boolean"},
                                "evidence": {"type": ["string", "null"]},
                                "sentiment": {"type": ["string", "null"]},
                                "confidence": {"type": ["number", "null"]},
                                "implicit": {"type": ["boolean", "null"]},
                            },
                            "required": ["aspect", "present"],
                        },
                    },
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["id", "sentiment", "aspects"],
            },
        },
    },
    "required": ["story_id", "results"],
}

STORY_SYSTEM_PROMPT = (
    "You are an expert sentiment analyst. You will receive metadata for one Hacker News root "
    "story along with a list of comment payloads. For *each* payload you must issue an "
    "independent judgment and keep the order exactly as provided."
    "\nReturn STRICT JSON matching this schema (no prose):\n"
    "{"
    "\n  \"story_id\": <int>,"
    "\n  \"results\": ["
    "\n    {"
    "\n      \"id\": <payload id>,"
    "\n      \"sentiment\": \"positive|neutral|negative|mixed\","
    "\n      \"aspects\": ["
    "\n        {"
    "\n          \"aspect\": <string>,"
    "\n          \"present\": true|false,"
    "\n          \"evidence\": <string|null>,"
    "\n          \"sentiment\": <sentiment|null>,"
    "\n          \"confidence\": <0-1>,"
    "\n          \"implicit\": true|false"
    "\n        }"
    "\n      ]"
    "\n    }"
    "\n  ]"
    "\n}"
    "\nAlways ensure the number of results equals the number of payloads you received and that the ids match."
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


def call_openrouter(
    token: str,
    model: str,
    messages: List[Dict],
    reasoning: bool,
    timeout: int = 60,
    response_format: Optional[Dict] = None,
) -> Dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": reasoning},
    }
    if response_format:
        payload["response_format"] = response_format
    response = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not data.get("choices"):
        raise RuntimeError(f"OpenRouter response missing choices: {data}")
    return data["choices"][0]["message"]


def truncate_text(value: Optional[str], limit: int = MAX_TEXT_CHARS) -> str:
    if not value:
        return ""
    value = value.strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def format_story_batch_message(story_id: int, payloads: List[Dict]) -> str:
    taxonomy = payloads[0].get("taxonomy") if payloads else []
    root_ctx = payloads[0].get("context", {}) if payloads else {}
    story_meta = {
        "id": story_id,
        "title": root_ctx.get("root_title"),
        "author": root_ctx.get("root_author"),
        "url": root_ctx.get("root_url"),
    }
    comments = []
    for order_index, payload in enumerate(payloads, start=1):
        ctx = payload.get("context", {}) or {}
        comments.append(
            {
                "payload_id": payload.get("id"),
                "order": order_index,
                "node_type": payload.get("node_type"),
                "comment_depth": payload.get("comment_depth"),
                "model_tags": payload.get("model_tags", []),
                "aspect_hints": payload.get("aspect_hints", []),
                "author": ctx.get("node_author"),
                "published": ctx.get("node_time"),
                "text": truncate_text(payload.get("normalized_text") or payload.get("text")),
            }
        )

    instructions = (
        f"Analyze each comment independently (total payloads: {len(payloads)}). Use the taxonomy for aspect names. "
        "Set present=false when an aspect is not referenced. Maintain the incoming order and ids."
    )

    message = {
        "task": instructions,
        "root_story": story_meta,
        "taxonomy": taxonomy,
        "comment_count": len(payloads),
        "comments": comments,
    }
    return json.dumps(message, ensure_ascii=False, indent=2)


def parse_json_block(raw: Optional[str]) -> Dict:
    if not raw:
        raise ValueError("Model response was empty")
    candidate = raw.strip()
    if not candidate:
        raise ValueError("Model response was blank")
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Could not locate JSON object in model response")
        return json.loads(candidate[start : end + 1])


def parse_story_response(raw_content: Optional[str], story_id: int, payloads: List[Dict]) -> List[Dict]:
    parsed = parse_json_block(raw_content)
    if int(parsed.get("story_id", story_id)) != int(story_id):
        raise ValueError("Model response story_id mismatch")
    results = parsed.get("results")
    if not isinstance(results, list):
        raise ValueError("Model response missing 'results' array")

    id_to_payload = {int(p.get("id")): p for p in payloads if p.get("id") is not None}
    cleaned_results: List[Dict] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        result_id = result.get("id")
        if result_id is None:
            continue
        try:
            result_id = int(result_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid result id: {result.get('id')}") from exc
        if result_id not in id_to_payload:
            continue
        cleaned_results.append(result)

    ordered_results: List[Dict] = []
    for payload in payloads:
        pid = payload.get("id")
        if pid is None:
            continue
        pid_int = int(pid)
        match = next((res for res in cleaned_results if int(res.get("id")) == pid_int), None)
        if match is None:
            raise ValueError(f"Model response missing payload id {pid_int}")
        ordered_results.append(match)

    return ordered_results


def process_story_batch(
    story_id: int,
    batch_payloads: List[Dict],
    token: str,
    model: str,
    sleep: float,
) -> Dict[str, List[Dict]]:
    user_content = format_story_batch_message(story_id, batch_payloads)
    messages = [
        {"role": "system", "content": STORY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response_message = call_openrouter(
        token,
        model,
        messages,
        reasoning=True,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "story_batch", "schema": STORY_RESPONSE_SCHEMA},
        },
    )
    parsed_results = parse_story_response(response_message.get("content"), story_id, batch_payloads)
    if sleep > 0:
        time.sleep(sleep)
    return {
        "raw_response": response_message.get("content"),
        "results": parsed_results,
    }


def build_comment_record(
    payload: Dict,
    result: Dict,
    batch_meta: Dict,
    payload_index: int,
    model: str,
    raw_story_response: Optional[str],
) -> Dict:
    return {
        "id": payload.get("id"),
        "root_story_id": payload.get("root_story_id"),
        "story_id": batch_meta["story_id"],
        "story_batch_index": batch_meta["batch_index"],
        "story_batch_count": batch_meta["batch_count"],
        "story_order_index": batch_meta["story_order_index"],
        "story_payload_index": payload_index,
        "model": model,
        "final_response": json.dumps(result, ensure_ascii=False),
        "raw_story_response": raw_story_response,
    }


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
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="How many times to retry a failed story batch before aborting",
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
        if limit is not None and processed_payloads >= limit:
            print("Reached global --limit; stopping further processing.")
            break

        story_id = batch["story_id"]
        payloads_for_batch = batch["payloads"]
        if limit is not None:
            remaining = limit - processed_payloads
            if remaining <= 0:
                print("Reached global --limit; stopping further processing.")
                break
            if remaining < len(payloads_for_batch):
                payloads_for_batch = payloads_for_batch[:remaining]

        print(
            f"Story {story_id} batch {batch['batch_index'] + 1}/{batch['batch_count']}"
            f" ({len(payloads_for_batch)} payloads) [{batch_number}/{total_batches}]"
        )

        attempt = 0
        while True:
            try:
                story_output = process_story_batch(story_id, payloads_for_batch, api_key, args.model, args.sleep)
                break
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt >= args.max_retries:
                    print(
                        f"Failed to process story {story_id} batch {batch['batch_index'] + 1} after "
                        f"{args.max_retries} attempts: {exc}"
                    )
                    write_results(results, Path(args.output), append=args.append_output)
                    return
                backoff = min(args.sleep * attempt, 5)
                print(
                    f"Error for story {story_id} batch {batch['batch_index'] + 1}: {exc}. "
                    f"Retrying in {backoff:.1f}s ({attempt}/{args.max_retries})."
                )
                if backoff > 0:
                    time.sleep(backoff)

        raw_story_response = story_output.get("raw_response")
        for payload_index, (payload, comment_result) in enumerate(
            zip(payloads_for_batch, story_output.get("results", [])), start=0
        ):
            if limit is not None and processed_payloads >= limit:
                print("Reached global --limit; stopping further processing.")
                write_results(results, Path(args.output), append=args.append_output)
                return
            record = build_comment_record(
                payload,
                comment_result,
                batch,
                payload_index,
                args.model,
                raw_story_response,
            )
            results.append(record)
            processed_payloads += 1

    write_results(results, Path(args.output), append=args.append_output)


if __name__ == "__main__":
    main()
