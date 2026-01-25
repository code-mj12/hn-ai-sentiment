#!/usr/bin/env python3
"""
V3 OPTIMIZED: SINGLE LLM CALL sentiment analysis pipeline

Works with v2 payloads from prepare_sentiment_payloads_v2.py

Single execution: Stage 1 (aspect + sentiment) → Confidence Filter → Final output

OPTIMIZATIONS (v3):
- Combined aspect detection + sentiment into single LLM call (50% fewer API calls)
- Sentiment derived from rating (1-2=negative, 3=neutral, 4-5=positive)
- Removed redundant Stage 2 LLM call

FEATURES:
- parse_confidence() handles string values ("high", "medium", "low")
- 6-aspect strict taxonomy filtering
- Batched LLM requests for efficiency
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests

# InnKube endpoint configuration
INNKUBE_URL = "https://llms.innkube.fim.uni-passau.de/v1/chat/completions"
ALLOWED_MODELS = ["qwen3-next-80b-a3b-instruct", "webthinker-qwq-32b"]
DEFAULT_MODEL = "qwen3-next-80b-a3b-instruct"
DEFAULT_TIMEOUT = 180  # seconds (increased for reliability)
DEFAULT_MIN_CONFIDENCE = 0.7

# STRICT TAXONOMY: Only these 6 aspects allowed
ALLOWED_ASPECTS = {
    "coding_speed",         # Speed, latency, execution time
    "coding_performance",   # Quality, correctness, accuracy
    "cost_price",          # Pricing, compute costs, ROI
    "ethics",              # Bias, fairness, transparency
    "innovation",          # Novelty, breakthroughs, SOTA
    "skepticism_hype"      # Doubt, feasibility concerns
}

# AI MODEL NORMALIZATION: Map variations to canonical names
MODEL_ALIASES = {
    # GPT family
    "gpt-4": "gpt-4", "gpt4": "gpt-4", "gpt 4": "gpt-4", "gpt-4o": "gpt-4",
    "gpt-4-turbo": "gpt-4", "gpt4o": "gpt-4", "gpt-4o-mini": "gpt-4",
    "gpt-3.5": "gpt-3.5", "gpt3.5": "gpt-3.5", "gpt-3.5-turbo": "gpt-3.5",
    "chatgpt": "chatgpt", "chat gpt": "chatgpt",
    "openai": "openai", "o1": "o1", "o1-preview": "o1", "o1-mini": "o1",
    "o3": "o3", "o3-mini": "o3",
    # Claude family
    "claude": "claude", "claude-3": "claude", "claude 3": "claude",
    "claude-3.5": "claude", "claude-3-opus": "claude", "claude-3-sonnet": "claude",
    "claude-opus": "claude", "claude-sonnet": "claude", "claude-haiku": "claude",
    "anthropic": "claude",
    # Gemini family
    "gemini": "gemini", "gemini-pro": "gemini", "gemini-ultra": "gemini",
    "gemini 2.0": "gemini", "gemini 1.5": "gemini", "bard": "gemini",
    # Llama family
    "llama": "llama", "llama-3": "llama", "llama3": "llama", "llama 3": "llama",
    "llama-2": "llama", "llama2": "llama", "meta llama": "llama",
    # DeepSeek family
    "deepseek": "deepseek", "deepseek-r1": "deepseek", "deepseek r1": "deepseek",
    "deepseek-v3": "deepseek", "deepseek-coder": "deepseek",
    # Mistral family
    "mistral": "mistral", "mixtral": "mistral", "mistral-large": "mistral",
    # Qwen family
    "qwen": "qwen", "qwen-2": "qwen", "qwen2": "qwen",
    # Copilot
    "copilot": "copilot", "github copilot": "copilot",
    # Cursor
    "cursor": "cursor",
    # Codeium
    "codeium": "codeium",
    # Other
    "codellama": "codellama", "code llama": "codellama",
    "phi": "phi", "phi-3": "phi", "phi-4": "phi",
    "grok": "grok",
    "perplexity": "perplexity",
    "cohere": "cohere", "command-r": "cohere",
}

def normalize_model(model_name: Any) -> str:
    """Normalize AI model name to canonical form."""
    if not model_name or not isinstance(model_name, str):
        return "unknown"
    name_lower = model_name.lower().strip()
    # Direct match
    if name_lower in MODEL_ALIASES:
        return MODEL_ALIASES[name_lower]
    # Partial match
    for alias, canonical in MODEL_ALIASES.items():
        if alias in name_lower or name_lower in alias:
            return canonical
    return model_name.lower().replace(" ", "-")[:20]  # Truncate unknown models


def parse_confidence(conf_val: Any) -> float:
    """Convert confidence to float (handle 'high'/'medium'/'low' strings)."""
    if isinstance(conf_val, (int, float)):
        return float(conf_val)
    if isinstance(conf_val, str):
        val_lower = conf_val.lower().strip()
        if val_lower in ('high', 'very high', '1.0', '1'):
            return 0.9
        elif val_lower in ('medium', '0.5', '0'):
            return 0.7
        elif val_lower in ('low', 'very low', '0.0'):
            return 0.5
    return 0.7  # default fallback


def parse_rating(rating_val: Any) -> int:
    """Convert rating to int (1-5 scale)."""
    if isinstance(rating_val, int):
        return max(1, min(5, rating_val))
    if isinstance(rating_val, float):
        return max(1, min(5, int(round(rating_val))))
    if isinstance(rating_val, str):
        try:
            return max(1, min(5, int(float(rating_val))))
        except ValueError:
            pass
    return 3  # default middle rating


def derive_sentiment(rating: int) -> str:
    """Derive sentiment from rating (1-5 scale)."""
    if rating <= 2:
        return "negative"
    elif rating == 3:
        return "neutral"
    else:
        return "positive"


def normalize_aspects(aspects: Any) -> List[Dict[str, Any]]:
    """Normalize LLM Stage 1 output to consistent format and FILTER to taxonomy only."""
    normalized: List[Dict[str, Any]] = []
    if not aspects:
        return normalized

    if isinstance(aspects, dict):
        for k, v in aspects.items():
            aspect_name = str(k).lower().replace(" ", "_")
            # STRICT: Only include if in allowed taxonomy
            if aspect_name not in ALLOWED_ASPECTS:
                continue
            try:
                present = bool(v)
            except Exception:
                present = True
            rating = 3  # default rating
            normalized.append({
                "aspect": aspect_name,
                "present": present,
                "rating": rating,
                "sentiment": derive_sentiment(rating),
                "evidence": "",
                "confidence": 0.7 if present else 0.5,
            })
        return normalized

    if isinstance(aspects, list):
        for item in aspects:
            if isinstance(item, dict):
                aspect_name = item.get("aspect") or item.get("name") or item.get("type")
                if not aspect_name and len(item) == 1:
                    k = next(iter(item.keys()))
                    v = item[k]
                    aspect_name_normalized = str(k).lower().replace(" ", "_")
                    # STRICT: Only include if in allowed taxonomy
                    if aspect_name_normalized not in ALLOWED_ASPECTS:
                        continue
                    rating = parse_rating(item.get("rating", 3))
                    normalized.append({
                        "aspect": aspect_name_normalized,
                        "present": bool(v),
                        "rating": rating,
                        "sentiment": item.get("sentiment") or derive_sentiment(rating),
                        "evidence": item.get("evidence", ""),
                        "confidence": parse_confidence(item.get("confidence", 0.7)),
                    })
                    continue

                aspect_name_normalized = str(aspect_name).lower().replace(" ", "_") if aspect_name else "unknown"
                # STRICT: Only include if in allowed taxonomy
                if aspect_name_normalized not in ALLOWED_ASPECTS:
                    continue

                rating = parse_rating(item.get("rating", 3))
                normalized.append({
                    "aspect": aspect_name_normalized,
                    "present": bool(item.get("present", True)),
                    "rating": rating,
                    "sentiment": item.get("sentiment") or derive_sentiment(rating),
                    "evidence": item.get("evidence", ""),
                    "confidence": parse_confidence(item.get("confidence", 0.7)),
                })
            elif isinstance(item, str):
                aspect_name_normalized = str(item).lower().replace(" ", "_")
                # STRICT: Only include if in allowed taxonomy
                if aspect_name_normalized not in ALLOWED_ASPECTS:
                    continue
                rating = 3  # default rating
                normalized.append({
                    "aspect": aspect_name_normalized,
                    "present": True,
                    "rating": rating,
                    "sentiment": derive_sentiment(rating),
                    "evidence": "",
                    "confidence": 0.7,
                })
        return normalized

    return normalized


def load_api_key(key_path: Path) -> str:
    """Load API key from file."""
    with open(key_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def call_innkube(
    messages: list,
    model: str,
    api_key: str,
    timeout: int = DEFAULT_TIMEOUT
) -> Dict[str, Any]:
    """Call InnKube chat completions endpoint."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.3
    }

    try:
        resp = requests.post(
            INNKUBE_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )
        resp.raise_for_status()

        data = resp.json()
        content = data["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(content)
            return {"response": parsed, "error": None}
        except json.JSONDecodeError as e:
            return {
                "response": None,
                "error": f"JSON decode error: {e}. Raw: {content[:200]}"
            }

    except requests.exceptions.Timeout:
        return {"response": None, "error": f"Request timeout after {timeout}s"}
    except requests.exceptions.RequestException as e:
        return {"response": None, "error": f"Request error: {e}"}
    except Exception as e:
        return {"response": None, "error": f"Unexpected error: {e}"}


def build_stage1_batch_messages(batch: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Construct single Stage 1 request for a batch (combined aspect detection + sentiment)."""
    batch_size = len(batch)
    instructions = [
        "You are an AI discussion analyzer for Hacker News comments.",
        f"Process ALL {batch_size} items. Return JSON ARRAY of EXACTLY {batch_size} elements.",
        "",
        "FOR EACH ITEM DETECT:",
        "1. ai_model: Which AI model/tool is discussed (gpt-4, claude, deepseek, gemini, copilot, cursor, llama, etc). Use 'unknown' if unclear.",
        "2. aspects: Which of these 6 aspects are discussed, with RATING (1-5) and SENTIMENT:",
        "   - coding_speed: Speed, latency, execution time. Rating: 1=very slow, 5=very fast",
        "   - coding_performance: Quality, correctness, accuracy. Rating: 1=poor quality, 5=excellent",
        "   - cost_price: Pricing, API costs. Rating: 1=expensive, 5=affordable/good value",
        "   - ethics: Bias, fairness, transparency. Rating: 1=major concerns, 5=ethical/responsible",
        "   - innovation: Novelty, breakthroughs. Rating: 1=nothing new, 5=highly innovative",
        "   - skepticism_hype: Doubt vs hype. Rating: 1=overhyped/skeptical, 5=justified/realistic",
        "",
        "OUTPUT FORMAT (JSON array):",
        "{ 'payload_id': str, 'ai_model': str, 'aspects': [{aspect, present, rating, sentiment, evidence, confidence}] }",
        "",
        "RULES: Use ONLY 6 aspect names. rating: 1-5 integer. sentiment: positive/negative/neutral/mixed. confidence: high/medium/low. JSON ONLY.",
    ]

    user_blocks = []
    for idx, item in enumerate(batch):
        user_blocks.append(
            f"[{idx+1}] id={item.get('payload_id','?')} | {item.get('story_title','')[:80]} | {item.get('comment_text','')[:200]}"
        )

    user_content = "\n".join(instructions) + "\n\n" + "\n".join(user_blocks)

    return [
        {"role": "user", "content": user_content}
    ]


def run_stage1(
    input_jsonl: Path,
    output_jsonl: Path,
    model: str,
    api_key: str,
    batch_size: int,
    timeout: int
):
    """Stage 1: Aspect Detection (batched)."""
    print(f"\n=== STAGE 1: Aspect Detection ===")
    print(f"Input: {input_jsonl}")
    print(f"Batch size: {batch_size}\n")

    stage1_success = 0
    stage1_error = 0

    with open(input_jsonl, "r", encoding="utf-8") as infile, \
         open(output_jsonl, "w", encoding="utf-8") as outfile:
        batch: List[Dict[str, Any]] = []
        idx = 0

        for line in infile:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            batch.append(payload)
            if len(batch) >= batch_size:
                # Process batch
                messages = build_stage1_batch_messages(batch)
                start_time = time.time()
                result = call_innkube(messages, model, api_key, timeout)
                elapsed = time.time() - start_time

                if result["error"]:
                    for item in batch:
                        idx += 1
                        stage1_error += 1
                        output_record = {
                            "payload_id": item["payload_id"],
                            "payload_type": item.get("payload_type"),
                            "story_id": item.get("story_id"),
                            "ai_model": "unknown",
                            "detected_aspects": [],
                            "stage1_elapsed": round(elapsed, 2),
                            "error": result["error"],
                        }
                        outfile.write(json.dumps(output_record) + "\n")
                        print(f"[S1 {idx}] {item['payload_id']}: ❌ ERROR")
                else:
                    try:
                        responses = result["response"]
                        if not isinstance(responses, list):
                            raise ValueError("Stage1 batch response is not a list")
                        if len(responses) != len(batch):
                            raise ValueError(f"Stage1 batch length mismatch: expected {len(batch)}, got {len(responses)}")

                        for item, resp in zip(batch, responses):
                            idx += 1
                            stage1_success += 1
                            if isinstance(resp, dict):
                                aspects_data = resp.get("aspects", [])
                                ai_model = normalize_model(resp.get("ai_model", "unknown"))
                            else:
                                aspects_data = []
                                ai_model = "unknown"
                            aspects = normalize_aspects(aspects_data)
                            output_record = {
                                "payload_id": item["payload_id"],
                                "payload_type": item.get("payload_type"),
                                "story_id": item.get("story_id"),
                                "comment_text": item.get("comment_text", ""),
                                "ai_model": ai_model,
                                "detected_aspects": aspects,
                                "stage1_elapsed": round(elapsed, 2),
                                "error": None,
                            }
                            outfile.write(json.dumps(output_record) + "\n")
                            print(f"[S1 {idx}] {item['payload_id']}: ✅ ({len(aspects)} aspects, {ai_model})")
                    except Exception as e:
                        for item in batch:
                            idx += 1
                            stage1_error += 1
                            output_record = {
                                "payload_id": item["payload_id"],
                                "payload_type": item.get("payload_type"),
                                "story_id": item.get("story_id"),
                                "ai_model": "unknown",
                                "detected_aspects": [],
                                "stage1_elapsed": round(elapsed, 2),
                                "error": f"Parse error: {e}",
                            }
                            outfile.write(json.dumps(output_record) + "\n")
                            print(f"[S1 {idx}] {item['payload_id']}: ❌ PARSE")

                outfile.flush()
                batch = []

        # Process remaining batch
        if batch:
            messages = build_stage1_batch_messages(batch)
            start_time = time.time()
            result = call_innkube(messages, model, api_key, timeout)
            elapsed = time.time() - start_time

            if result["error"]:
                for item in batch:
                    idx += 1
                    stage1_error += 1
                    output_record = {
                        "payload_id": item["payload_id"],
                        "payload_type": item.get("payload_type"),
                        "story_id": item.get("story_id"),
                        "ai_model": "unknown",
                        "detected_aspects": [],
                        "stage1_elapsed": round(elapsed, 2),
                        "error": result["error"],
                    }
                    outfile.write(json.dumps(output_record) + "\n")
                    print(f"[S1 {idx}] {item['payload_id']}: ❌ ERROR")
            else:
                try:
                    responses = result["response"]
                    if not isinstance(responses, list):
                        raise ValueError("Stage1 batch response is not a list")
                    if len(responses) != len(batch):
                        raise ValueError(f"Stage1 batch length mismatch: expected {len(batch)}, got {len(responses)}")

                    for item, resp in zip(batch, responses):
                        idx += 1
                        stage1_success += 1
                        if isinstance(resp, dict):
                            aspects_data = resp.get("aspects", [])
                            ai_model = normalize_model(resp.get("ai_model", "unknown"))
                        else:
                            aspects_data = []
                            ai_model = "unknown"
                        aspects = normalize_aspects(aspects_data)
                        output_record = {
                            "payload_id": item["payload_id"],
                            "payload_type": item.get("payload_type"),
                            "story_id": item.get("story_id"),
                            "comment_text": item.get("comment_text", ""),
                            "ai_model": ai_model,
                            "detected_aspects": aspects,
                            "stage1_elapsed": round(elapsed, 2),
                            "error": None,
                        }
                        outfile.write(json.dumps(output_record) + "\n")
                        print(f"[S1 {idx}] {item['payload_id']}: ✅ ({len(aspects)} aspects, {ai_model})")
                except Exception as e:
                    for item in batch:
                        idx += 1
                        stage1_error += 1
                        output_record = {
                            "payload_id": item["payload_id"],
                            "payload_type": item.get("payload_type"),
                            "story_id": item.get("story_id"),
                            "ai_model": "unknown",
                            "detected_aspects": [],
                            "stage1_elapsed": round(elapsed, 2),
                            "error": f"Parse error: {e}",
                        }
                        outfile.write(json.dumps(output_record) + "\n")
                        print(f"[S1 {idx}] {item['payload_id']}: ❌ PARSE")

    print(f"\nStage 1 Summary: {stage1_success} ✅  {stage1_error} ❌\n")
    return output_jsonl


def run_stage2_prep(
    stage1_output: Path,
    stage2_input: Path,
    min_confidence: float
):
    """Prep Stage 2: Filter by confidence, prepare for batch classification."""
    print(f"\n=== STAGE 2 PREP: Filter & Prepare ===")
    print(f"Min confidence threshold: {min_confidence}\n")

    qualified = 0
    filtered = 0

    with open(stage1_output, "r", encoding="utf-8") as infile, \
         open(stage2_input, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

            aspects = payload.get("detected_aspects", [])
            # Filter to aspects meeting confidence threshold (use parse_confidence for safety)
            qualified_aspects = [
                a for a in aspects
                if a.get("present", False) and parse_confidence(a.get("confidence", 0)) >= min_confidence
            ]

            if qualified_aspects:
                payload["detected_aspects"] = qualified_aspects
                outfile.write(json.dumps(payload) + "\n")
                qualified += 1
            else:
                filtered += 1

    print(f"Stage 2 Prep: {qualified} qualified  {filtered} filtered\n")
    return stage2_input


def merge_results(
    input_jsonl: Path,
    final_output: Path
):
    """Merge filtered results into final output."""
    print(f"\n=== MERGE: Final Output ===")
    print(f"Output: {final_output}\n")

    merged = 0

    with open(input_jsonl, "r", encoding="utf-8") as infile, \
         open(final_output, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                outfile.write(json.dumps(record) + "\n")
                merged += 1
            except json.JSONDecodeError:
                continue

    print(f"Merged {merged} records to final output\n")
    return final_output


def main():
    parser = argparse.ArgumentParser(description="V3 Optimized single-LLM sentiment analysis pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Input payloads JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Final output JSONL")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size (default: 10)")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE, help=f"Min confidence (default: {DEFAULT_MIN_CONFIDENCE})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--key-path", type=Path, default=Path("key"), help="Path to API key file")
    parser.add_argument("--max-payloads", type=int, default=None, help="Max payloads to process (for testing)")

    args = parser.parse_args()

    # Validate inputs
    if not args.input.exists():
        print(f"❌ Input file not found: {args.input}")
        return

    # Load API key
    try:
        api_key = load_api_key(args.key_path)
    except FileNotFoundError:
        print(f"❌ API key file not found: {args.key_path}")
        return

    print(f"\n{'='*60}")
    print(f"V3 OPTIMIZED SINGLE-LLM SENTIMENT ANALYSIS PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Model: {args.model}")
    if args.max_payloads:
        print(f"Max payloads: {args.max_payloads} (testing mode)")
    print(f"{'='*60}\n")

    # If max_payloads specified, create temporary input file
    if args.max_payloads:
        temp_input = args.output.parent / f"temp_input_{args.max_payloads}.jsonl"
        with open(args.input, "r", encoding="utf-8") as infile, \
             open(temp_input, "w", encoding="utf-8") as outfile:
            for i, line in enumerate(infile):
                if i >= args.max_payloads:
                    break
                outfile.write(line)
        input_file = temp_input
        print(f"Created temporary input with {args.max_payloads} payloads\n")
    else:
        input_file = args.input

    # Stage 1: Aspect Detection + Sentiment (combined - single LLM call)
    stage1_output = args.output.parent / f"{args.output.stem}_stage1.jsonl"
    run_stage1(input_file, stage1_output, args.model, api_key, args.batch_size, args.timeout)

    # Filter by confidence (no LLM call - just Python filtering)
    filtered_output = args.output.parent / f"{args.output.stem}_filtered.jsonl"
    run_stage2_prep(stage1_output, filtered_output, args.min_confidence)

    # Merge: Final output
    merge_results(filtered_output, args.output)

    # Clean up temporary file
    if args.max_payloads and temp_input.exists():
        temp_input.unlink()
        print(f"Cleaned up temporary input file\n")

    print(f"✅ PIPELINE COMPLETE: {args.output}\n")


if __name__ == "__main__":
    main()
