#!/usr/bin/env python3
"""Temporary helper to detect and add model names to final_sentiment_results_v2.jsonl.

Strategy:
  1. Scan comment_text and story_title from original payloads (sentiment_payloads_v2.jsonl)
  2. Extract all model mentions (gpt, claude, gemini, llama, mistral) from combined text
  3. Use FIRST mentioned model, or default to 'gpt' if none found

Usage:
  python add_model_tags_temp.py --input final_sentiment_results_v2.jsonl --payloads sentiment_payloads_v2.jsonl --output final_sentiment_results_v2_tagged.jsonl
"""
import argparse
import json
from pathlib import Path
from typing import List, Dict

# Model keywords to search for (in order of preference for ambiguous matches)
TOP_MODELS: List[str] = [
    "gpt",
    "claude",
    "gemini",
    "llama",
    "mistral",
]

def extract_model_from_text(text: str) -> str:
    """Extract the FIRST mentioned model from text, case-insensitive."""
    if not text:
        return None
    text_lower = text.lower()
    for model in TOP_MODELS:
        if model in text_lower:
            return model
    return None

def load_payload_map(payloads_file: Path) -> Dict[str, Dict]:
    """Load payload_id -> {comment_text, story_title} mapping."""
    payload_map = {}
    if not payloads_file.is_file():
        return payload_map
    with open(payloads_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                payload_map[p["payload_id"]] = {
                    "comment_text": p.get("comment_text", ""),
                    "story_title": p.get("story_title", ""),
                }
            except (json.JSONDecodeError, KeyError):
                continue
    return payload_map

def main():
    parser = argparse.ArgumentParser(description="Add/overwrite model names in final sentiment results")
    parser.add_argument("--input", type=Path, default=Path("final_sentiment_results_v2.jsonl"), help="Input JSONL file")
    parser.add_argument("--payloads", type=Path, default=Path("sentiment_payloads_v2.jsonl"), help="Original payloads JSONL to extract text")
    parser.add_argument("--output", type=Path, default=Path("final_sentiment_results_v2_tagged.jsonl"), help="Output JSONL file")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input file not found: {args.input}")

    # Load payloads to extract text
    print(f"Loading payloads from {args.payloads}...")
    payload_map = load_payload_map(args.payloads)
    print(f"  Loaded {len(payload_map)} payloads\n")

    written = 0
    models_found = {"gpt": 0, "claude": 0, "gemini": 0, "llama": 0, "mistral": 0, "default": 0}
    
    with open(args.input, "r", encoding="utf-8") as infile, open(args.output, "w", encoding="utf-8") as outfile:
        for line in infile:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            payload_id = record.get("payload_id")
            payload_info = payload_map.get(payload_id, {})
            combined_text = f"{payload_info.get('story_title', '')} {payload_info.get('comment_text', '')}"
            
            # Extract first mentioned model
            detected_model = extract_model_from_text(combined_text)
            if detected_model:
                record["model"] = detected_model
                models_found[detected_model] += 1
            else:
                # Default to gpt if no model mentioned
                record["model"] = "gpt"
                models_found["default"] += 1
            
            outfile.write(json.dumps(record) + "\n")
            written += 1

    print(f"Wrote {written} records to {args.output}")
    print(f"Model distribution:")
    for model, count in models_found.items():
        if count > 0:
            pct = 100 * count / written
            print(f"  {model}: {count:>5} ({pct:>5.1f}%)")


if __name__ == "__main__":
    main()
