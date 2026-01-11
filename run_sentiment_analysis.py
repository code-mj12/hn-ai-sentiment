#!/usr/bin/env python3
"""
UNIFIED sentiment analysis runner - TWO-STAGE COMPLETE PIPELINE

Consolidated from: run_stage2_analysis.py, prepare_stage2_payloads.py, merge_results.py

PIPELINE:
  Stage 1: Aspect Detection (input: sentiment_payloads.jsonl)
  ↓
  Stage 2 Prep: Filter by confidence, generate stage2_messages
  ↓
  Stage 2: Aspect Sentiment Classification
  ↓
  Merge: Combine results into final output

Single Command Execution:
  python run_sentiment_analysis.py --input sentiment_payloads.jsonl --output final_sentiment_results.jsonl

Input Format:
  sentiment_payloads.jsonl with stage1_messages field

Output Format:
  {
    "payload_id": "...",
    "payload_type": "...",
    "story_id": "...",
    "detected_aspects": [...],
    "aspect_sentiments": [{"aspect": "...", "sentiment": "...", "score": ...}],
    "overall_sentiment": "positive|neutral|negative|mixed",
    "overall_score": 0.75,
    "stage1_elapsed": 5.2,
    "stage2_elapsed": 3.1,
    "total_elapsed": 8.3,
    "error": null
  }
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
DEFAULT_TIMEOUT = 120  # seconds
DEFAULT_MIN_CONFIDENCE = 0.7


# STAGE 2 System Prompt (for aspect sentiment classification)
STAGE2_SYSTEM_PROMPT = """You are an expert AI sentiment analyst. Your task is to classify sentiment for SPECIFIC AI-RELATED ASPECTS in technical comments.

For each aspect provided, classify the sentiment ONLY for that aspect.

IMPORTANT: 
- Only classify sentiment if the comment contains information relevant to the aspect
- Rate confidence as 1.0 (certain), 0.8 (likely), 0.6 (moderate), 0.4 (uncertain)
- Sentiment scale: -1.0 (strongly negative), -0.5 (negative), 0.0 (neutral), +0.5 (positive), +1.0 (strongly positive)
- If comment doesn't address the aspect, use score 0.0 and sentiment "neutral"

Respond in JSON format:
{
  "aspect_sentiments": [
    {
      "aspect": "aspect_name",
      "sentiment": "positive|neutral|negative|mixed",
      "score": -1.0 to 1.0,
      "confidence": 0.0 to 1.0,
      "reasoning": "brief explanation"
    }
  ]
}"""


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
        "response_format": {"type": "json_object"},
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


def load_processed_payload_ids(output_path: Path) -> set:
    """Load already processed payload IDs from output JSONL."""
    processed = set()
    if not output_path.exists():
        return processed
    
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    result = json.loads(line)
                    processed.add(result["payload_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    
    return processed


def create_stage2_messages(
    comment_text: str,
    detected_aspects: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Create Stage 2 messages for aspect sentiment classification."""
    aspects_text = "\n".join([
        f"- {a['aspect']}: {a.get('evidence', 'No specific evidence')}"
        for a in detected_aspects
    ])
    
    user_message = f"""Comment: {comment_text}

Detected aspects (from Stage 1):
{aspects_text}

Classify sentiment for each aspect above."""
    
    return [
        {"role": "system", "content": STAGE2_SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]


def run_unified_pipeline(
    input_jsonl: Path,
    output_jsonl: Path,
    model: str,
    api_key: str,
    timeout: int,
    max_payloads: int = 0,
    sleep_seconds: float = 0.0,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    resume: bool = True
):
    """
    Run complete TWO-STAGE pipeline:
    Stage 1 → Stage 2 Prep → Stage 2 → Merge
    """
    
    # Stage 1 temporary files
    stage1_output = Path("stage1_temp.jsonl")
    stage2_input = Path("stage2_payloads_temp.jsonl")
    stage2_output = Path("stage2_temp.jsonl")
    
    print(f"\n{'='*70}")
    print(f"UNIFIED TWO-STAGE PIPELINE")
    print(f"{'='*70}")
    print(f"Input: {input_jsonl}")
    print(f"Output: {output_jsonl}")
    print(f"Model: {model}")
    print(f"Min Confidence (Stage 2 filter): {min_confidence}")
    print(f"{'='*70}\n")
    
    # =====================================================================
    # STAGE 1: ASPECT DETECTION
    # =====================================================================
    print("STAGE 1: Running aspect detection...\n")
    
    processed_ids = set()
    if resume and stage1_output.exists():
        processed_ids = load_processed_payload_ids(stage1_output)
        if processed_ids:
            print(f"Resume: Found {len(processed_ids):,} Stage 1 results\n")
    
    stage1_mode = "a" if resume else "w"
    stage1_count = 0
    stage1_success = 0
    stage1_error = 0
    
    with open(input_jsonl, "r", encoding="utf-8") as infile, \
         open(stage1_output, stage1_mode, encoding="utf-8") as outfile:
        
        for line in infile:
            if not line.strip():
                continue
            
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            payload_id = payload.get("payload_id", f"unknown_{stage1_count}")
            
            if payload_id in processed_ids:
                continue
            
            if max_payloads > 0 and stage1_count >= max_payloads:
                break
            
            story_id = payload.get("story_id", "unknown")
            payload_type = payload.get("payload_type", "unknown")
            messages = payload.get("stage1_messages", [])
            
            if not messages:
                continue
            
            start_time = time.time()
            result = call_innkube(messages, model, api_key, timeout)
            elapsed = time.time() - start_time
            
            output_record = {
                "payload_id": payload_id,
                "payload_type": payload_type,
                "story_id": story_id,
                "stage1_result": result["response"],
                "stage1_elapsed": round(elapsed, 2),
                "error": result["error"]
            }
            
            outfile.write(json.dumps(output_record) + "\n")
            outfile.flush()
            
            stage1_count += 1
            
            if result["error"]:
                stage1_error += 1
                print(f"[Stage1 {stage1_count}] {payload_id}: ❌ ERROR")
            else:
                stage1_success += 1
                aspects = result["response"].get("aspects", []) if result["response"] else []
                print(f"[Stage1 {stage1_count}] {payload_id}: ✅ {len(aspects)} aspects ({elapsed:.1f}s)")
            
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    
    print(f"\nStage 1 complete: {stage1_success} success, {stage1_error} errors\n")
    
    # =====================================================================
    # STAGE 2 PREP: FILTER & GENERATE STAGE 2 PAYLOADS
    # =====================================================================
    print("STAGE 2 PREP: Filtering by confidence & generating payloads...\n")
    
    stage2_count = 0
    stage2_qualified = 0
    
    with open(stage1_output, "r", encoding="utf-8") as infile, \
         open(stage2_input, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            if not line.strip():
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            stage1_result = record.get("stage1_result", {})
            if not stage1_result:
                continue
            
            aspects = stage1_result.get("aspects", [])
            if not aspects:
                stage2_count += 1
                print(f"[S2Prep {stage2_count}] {record['payload_id']}: No aspects detected, skip Stage 2")
                continue
            
            # Filter by confidence
            qualified_aspects = [
                a for a in aspects 
                if a.get("confidence", 0) >= min_confidence
            ]
            
            if not qualified_aspects:
                stage2_count += 1
                print(f"[S2Prep {stage2_count}] {record['payload_id']}: No aspects above confidence {min_confidence}, skip Stage 2")
                continue
            
            # Generate Stage 2 messages
            stage2_messages = create_stage2_messages(
                record.get("comment_text", ""),
                qualified_aspects
            )
            
            stage2_payload = {
                "payload_id": record["payload_id"],
                "payload_type": record.get("payload_type"),
                "story_id": record.get("story_id"),
                "comment_text": record.get("comment_text", ""),
                "detected_aspects": qualified_aspects,
                "stage2_messages": stage2_messages,
                "stage1_elapsed": record.get("stage1_elapsed", 0)
            }
            
            outfile.write(json.dumps(stage2_payload) + "\n")
            
            stage2_count += 1
            stage2_qualified += 1
            aspect_names = [a["aspect"] for a in qualified_aspects[:2]]
            print(f"[S2Prep {stage2_count}] {record['payload_id']}: ✅ {len(qualified_aspects)} aspects → Stage 2 ({', '.join(aspect_names)}...)")
    
    reduction_pct = 100 * (1 - stage2_qualified / max(1, stage2_count))
    print(f"\nStage 2 Prep complete: {stage2_qualified}/{stage2_count} qualified ({reduction_pct:.0f}% filtered)\n")
    
    # =====================================================================
    # STAGE 2: ASPECT SENTIMENT CLASSIFICATION
    # =====================================================================
    print("STAGE 2: Running aspect sentiment classification...\n")
    
    stage2_success = 0
    stage2_error = 0
    
    with open(stage2_input, "r", encoding="utf-8") as infile, \
         open(stage2_output, "w", encoding="utf-8") as outfile:
        
        for idx, line in enumerate(infile, 1):
            if not line.strip():
                continue
            
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            payload_id = payload["payload_id"]
            stage2_messages = payload.get("stage2_messages", [])
            detected_aspects = payload.get("detected_aspects", [])
            
            start_time = time.time()
            result = call_innkube(stage2_messages, model, api_key, timeout)
            stage2_elapsed = time.time() - start_time
            
            output_record = {
                "payload_id": payload_id,
                "payload_type": payload.get("payload_type"),
                "story_id": payload.get("story_id"),
                "detected_aspects": detected_aspects,
                "stage2_result": result["response"],
                "stage1_elapsed": payload.get("stage1_elapsed", 0),
                "stage2_elapsed": round(stage2_elapsed, 2),
                "error": result["error"]
            }
            
            outfile.write(json.dumps(output_record) + "\n")
            outfile.flush()
            
            if result["error"]:
                stage2_error += 1
                print(f"[Stage2 {idx}] {payload_id}: ❌ ERROR")
            else:
                stage2_success += 1
                print(f"[Stage2 {idx}] {payload_id}: ✅ ({stage2_elapsed:.1f}s)")
            
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    
    print(f"\nStage 2 complete: {stage2_success} success, {stage2_error} errors\n")
    
    # =====================================================================
    # MERGE: COMBINE STAGE 1 + STAGE 2 RESULTS
    # =====================================================================
    print("MERGE: Combining Stage 1 + Stage 2 results...\n")
    
    # Load Stage 2 results
    stage2_data = {}
    with open(stage2_output, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    stage2_data[record["payload_id"]] = record
                except (json.JSONDecodeError, KeyError):
                    continue
    
    merge_count = 0
    
    # Merge Stage 1 + Stage 2
    with open(stage1_output, "r", encoding="utf-8") as infile, \
         open(output_jsonl, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            if not line.strip():
                continue
            
            try:
                stage1_record = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            payload_id = stage1_record["payload_id"]
            stage1_result = stage1_record.get("stage1_result")
            
            # Handle None or missing stage1_result
            if stage1_result is None:
                detected_aspects = []
            else:
                detected_aspects = stage1_result.get("aspects", [])
            
            # Check if this payload went to Stage 2
            if payload_id in stage2_data:
                stage2_record = stage2_data[payload_id]
                stage2_result = stage2_record.get("stage2_result", {})
                aspect_sentiments = stage2_result.get("aspect_sentiments", [])
                
                # Calculate overall sentiment from aspect sentiments
                if aspect_sentiments:
                    scores = [a.get("score", 0) for a in aspect_sentiments]
                    avg_score = sum(scores) / len(scores)
                    if avg_score > 0.3:
                        overall_sentiment = "positive"
                    elif avg_score < -0.3:
                        overall_sentiment = "negative"
                    else:
                        overall_sentiment = "neutral"
                else:
                    aspect_sentiments = []
                    overall_sentiment = "neutral"
                    avg_score = 0.0
                
                total_elapsed = stage1_record.get("stage1_elapsed", 0) + stage2_record.get("stage2_elapsed", 0)
            else:
                # No Stage 2 (below confidence threshold)
                aspect_sentiments = []
                overall_sentiment = "neutral"
                avg_score = 0.0
                total_elapsed = stage1_record.get("stage1_elapsed", 0)
            
            final_record = {
                "payload_id": payload_id,
                "payload_type": stage1_record.get("payload_type"),
                "story_id": stage1_record.get("story_id"),
                "detected_aspects": detected_aspects,
                "aspect_sentiments": aspect_sentiments,
                "overall_sentiment": overall_sentiment,
                "overall_score": round(avg_score, 2),
                "stage1_elapsed": stage1_record.get("stage1_elapsed", 0),
                "stage2_elapsed": stage2_data.get(payload_id, {}).get("stage2_elapsed", 0),
                "total_elapsed": round(total_elapsed, 2),
                "error": stage1_record.get("error")
            }
            
            outfile.write(json.dumps(final_record) + "\n")
            merge_count += 1
    
    print(f"Merge complete: {merge_count} records written to {output_jsonl}\n")
    
    # Cleanup temporary files
    for temp_file in [stage1_output, stage2_input, stage2_output]:
        if temp_file.exists():
            temp_file.unlink()
    
    print(f"{'='*70}")
    print(f"PIPELINE COMPLETE")
    print(f"  Stage 1: {stage1_success} success, {stage1_error} errors")
    print(f"  Stage 2: {stage2_qualified} qualified, {stage2_success} success, {stage2_error} errors")
    print(f"  Final Output: {output_jsonl}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified two-stage sentiment analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("sentiment_payloads.jsonl"),
        help="Input JSONL with stage1_messages (default: sentiment_payloads.jsonl)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("final_sentiment_results.jsonl"),
        help="Output JSONL for final results (default: final_sentiment_results.jsonl)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=ALLOWED_MODELS,
        help=f"Model to use (default: {DEFAULT_MODEL})"
    )
    
    parser.add_argument(
        "--api-key-path",
        type=Path,
        default=Path("key"),
        help="Path to API key file (default: key)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    
    parser.add_argument(
        "--max-payloads",
        type=int,
        default=0,
        help="Maximum payloads to process (0 = unlimited, default: 0)"
    )
    
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep seconds between requests (default: 0.0)"
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Minimum confidence for Stage 2 filtering (default: {DEFAULT_MIN_CONFIDENCE})"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume mode"
    )
    
    args = parser.parse_args()
    
    if not args.input.is_file():
        print(f"ERROR: Input file does not exist: {args.input}")
        return
    
    if not args.api_key_path.is_file():
        print(f"ERROR: API key file does not exist: {args.api_key_path}")
        return
    
    api_key = load_api_key(args.api_key_path)
    
    run_unified_pipeline(
        input_jsonl=args.input,
        output_jsonl=args.output,
        model=args.model,
        api_key=api_key,
        timeout=args.timeout,
        max_payloads=args.max_payloads,
        sleep_seconds=args.sleep,
        min_confidence=args.min_confidence,
        resume=not args.no_resume
    )
    
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
