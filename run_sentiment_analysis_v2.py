#!/usr/bin/env python3
"""
V2: UNIFIED sentiment analysis runner - TWO-STAGE COMPLETE PIPELINE

Works with v2 payloads from prepare_sentiment_payloads_v2.py

Single execution: Stage 1 → Stage 2 Prep → Stage 2 → Merge into final output
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


def normalize_aspects(aspects: Any) -> List[Dict[str, Any]]:
    """Normalize LLM Stage 1 output to consistent format."""
    normalized: List[Dict[str, Any]] = []
    if not aspects:
        return normalized

    if isinstance(aspects, dict):
        for k, v in aspects.items():
            try:
                present = bool(v)
            except Exception:
                present = True
            normalized.append({
                "aspect": str(k),
                "present": present,
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
                    normalized.append({
                        "aspect": str(k),
                        "present": bool(v),
                        "evidence": item.get("evidence", ""),
                        "confidence": float(item.get("confidence", 0.7)),
                    })
                    continue
                normalized.append({
                    "aspect": str(aspect_name) if aspect_name is not None else "unknown",
                    "present": bool(item.get("present", True)),
                    "evidence": item.get("evidence", ""),
                    "confidence": float(item.get("confidence", 0.7)),
                })
            elif isinstance(item, str):
                normalized.append({
                    "aspect": item,
                    "present": True,
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
    """Construct single Stage 1 request for a batch."""
    batch_size = len(batch)
    instructions = [
        "You are an aspect detection system for AI-related Hacker News discussions.",
        f"Process ALL {batch_size} items in this batch and return a JSON ARRAY of length EXACTLY {batch_size}.",
        "Each element: { 'payload_id': str, 'aspects': [ {aspect, present, evidence, confidence} ] }",
        "If no aspects, use empty array. JSON ONLY. No markdown, prose, or explanations.",
        f"Do not invent or drop payloads. Array length MUST equal {batch_size}.",
    ]
    
    user_blocks = []
    for idx, item in enumerate(batch):
        user_blocks.append(
            f"Item {idx+1} | payload_id={item.get('payload_id','unknown')}\n"
            f"Story: {item.get('story_title','')}\n"
            f"Comment: {item.get('comment_text','')}\n"
        )
    
    user_content = "\n".join(instructions) + "\n\n" + "\n---\n".join(user_blocks)
    
    return [
        {"role": "user", "content": user_content}
    ]


def build_stage2_batch_messages(batch: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Construct single Stage 2 request for a batch."""
    batch_size = len(batch)
    instructions = [
        "You are an expert AI sentiment analyst.",
        f"For each of these {batch_size} items with detected aspects, classify sentiment ONLY for those aspects.",
        f"Return JSON ARRAY of length EXACTLY {batch_size} with: {{'payload_id': str, 'aspect_sentiments': [...] }}",
        "Each aspect_sentiment: {'aspect': str, 'sentiment': 'positive'|'negative'|'neutral'|'mixed', 'score': float 0-1}",
        f"JSON ONLY. No markdown, prose, or explanations. Array length MUST equal {batch_size}.",
    ]
    
    user_blocks = []
    for idx, item in enumerate(batch):
        aspects_str = ", ".join([a['aspect'] for a in item.get('detected_aspects', [])])
        user_blocks.append(
            f"Item {idx+1} | payload_id={item.get('payload_id','unknown')}\n"
            f"Aspects to classify: {aspects_str}\n"
            f"Comment: {item.get('comment_text','')}\n"
        )
    
    user_content = "\n".join(instructions) + "\n\n" + "\n---\n".join(user_blocks)
    
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
                            else:
                                aspects_data = []
                            aspects = normalize_aspects(aspects_data)
                            output_record = {
                                "payload_id": item["payload_id"],
                                "payload_type": item.get("payload_type"),
                                "story_id": item.get("story_id"),
                                "comment_text": item.get("comment_text", ""),
                                "detected_aspects": aspects,
                                "stage1_elapsed": round(elapsed, 2),
                                "error": None,
                            }
                            outfile.write(json.dumps(output_record) + "\n")
                            print(f"[S1 {idx}] {item['payload_id']}: ✅ ({len(aspects)} aspects)")
                    except Exception as e:
                        for item in batch:
                            idx += 1
                            stage1_error += 1
                            output_record = {
                                "payload_id": item["payload_id"],
                                "payload_type": item.get("payload_type"),
                                "story_id": item.get("story_id"),
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
                        else:
                            aspects_data = []
                        aspects = normalize_aspects(aspects_data)
                        output_record = {
                            "payload_id": item["payload_id"],
                            "payload_type": item.get("payload_type"),
                            "story_id": item.get("story_id"),
                            "comment_text": item.get("comment_text", ""),
                            "detected_aspects": aspects,
                            "stage1_elapsed": round(elapsed, 2),
                            "error": None,
                        }
                        outfile.write(json.dumps(output_record) + "\n")
                        print(f"[S1 {idx}] {item['payload_id']}: ✅ ({len(aspects)} aspects)")
                except Exception as e:
                    for item in batch:
                        idx += 1
                        stage1_error += 1
                        output_record = {
                            "payload_id": item["payload_id"],
                            "payload_type": item.get("payload_type"),
                            "story_id": item.get("story_id"),
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
            # Filter to aspects meeting confidence threshold
            qualified_aspects = [
                a for a in aspects
                if a.get("present", False) and a.get("confidence", 0) >= min_confidence
            ]
            
            if qualified_aspects:
                payload["detected_aspects"] = qualified_aspects
                outfile.write(json.dumps(payload) + "\n")
                qualified += 1
            else:
                filtered += 1
    
    print(f"Stage 2 Prep: {qualified} qualified  {filtered} filtered\n")
    return stage2_input


def run_stage2(
    stage2_input: Path,
    stage2_output: Path,
    model: str,
    api_key: str,
    batch_size: int,
    timeout: int
):
    """Stage 2: Aspect Sentiment Classification (batched)."""
    print(f"\n=== STAGE 2: Sentiment Classification ===")
    print(f"Input: {stage2_input}")
    print(f"Batch size: {batch_size}\n")
    
    stage2_success = 0
    stage2_error = 0
    
    with open(stage2_input, "r", encoding="utf-8") as infile, \
         open(stage2_output, "w", encoding="utf-8") as outfile:
        batch2: List[Dict[str, Any]] = []
        idx = 0
        
        for line in infile:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            batch2.append(payload)
            if len(batch2) >= batch_size:
                messages = build_stage2_batch_messages(batch2)
                start_time = time.time()
                result = call_innkube(messages, model, api_key, timeout)
                elapsed = time.time() - start_time
                
                if result["error"]:
                    for item in batch2:
                        idx += 1
                        stage2_error += 1
                        output_record = {
                            "payload_id": item["payload_id"],
                            "payload_type": item.get("payload_type"),
                            "story_id": item.get("story_id"),
                            "detected_aspects": item.get("detected_aspects", []),
                            "stage2_result": None,
                            "stage1_elapsed": item.get("stage1_elapsed", 0),
                            "stage2_elapsed": round(elapsed, 2),
                            "error": result["error"],
                        }
                        outfile.write(json.dumps(output_record) + "\n")
                        print(f"[S2 {idx}] {item['payload_id']}: ❌ ERROR")
                else:
                    try:
                        responses = result["response"]
                        if not isinstance(responses, list):
                            raise ValueError("Stage2 batch response is not a list")
                        if len(responses) != len(batch2):
                            raise ValueError(f"Stage2 batch length mismatch: expected {len(batch2)}, got {len(responses)}")
                        
                        for item, resp in zip(batch2, responses):
                            idx += 1
                            stage2_success += 1
                            output_record = {
                                "payload_id": item["payload_id"],
                                "payload_type": item.get("payload_type"),
                                "story_id": item.get("story_id"),
                                "detected_aspects": item.get("detected_aspects", []),
                                "stage2_result": resp,
                                "stage1_elapsed": item.get("stage1_elapsed", 0),
                                "stage2_elapsed": round(elapsed, 2),
                                "error": None,
                            }
                            outfile.write(json.dumps(output_record) + "\n")
                            print(f"[S2 {idx}] {item['payload_id']}: ✅")
                    except Exception as e:
                        for item in batch2:
                            idx += 1
                            stage2_error += 1
                            output_record = {
                                "payload_id": item["payload_id"],
                                "payload_type": item.get("payload_type"),
                                "story_id": item.get("story_id"),
                                "detected_aspects": item.get("detected_aspects", []),
                                "stage2_result": None,
                                "stage1_elapsed": item.get("stage1_elapsed", 0),
                                "stage2_elapsed": round(elapsed, 2),
                                "error": f"Batch parse error: {e}",
                            }
                            outfile.write(json.dumps(output_record) + "\n")
                            print(f"[S2 {idx}] {item['payload_id']}: ❌ PARSE")
                
                outfile.flush()
                batch2 = []
        
        # Process remaining batch
        if batch2:
            messages = build_stage2_batch_messages(batch2)
            start_time = time.time()
            result = call_innkube(messages, model, api_key, timeout)
            elapsed = time.time() - start_time
            
            if result["error"]:
                for item in batch2:
                    idx += 1
                    stage2_error += 1
                    output_record = {
                        "payload_id": item["payload_id"],
                        "payload_type": item.get("payload_type"),
                        "story_id": item.get("story_id"),
                        "detected_aspects": item.get("detected_aspects", []),
                        "stage2_result": None,
                        "stage1_elapsed": item.get("stage1_elapsed", 0),
                        "stage2_elapsed": round(elapsed, 2),
                        "error": result["error"],
                    }
                    outfile.write(json.dumps(output_record) + "\n")
                    print(f"[S2 {idx}] {item['payload_id']}: ❌ ERROR")
            else:
                try:
                    responses = result["response"]
                    if not isinstance(responses, list):
                        raise ValueError("Stage2 batch response is not a list")
                    if len(responses) != len(batch2):
                        raise ValueError(f"Stage2 batch length mismatch: expected {len(batch2)}, got {len(responses)}")
                    
                    for item, resp in zip(batch2, responses):
                        idx += 1
                        stage2_success += 1
                        output_record = {
                            "payload_id": item["payload_id"],
                            "payload_type": item.get("payload_type"),
                            "story_id": item.get("story_id"),
                            "detected_aspects": item.get("detected_aspects", []),
                            "stage2_result": resp,
                            "stage1_elapsed": item.get("stage1_elapsed", 0),
                            "stage2_elapsed": round(elapsed, 2),
                            "error": None,
                        }
                        outfile.write(json.dumps(output_record) + "\n")
                        print(f"[S2 {idx}] {item['payload_id']}: ✅")
                except Exception as e:
                    for item in batch2:
                        idx += 1
                        stage2_error += 1
                        output_record = {
                            "payload_id": item["payload_id"],
                            "payload_type": item.get("payload_type"),
                            "story_id": item.get("story_id"),
                            "detected_aspects": item.get("detected_aspects", []),
                            "stage2_result": None,
                            "stage1_elapsed": item.get("stage1_elapsed", 0),
                            "stage2_elapsed": round(elapsed, 2),
                            "error": f"Batch parse error: {e}",
                        }
                        outfile.write(json.dumps(output_record) + "\n")
                        print(f"[S2 {idx}] {item['payload_id']}: ❌ PARSE")
    
    print(f"\nStage 2 Summary: {stage2_success} ✅  {stage2_error} ❌\n")
    return stage2_output


def merge_results(
    stage2_output: Path,
    final_output: Path
):
    """Merge Stage 2 results into final output."""
    print(f"\n=== MERGE: Final Output ===")
    print(f"Output: {final_output}\n")
    
    merged = 0
    
    with open(stage2_output, "r", encoding="utf-8") as infile, \
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
    parser = argparse.ArgumentParser(description="V2 Batched sentiment analysis pipeline")
    parser.add_argument("--input", type=Path, required=True, help="Input payloads JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Final output JSONL")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size (default: 10)")
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE, help=f"Min confidence (default: {DEFAULT_MIN_CONFIDENCE})")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"LLM model (default: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--key-path", type=Path, default=Path("key"), help="Path to API key file")
    
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
    print(f"V2 BATCHED SENTIMENT ANALYSIS PIPELINE")
    print(f"{'='*60}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Batch size: {args.batch_size}")
    print(f"Min confidence: {args.min_confidence}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")
    
    # Stage 1: Aspect Detection
    stage1_output = args.output.parent / f"{args.output.stem}_stage1.jsonl"
    run_stage1(args.input, stage1_output, args.model, api_key, args.batch_size, args.timeout)
    
    # Stage 2 Prep: Filter by confidence
    stage2_input = args.output.parent / f"{args.output.stem}_stage2_input.jsonl"
    run_stage2_prep(stage1_output, stage2_input, args.min_confidence)
    
    # Stage 2: Sentiment Classification
    stage2_output = args.output.parent / f"{args.output.stem}_stage2_output.jsonl"
    run_stage2(stage2_input, stage2_output, args.model, api_key, args.batch_size, args.timeout)
    
    # Merge: Final output
    merge_results(stage2_output, args.output)
    
    print(f"✅ PIPELINE COMPLETE: {args.output}\n")


if __name__ == "__main__":
    main()
