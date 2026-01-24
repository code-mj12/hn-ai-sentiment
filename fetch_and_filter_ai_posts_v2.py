#!/usr/bin/env python3
"""
V2 pipeline: read stories.csv + comments.csv + subcomments.csv, filter for AI content
using a strict multi-layer scoring system, and generate Stage 1 sentiment payloads.

Outputs:
- Filtered CSV: hn_ai_filtered_v2.csv (AI-related comments/subcomments with story metadata)
- Payloads JSONL: sentiment_payloads_v2.jsonl (one payload per filtered row)

Usage (defaults assume data/ folder):
  python fetch_and_filter_ai_posts_v2.py
  python fetch_and_filter_ai_posts_v2.py --threshold 6 --max-records 10000
"""

import argparse
import csv
import html
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# =============================================================================
# SCORING VOCABULARY (same spirit as v1)
# =============================================================================
CORE_CONCEPTS = [
    "transformer", "attention mechanism", "fine-tuning", "fine-tune",
    "embeddings", "token", "tokenization", "neural network",
    "backpropagation", "gradient descent", "loss function",
    "training data", "inference", "model weights", "parameters",
    "hyperparameters", "overfitting", "underfitting", "regularization",
    "dropout", "batch normalization", "activation function",
    "convolutional", "recurrent", "lstm", "gru", "bert", "gpt",
    "llm", "large language model", "foundation model",
    "zero-shot", "few-shot", "prompt engineering", "in-context learning",
    "rag", "retrieval augmented", "vector database", "semantic search",
    "reinforcement learning", "rlhf", "ppo", "dpo",
    "supervised learning", "unsupervised learning", "self-supervised",
    "contrastive learning", "diffusion model", "gan", "vae",
    "autoencoder", "quantization", "pruning", "distillation",
    "moe", "mixture of experts", "multi-modal", "cross-attention"
]

AI_COMPANIES = [
    "openai", "anthropic", "deepmind", "google ai", "meta ai",
    "hugging face", "huggingface", "cohere", "stability ai",
    "midjourney", "runway", "character.ai", "inflection",
    "adept", "ai21", "aleph alpha", "deepseek", "mistral",
    "together ai", "replicate", "modal labs"
]

TECHNICAL_JARGON = [
    "mmlu", "hellaswag", "humaneval", "gsm8k", "truthfulqa",
    "kv cache", "flash attention", "rope", "alibi", "sliding window",
    "lora", "qlora", "peft", "adapter", "prefix tuning",
    "chain of thought", "cot", "react", "tool use", "function calling",
    "constitutional ai", "red teaming", "jailbreak", "prompt injection",
    "temperature", "top-p", "top-k", "nucleus sampling",
    "perplexity", "bleu score", "rouge", "bertscore",
    "wandb", "mlflow", "tensorboard", "vllm", "tgi",
    "safetensors", "gguf", "ggml", "llama.cpp", "exllama"
]

LIBRARIES = [
    "pytorch", "tensorflow", "keras", "jax", "flax",
    "transformers", "diffusers", "accelerate", "bitsandbytes",
    "langchain", "llamaindex", "haystack", "semantic kernel",
    "autogen", "crewai", "guidance", "outlines", "instructor",
    "scikit-learn", "sklearn", "xgboost", "lightgbm",
    "opencv", "pillow", "albumentations", "timm"
]

MODEL_NAMES = [
    "gpt-3", "gpt-4", "gpt-3.5", "chatgpt", "claude", "claude-3",
    "llama", "llama-2", "llama-3", "mistral", "mixtral",
    "gemini", "palm", "bard", "falcon", "mpt", "stablelm",
    "vicuna", "alpaca", "dolly", "koala", "orca", "wizard",
    "codellama", "starcoder", "phind", "deepseek-coder",
    "stable diffusion", "sdxl", "dall-e", "midjourney",
    "whisper", "wav2vec", "clip", "blip", "sam"
]

TECHNICAL_PHRASES = [
    (r"(how to|guide to|tutorial on).*(fine-tune|train|implement|optimize|deploy)", 3),
    (r".*(vs\.?|versus|compared to|better than).*(gpt|claude|llama|mistral|gemini)", 3),
    (r"(implement|build|create).*(chatbot|agent|rag|pipeline)", 3),
    (r"(latency|throughput|tokens per second|inference speed|batch size)", 2),
    (r"(cost|price|pricing|api cost|\$.*per.*token)", 2),
    (r"(error|issue|problem|bug).*(training|inference|model|fine-tun)", 3),
    (r"(paper|research|arxiv|according to.*study)", 3),
    (r"(benchmark|score|evaluation|performance on).*(mmlu|hellaswag|humaneval)", 3)
]

CONTEXT_RULES: Dict[str, List[str]] = {
    "model": ["transformer", "train", "fine-tune", "inference", "weights", "architecture", "release", "open source"],
    "ai": ["safety", "alignment", "research", "lab", "ethics", "training", "model", "system"],
    "ml": ["pipeline", "training", "model", "inference", "production", "deployment"],
    "data": ["training", "dataset", "annotation", "quality", "synthetic", "augmentation"],
    "learning": ["machine", "deep", "reinforcement", "supervised", "unsupervised", "transfer"],
}

STRUCTURAL_PATTERNS = [
    (r"```[\s\S]*?```", 5),
    (r"`[^`]+`", 2),
    (r"arxiv\.org/abs/", 3),
    (r"github\.com/[\w\-]+/[\w\-]+", 3),
    (r"huggingface\.co/(models|datasets|spaces)", 3),
    (r"pip install|conda install|npm install", 4),
    (r"(import|from)\s+(torch|tensorflow|transformers|langchain)", 3),
    (r"(config|hyperparameters?):\s*\{", 2),
]

STAGE1_SYSTEM_PROMPT = """You are an aspect detection system for AI-related Hacker News discussions.\n\nIdentify which aspects are mentioned. For each aspect: present=true/false, evidence span, confidence 0.0-1.0.\nIf none apply, return an empty aspects array. Be strict. Respond with JSON only."""


# =============================================================================
# HELPERS
# =============================================================================

def clean_text(value: str) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_ai_score(text: str) -> Tuple[int, str]:
    if not text:
        return 0, "no_text"

    text_lower = text.lower()
    score = 0
    matches: List[str] = []

    for term in CORE_CONCEPTS:
        if term in text_lower:
            score += 2
            matches.append(f"core:{term}")

    for company in AI_COMPANIES:
        if company in text_lower:
            score += 2
            matches.append(f"company:{company}")

    for jargon in TECHNICAL_JARGON:
        if jargon in text_lower:
            score += 3
            matches.append(f"jargon:{jargon}")

    for lib in LIBRARIES:
        if lib in text_lower:
            score += 2
            matches.append(f"lib:{lib}")

    for model in MODEL_NAMES:
        if model in text_lower:
            score += 2
            matches.append(f"model:{model}")

    for pattern, points in TECHNICAL_PHRASES:
        if re.search(pattern, text_lower, re.IGNORECASE):
            score += points
            matches.append(f"phrase:+{points}")

    words = text_lower.split()
    for weak_term, context_words in CONTEXT_RULES.items():
        positions = [i for i, w in enumerate(words) if weak_term == w or weak_term in w]
        for pos in positions:
            start = max(0, pos - 10)
            end = min(len(words), pos + 10)
            window_text = " ".join(words[start:end])
            if any(ctx in window_text for ctx in context_words):
                score += 2
                matches.append(f"context:{weak_term}")
                break

    for pattern, points in STRUCTURAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            score += points
            matches.append(f"struct:+{points}")

    debug = ",".join(matches[:5]) if matches else "no_match"
    return score, debug


def is_ai_related(text: str, threshold: int) -> Tuple[bool, int, str]:
    score, info = calculate_ai_score(text)
    return score >= threshold, score, info


def load_stories(path: Path) -> Dict[str, Dict[str, str]]:
    stories: Dict[str, Dict[str, str]] = {}
    with open(path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            story_id = row["story_id"]
            stories[story_id] = {
                "title": clean_text(row.get("title", "")),
                "author": row.get("author", ""),
                "url": row.get("url", ""),
                "score": row.get("score", ""),
                "time": row.get("time", ""),
            }
    return stories


def load_comments(path: Path) -> Tuple[List[dict], Dict[str, dict]]:
    comments: List[dict] = []
    index: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            row["comment_text"] = clean_text(row.get("comment_text", ""))
            comments.append(row)
            index[row["comment_id"]] = row
    return comments, index


def load_subcomments(path: Path) -> List[dict]:
    subcomments: List[dict] = []
    with open(path, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            row["subcomment_text"] = clean_text(row.get("subcomment_text", ""))
            subcomments.append(row)
    return subcomments


def build_stage1_messages(story_title: str, comment_text: str, parent_text: str = "") -> List[dict]:
    if parent_text:
        user_content = (
            f"Story: {story_title}\n\n"
            f"Parent comment: {parent_text}\n\n"
            f"Reply: {comment_text}\n\n"
            "Detect which aspects are mentioned in this reply about the AI story above."
        )
        payload_type = "nested_comment"
    else:
        user_content = (
            f"Story: {story_title}\n\n"
            f"Comment: {comment_text}\n\n"
            "Detect which aspects are mentioned in this comment about the AI story above."
        )
        payload_type = "direct_comment"

    return payload_type, [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def process_records(
    stories_path: Path,
    comments_path: Path,
    subcomments_path: Path,
    threshold: int,
    max_records: int,
) -> Tuple[List[dict], List[dict]]:
    stories = load_stories(stories_path)
    comments, comment_index = load_comments(comments_path)
    subcomments = load_subcomments(subcomments_path)

    combined: List[dict] = []

    for row in comments:
        story = stories.get(row["story_id"], {})
        combined.append({
            "comment_id": row.get("comment_id"),
            "story_id": row.get("story_id"),
            "story_title": story.get("title", ""),
            "story_by": story.get("author", ""),
            "comment_time": row.get("time", ""),
            "comment_text": row.get("comment_text", ""),
            "comment_by": row.get("author", ""),
            "parent_id": row.get("parent_comment_id", ""),
            "parent_text": "",
            "comment_type": "comment",
        })

    for row in subcomments:
        story = stories.get(row["story_id"], {})
        parent_id = row.get("parent_comment_id", "")
        parent = comment_index.get(parent_id, {})
        combined.append({
            "comment_id": row.get("subcomment_id"),
            "story_id": row.get("story_id"),
            "story_title": story.get("title", ""),
            "story_by": story.get("author", ""),
            "comment_time": row.get("time", ""),
            "comment_text": row.get("subcomment_text", ""),
            "comment_by": row.get("author", ""),
            "parent_id": parent_id,
            "parent_text": parent.get("comment_text", ""),
            "comment_type": "subcomment",
        })

    filtered: List[dict] = []
    skipped = 0

    for idx, row in enumerate(combined, start=1):
        if max_records and idx > max_records:
            break

        combined_text = f"{row['story_title']} -- {row['comment_text']}"
        if row.get("parent_text"):
            combined_text = f"{combined_text} -- parent: {row['parent_text']}"

        passes, score, info = is_ai_related(combined_text, threshold=threshold)
        if not passes:
            skipped += 1
            continue

        row["ai_score"] = score
        row["match_info"] = info
        filtered.append(row)

    return filtered, combined


def write_filtered_csv(output_path: Path, rows: List[dict]) -> None:
    fieldnames = [
        "comment_id",
        "comment_type",
        "story_id",
        "story_title",
        "story_by",
        "comment_time",
        "comment_text",
        "comment_by",
        "parent_id",
        "parent_text",
        "ai_score",
        "match_info",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_payloads(output_path: Path, rows: List[dict]) -> None:
    with open(output_path, "w", encoding="utf-8") as outfile:
        for idx, row in enumerate(rows):
            payload_id = f"payload_{idx:06d}"
            payload_type, stage1_messages = build_stage1_messages(
                story_title=row.get("story_title", ""),
                comment_text=row.get("comment_text", ""),
                parent_text=row.get("parent_text", ""),
            )

            payload = {
                "payload_id": payload_id,
                "payload_type": payload_type,
                "story_id": row.get("story_id"),
                "story_title": row.get("story_title"),
                "comment_id": row.get("comment_id"),
                "comment_text": row.get("comment_text"),
                "parent_comment_id": row.get("parent_id", ""),
                "parent_comment_text": row.get("parent_text", ""),
                "stage1_messages": stage1_messages,
                "ai_score": row.get("ai_score"),
                "match_info": row.get("match_info"),
            }
            outfile.write(json.dumps(payload) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Filter AI posts/comments (v2) and build payloads")
    parser.add_argument("--stories", type=Path, default=Path("data/stories.csv"), help="Path to stories.csv")
    parser.add_argument("--comments", type=Path, default=Path("data/comments.csv"), help="Path to comments.csv")
    parser.add_argument("--subcomments", type=Path, default=Path("data/subcomments.csv"), help="Path to subcomments.csv")
    parser.add_argument("--threshold", type=int, default=5, help="AI score threshold (default 5)")
    parser.add_argument("--max-records", type=int, default=0, help="Max records to scan (0=all)")
    parser.add_argument("--filtered-output", type=Path, default=Path("hn_ai_filtered_v2.csv"), help="Filtered CSV output path")
    parser.add_argument("--payload-output", type=Path, default=Path("sentiment_payloads_v2.jsonl"), help="Payload JSONL output path")

    args = parser.parse_args()

    missing_inputs = [p for p in [args.stories, args.comments, args.subcomments] if not p.is_file()]
    if missing_inputs:
        print(f"ERROR: Missing input file(s): {', '.join(str(p) for p in missing_inputs)}")
        return

    filtered_rows, combined_rows = process_records(
        stories_path=args.stories,
        comments_path=args.comments,
        subcomments_path=args.subcomments,
        threshold=args.threshold,
        max_records=args.max_records,
    )

    print(f"Total records scanned: {len(combined_rows):,}")
    print(f"AI-related (score >= {args.threshold}): {len(filtered_rows):,}")

    write_filtered_csv(args.filtered_output, filtered_rows)
    print(f"Filtered CSV written to {args.filtered_output}")

    write_payloads(args.payload_output, filtered_rows)
    print(f"Payloads JSONL written to {args.payload_output}")


if __name__ == "__main__":
    main()
