#!/usr/bin/env python3
"""
Merge CSV shards from data/ folder and filter for AI-related posts using
a strict, multi-layered rule-based scoring system.

This script:
1. Reads all hn_stories_*.csv files from data/ directory
2. Applies 4-layer strict filtering with scoring:
   - Layer 1: Core vocabulary (terms, companies, jargon)
   - Layer 2: Technical phrases and patterns
   - Layer 3: Context and proximity rules
   - Layer 4: Structural markers (code, links, libraries)
3. Outputs rows exceeding threshold score to filtered CSV

CSV Schema:
- comment_id: Unique comment identifier
- story_id: Parent story ID
- story_title: Title of the story
- story_by: Story author username
- comment_time: Unix timestamp
- comment_text: Comment text content
- comment_by: Comment author username
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Tuple


# ============================================================================
# LAYER 1: DOMAIN VOCABULARY (Lists for precise matching)
# ============================================================================

# Core technical concepts (high confidence) - 2 points each
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

# AI companies and labs (high signal) - 2 points each
AI_COMPANIES = [
    "openai", "anthropic", "deepmind", "google ai", "meta ai",
    "hugging face", "huggingface", "cohere", "stability ai",
    "midjourney", "runway", "character.ai", "inflection",
    "adept", "ai21", "aleph alpha", "deepseek", "mistral",
    "together ai", "replicate", "modal labs"
]

# Technical jargon and benchmarks (very high confidence) - 3 points each
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

# Libraries and frameworks - 2 points each
LIBRARIES = [
    "pytorch", "tensorflow", "keras", "jax", "flax",
    "transformers", "diffusers", "accelerate", "bitsandbytes",
    "langchain", "llamaindex", "haystack", "semantic kernel",
    "autogen", "crewai", "guidance", "outlines", "instructor",
    "scikit-learn", "sklearn", "xgboost", "lightgbm",
    "opencv", "pillow", "albumentations", "timm"
]

# Model names (strong signal) - 2 points each
MODEL_NAMES = [
    "gpt-3", "gpt-4", "gpt-3.5", "chatgpt", "claude", "claude-3",
    "llama", "llama-2", "llama-3", "mistral", "mixtral",
    "gemini", "palm", "bard", "falcon", "mpt", "stablelm",
    "vicuna", "alpaca", "dolly", "koala", "orca", "wizard",
    "codellama", "starcoder", "phind", "deepseek-coder",
    "stable diffusion", "sdxl", "dall-e", "midjourney",
    "whisper", "wav2vec", "clip", "blip", "sam"
]

# ============================================================================
# LAYER 2: TECHNICAL PHRASES & PATTERNS (Regex-based)
# ============================================================================

TECHNICAL_PHRASES = [
    # How-to and guides (3 points)
    (r"(how to|guide to|tutorial on).*(fine-tune|train|implement|optimize|deploy)", 3),
    # Comparisons (3 points)
    (r".*(vs\.?|versus|compared to|better than).*(gpt|claude|llama|mistral|gemini)", 3),
    # Implementation details (3 points)
    (r"(implement|build|create).*(chatbot|agent|rag|pipeline)", 3),
    # Performance discussions (2 points)
    (r"(latency|throughput|tokens per second|inference speed|batch size)", 2),
    # Cost/pricing (2 points)
    (r"(cost|price|pricing|api cost|\$.*per.*token)", 2),
    # Technical problems (3 points)
    (r"(error|issue|problem|bug).*(training|inference|model|fine-tun)", 3),
    # Research/paper mentions (3 points)
    (r"(paper|research|arxiv|according to.*study)", 3),
    # Benchmarks (3 points)
    (r"(benchmark|score|evaluation|performance on).*(mmlu|hellaswag|humaneval)", 3)
]

# ============================================================================
# LAYER 3: CONTEXT & PROXIMITY RULES
# ============================================================================

# Weak terms that need strong context nearby (within 10 words)
CONTEXT_RULES = {
    "model": ["transformer", "train", "fine-tune", "inference", "weights", "architecture", "release", "open source"],
    "ai": ["safety", "alignment", "research", "lab", "ethics", "training", "model", "system"],
    "ml": ["pipeline", "training", "model", "inference", "production", "deployment"],
    "data": ["training", "dataset", "annotation", "quality", "synthetic", "augmentation"],
    "learning": ["machine", "deep", "reinforcement", "supervised", "unsupervised", "transfer"]
}

# ============================================================================
# LAYER 4: STRUCTURAL MARKERS
# ============================================================================

STRUCTURAL_PATTERNS = [
    # Code blocks (5 points - very strong signal)
    (r"```[\s\S]*?```", 5),
    (r"`[^`]+`", 2),  # Inline code (2 points)
    # Links to technical resources (3 points each)
    (r"arxiv\.org/abs/", 3),
    (r"github\.com/[\w\-]+/[\w\-]+", 3),
    (r"huggingface\.co/(models|datasets|spaces)", 3),
    # Package installation (4 points)
    (r"pip install|conda install|npm install", 4),
    # Imports (3 points)
    (r"(import|from)\s+(torch|tensorflow|transformers|langchain)", 3),
    # Config/parameters (2 points)
    (r"(config|hyperparameters?):\s*\{", 2)
]


def calculate_ai_score(text: str) -> Tuple[int, str]:
    """
    Calculate AI relevance score using multi-layered rule-based system.
    
    Returns:
        Tuple of (score, debug_info)
    """
    if not text:
        return 0, ""
    
    text_lower = text.lower()
    score = 0
    matches = []
    
    # LAYER 1: Core vocabulary
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
    
    # LAYER 2: Technical phrases
    for pattern, points in TECHNICAL_PHRASES:
        if re.search(pattern, text_lower, re.IGNORECASE):
            score += points
            matches.append(f"phrase:+{points}")
    
    # LAYER 3: Context proximity
    words = text_lower.split()
    for weak_term, context_words in CONTEXT_RULES.items():
        # Find all positions of weak term
        weak_positions = [i for i, w in enumerate(words) if weak_term in w]
        
        for pos in weak_positions:
            # Check 10-word window around weak term
            window_start = max(0, pos - 10)
            window_end = min(len(words), pos + 10)
            window = words[window_start:window_end]
            
            # If any context word found in window, award points
            if any(ctx in ' '.join(window) for ctx in context_words):
                score += 2
                matches.append(f"context:{weak_term}")
                break  # Only count once per weak term
    
    # LAYER 4: Structural markers
    for pattern, points in STRUCTURAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            score += points
            matches.append(f"struct:+{points}")
    
    debug_info = ",".join(matches[:5]) if matches else "no_match"
    return score, debug_info


def is_ai_related(text: str, threshold: int = 5) -> Tuple[bool, int, str]:
    """
    Check if text is AI-related using strict multi-layer scoring.
    
    Args:
        text: String to analyze (story title or comment text)
        threshold: Minimum score required to pass (default: 5)
    
    Returns:
        Tuple of (passes_threshold, score, debug_info)
    """
    score, debug_info = calculate_ai_score(text)
    return score >= threshold, score, debug_info


def merge_and_filter_csv(input_dir: Path, output_path: Path, max_rows: int = 0, 
                        score_threshold: int = 5, review_queue_path: Path = None):
    """
    Merge CSV shards and filter for AI-related posts using strict scoring.
    
    Args:
        input_dir: Directory containing hn_stories_*.csv files
        output_path: Path to output filtered CSV
        max_rows: Maximum rows to process (0 = unlimited)
        score_threshold: Minimum score to accept row (default: 5)
        review_queue_path: Optional path for manual review queue (scores 3-4)
    """
    # Find all CSV shards
    csv_files = sorted(input_dir.glob("hn_stories_*.csv"))
    
    if not csv_files:
        print(f"ERROR: No CSV files found matching pattern hn_stories_*.csv in {input_dir}")
        return
    
    print(f"Found {len(csv_files)} CSV shard(s):")
    for csv_file in csv_files:
        file_size_mb = csv_file.stat().st_size / (1024 * 1024)
        print(f"  - {csv_file.name} ({file_size_mb:.2f} MB)")
    
    print(f"\nFiltering settings:")
    print(f"  Score threshold: {score_threshold}+")
    print(f"  Review queue: {'enabled' if review_queue_path else 'disabled'}")
    
    total_rows = 0
    filtered_rows = 0
    review_queue_rows = 0
    score_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 10: 0, 15: 0}
    
    # Expected CSV header
    expected_header = [
        "comment_id", "story_id", "story_title", "story_by",
        "comment_time", "comment_text", "comment_by"
    ]
    
    review_writer = None
    if review_queue_path:
        review_file = open(review_queue_path, "w", newline="", encoding="utf-8")
    
    with open(output_path, "w", newline="", encoding="utf-8") as outfile:
        writer = None
        
        for csv_file in csv_files:
            print(f"\nProcessing {csv_file.name}...")
            
            with open(csv_file, "r", encoding="utf-8") as infile:
                reader = csv.DictReader(infile)
                
                # Verify header matches expected schema
                if not reader.fieldnames:
                    print(f"  WARNING: {csv_file.name} has no header, skipping")
                    continue
                
                # Initialize writers with first file's header
                if writer is None:
                    fieldnames_with_score = list(reader.fieldnames) + ["ai_score", "match_info"]
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                    
                    if review_queue_path:
                        review_writer = csv.DictWriter(review_file, fieldnames=fieldnames_with_score)
                        review_writer.writeheader()
                    
                    print(f"  Output header: {', '.join(reader.fieldnames)}")
                
                for row in reader:
                    total_rows += 1
                    
                    # Calculate combined score from title + comment
                    title_passes, title_score, title_info = is_ai_related(
                        row.get("story_title", ""), threshold=0
                    )
                    comment_passes, comment_score, comment_info = is_ai_related(
                        row.get("comment_text", ""), threshold=0
                    )
                    
                    combined_score = title_score + comment_score
                    combined_info = f"t:{title_score};c:{comment_score}"
                    
                    # Track score distribution
                    if combined_score == 0:
                        score_distribution[0] += 1
                    elif combined_score <= 2:
                        score_distribution[2] += 1
                    elif combined_score <= 4:
                        score_distribution[4] += 1
                    elif combined_score <= 9:
                        score_distribution[5] += 1
                    elif combined_score <= 14:
                        score_distribution[10] += 1
                    else:
                        score_distribution[15] += 1
                    
                    # Accept if meets threshold
                    if combined_score >= score_threshold:
                        writer.writerow(row)
                        filtered_rows += 1
                        
                        if filtered_rows % 1000 == 0:
                            print(f"  Accepted {filtered_rows:,} rows (score {score_threshold}+)...")
                    
                    # Add to review queue if medium score (3-4)
                    elif review_writer and 3 <= combined_score < score_threshold:
                        row_with_score = dict(row)
                        row_with_score["ai_score"] = str(combined_score)
                        row_with_score["match_info"] = combined_info
                        review_writer.writerow(row_with_score)
                        review_queue_rows += 1
                    
                    # Progress indicator every 50k rows
                    if total_rows % 50000 == 0:
                        print(f"  Processed {total_rows:,} rows...")
                    
                    # Check max_rows limit
                    if max_rows > 0 and total_rows >= max_rows:
                        print(f"  Reached max_rows limit ({max_rows:,}), stopping")
                        break
            
            # Break outer loop if max_rows reached
            if max_rows > 0 and total_rows >= max_rows:
                break
    
    if review_queue_path:
        review_file.close()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"FILTERING SUMMARY:")
    print(f"  Total rows processed: {total_rows:,}")
    print(f"  Rows accepted (score {score_threshold}+): {filtered_rows:,} ({filtered_rows/total_rows*100:.2f}%)")
    if review_queue_path:
        print(f"  Review queue (score 3-{score_threshold-1}): {review_queue_rows:,} ({review_queue_rows/total_rows*100:.2f}%)")
    print(f"  Rows rejected (score <3): {total_rows - filtered_rows - review_queue_rows:,}")
    print(f"\nScore distribution:")
    print(f"  0 points: {score_distribution[0]:,}")
    print(f"  1-2 points: {score_distribution[2]:,}")
    print(f"  3-4 points: {score_distribution[4]:,}")
    print(f"  5-9 points: {score_distribution[5]:,}")
    print(f"  10-14 points: {score_distribution[10]:,}")
    print(f"  15+ points: {score_distribution[15]:,}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge CSV shards and filter for AI-related posts using strict multi-layer scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scoring system (4 layers):
  Layer 1: Core vocabulary (2-3 points per match)
  Layer 2: Technical phrases (2-3 points per pattern)
  Layer 3: Context proximity (2 points for weak terms with context)
  Layer 4: Structural markers (2-5 points for code/links)

Default threshold: 5 points (strict technical filter)
  - Lower threshold (3-4): More permissive, captures broader discussions
  - Higher threshold (7+): Very strict, only deep technical content
"""
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing hn_stories_*.csv files (default: data/)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("hn_ai_filtered.csv"),
        help="Output CSV path (default: hn_ai_filtered.csv)"
    )
    
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Maximum rows to process across all shards (0 = unlimited, default: 0)"
    )
    
    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Minimum AI score to accept row (default: 5, range: 1-20)"
    )
    
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=None,
        help="Output path for manual review queue (rows with score 3-4)"
    )
    
    args = parser.parse_args()
    
    # Validate input directory exists
    if not args.input_dir.is_dir():
        print(f"ERROR: Input directory does not exist: {args.input_dir}")
        return
    
    # Validate threshold
    if args.threshold < 1 or args.threshold > 20:
        print(f"ERROR: Threshold must be between 1 and 20 (got {args.threshold})")
        return
    
    # Run merge and filter
    merge_and_filter_csv(
        input_dir=args.input_dir,
        output_path=args.output,
        max_rows=args.max_rows,
        score_threshold=args.threshold,
        review_queue_path=args.review_queue
    )
    
    print(f"\nFiltered CSV saved to: {args.output}")
    if args.review_queue:
        print(f"Review queue saved to: {args.review_queue}")


if __name__ == "__main__":
    main()
