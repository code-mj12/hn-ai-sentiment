#!/usr/bin/env python3
"""
Prepare TWO-STAGE sentiment analysis payloads from filtered AI posts.

STAGE 1: Aspect Detection
- Binary classification: is aspect mentioned? YES/NO
- Extract evidence span showing where aspect appears
- Provide confidence score (0.0-1.0)

STAGE 2: Aspect Sentiment (conditional - only if aspects detected)
- Classify sentiment: positive, neutral, negative, mixed
- Provide sentiment score (-1.0 to +1.0)
- Explain reasoning for classification

This script generates payloads with Stage 1 messages.
Stage 2 messages are generated after Stage 1 completes.

Payload types:
1. Type 1 (story_heading): Story title only
2. Type 2 (direct_comment): Story title + direct comment
3. Type 3 (nested_comment): Story title + parent + nested comment

Input CSV Schema:
- comment_id, story_id, story_title, story_by, comment_time, comment_text, comment_by

Output JSONL Schema:
{
  "payload_id": "payload_000042",
  "payload_type": "direct_comment",
  "story_id": "12345",
  "story_title": "...",
  "comment_id": "67890",
  "comment_text": "...",
  "stage1_messages": [...]  // Aspect detection messages
}
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict


# STAGE 1: Aspect Detection System Prompt
STAGE1_SYSTEM_PROMPT = """You are an aspect detection system for AI-related Hacker News discussions.

Your task: Identify which aspects are mentioned in the text. For each aspect:
1. present: true/false (is it explicitly or implicitly mentioned?)
2. evidence: exact text span or brief summary showing where it appears
3. confidence: 0.0 to 1.0 (how certain are you?)

ASPECT TAXONOMY (12 aspects):
- performance_speed: execution time, latency, throughput, inference speed, training time
- accuracy_reliability: correctness, precision, recall, error rates, model quality, consistency
- security: vulnerabilities, exploits, adversarial attacks, data breaches, access control
- privacy: data protection, PII handling, anonymization, surveillance, data collection
- usability_ux: ease of use, UI/UX, developer experience, API design, documentation
- cost_price: pricing models, compute costs, API fees, ROI, affordability, efficiency
- ethics: bias, fairness, transparency, accountability, AI alignment, moral implications
- regulation_policy: laws, governance, compliance, government intervention, frameworks
- community_tone: open source culture, collaboration, toxicity, inclusivity, discourse
- business_model: monetization, competitive dynamics, market strategy, partnerships
- innovation: novelty, breakthroughs, technical advancement, research impact, SOTA
- skepticism_hype: doubt about claims, overhype criticism, feasibility concerns, reality checks

RULES:
- Only mark present=true if the aspect is ACTUALLY discussed (not tangentially related)
- Provide exact text span as evidence when possible
- If multiple aspects apply, detect ALL of them
- If NO aspects are relevant, return empty aspects array: {"aspects": []}
- Be strict: don't hallucinate aspects that aren't there

Respond with valid JSON only:
{
  "aspects": [
    {
      "aspect": "performance_speed",
      "present": true,
      "evidence": "runs 10x faster than GPT-4",
      "confidence": 0.95
    }
  ]
}"""


def create_stage1_messages(story_title: str, comment_text: str = None) -> list:
    """
    Create Stage 1 (aspect detection) messages.
    
    Args:
        story_title: Story title text
        comment_text: Optional comment text (for Type 2/3 payloads)
    
    Returns:
        Messages array for aspect detection
    """
    if comment_text:
        user_content = f"""Story: {story_title}

Comment: {comment_text}

Detect which aspects are mentioned in this comment about the AI story above."""
    else:
        user_content = f"""Story: {story_title}

Detect which aspects are mentioned in this AI story title."""
    
    return [
        {"role": "system", "content": STAGE1_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def create_story_heading_payload(story_id: str, story_title: str, payload_id: str) -> dict:
    """
    Type 1: Story heading only (Stage 1 only).
    """
    return {
        "payload_id": payload_id,
        "payload_type": "story_heading",
        "story_id": story_id,
        "story_title": story_title,
        "stage1_messages": create_stage1_messages(story_title)
    }


def create_direct_comment_payload(
    story_id: str,
    story_title: str,
    comment_id: str,
    comment_text: str,
    payload_id: str
) -> dict:
    """
    Type 2: Story title + direct comment (Stage 1 only).
    """
    return {
        "payload_id": payload_id,
        "payload_type": "direct_comment",
        "story_id": story_id,
        "story_title": story_title,
        "comment_id": comment_id,
        "comment_text": comment_text,
        "stage1_messages": create_stage1_messages(story_title, comment_text)
    }


def create_nested_comment_payload(
    story_id: str,
    story_title: str,
    comment_id: str,
    comment_text: str,
    parent_comment_id: str,
    parent_comment_text: str,
    payload_id: str
) -> dict:
    """
    Type 3: Story title + parent comment + nested comment (Stage 1 only).
    """
    combined_text = f"Parent: {parent_comment_text}\n\nReply: {comment_text}"
    
    return {
        "payload_id": payload_id,
        "payload_type": "nested_comment",
        "story_id": story_id,
        "story_title": story_title,
        "comment_id": comment_id,
        "parent_comment_id": parent_comment_id,
        "comment_text": comment_text,
        "parent_comment_text": parent_comment_text,
        "stage1_messages": create_stage1_messages(story_title, combined_text)
    }


def prepare_payloads(
    input_csv: Path,
    output_jsonl: Path,
    max_payloads: int = 0,
    include_story_headings: bool = False,
    include_type3: bool = False
):
    """
    Generate sentiment analysis payloads from filtered CSV.
    
    Three payload types (mutually exclusive per comment):
    - Type 1 (optional): Story heading only
    - Type 2 (main): Story + direct comment (one payload per comment)
    - Type 3 (optional): Story + parent + nested comment (only if parent exists)
    
    IMPORTANT: Each comment generates ONE payload (either Type 2 or Type 3, not both)
    
    Args:
        input_csv: Path to filtered AI posts CSV
        output_jsonl: Path to output JSONL file
        max_payloads: Maximum payloads to generate (0 = unlimited)
        include_story_headings: Generate Type 1 payloads (story title only)
        include_type3: Generate Type 3 payloads (requires parent_id in CSV)
    """
    
    print(f"Reading CSV from {input_csv}...")
    
    # First pass: collect metadata
    story_metadata = {}  # story_id -> {title, author}
    comments_data = []   # list of comment rows
    
    with open(input_csv, "r", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        
        for row in reader:
            story_id = row["story_id"]
            
            # Store story metadata (only once per story)
            if story_id not in story_metadata:
                story_metadata[story_id] = {
                    "title": row["story_title"],
                    "author": row["story_by"]
                }
            
            # Store comment data
            comments_data.append(row)
    
    print(f"Found {len(story_metadata):,} unique stories")
    print(f"Found {len(comments_data):,} total comments")
    
    payload_count = 0
    type1_count = 0
    type2_count = 0
    type3_count = 0
    
    with open(output_jsonl, "w", encoding="utf-8") as outfile:
        
        # ===== TYPE 1: Story headings (optional) =====
        if include_story_headings:
            print("\nGenerating Type 1 (story_heading) payloads...")
            for story_id, metadata in story_metadata.items():
                if max_payloads > 0 and payload_count >= max_payloads:
                    break
                
                payload_id = f"payload_{payload_count:06d}"
                payload = create_story_heading_payload(
                    story_id=story_id,
                    story_title=metadata["title"],
                    payload_id=payload_id
                )
                
                outfile.write(json.dumps(payload) + "\n")
                payload_count += 1
                type1_count += 1
                
                if type1_count % 5000 == 0:
                    print(f"  Generated {type1_count:,} Type 1 payloads...")
        
        # ===== TYPE 2 & 3: Comments =====
        # Each comment gets ONE payload: Type 2 (default) or Type 3 (if has parent_id)
        print("\nGenerating Type 2 (direct_comment) and Type 3 (nested_comment) payloads...")
        
        for row in comments_data:
            if max_payloads > 0 and payload_count >= max_payloads:
                break
            
            comment_id = row["comment_id"]
            comment_text = row.get("comment_text", "").strip()
            story_id = row["story_id"]
            story_title = story_metadata[story_id]["title"]
            
            # Skip empty comments
            if not comment_text:
                continue
            
            payload_id = f"payload_{payload_count:06d}"
            
            # Check if this comment has a parent (for Type 3)
            parent_id = row.get("parent_id") or row.get("parent_comment_id")
            has_parent = bool(parent_id and str(parent_id).strip())
            
            if include_type3 and has_parent:
                # Type 3: Nested comment with parent context
                parent_text = row.get("parent_comment_text", "")
                
                if parent_text and parent_text.strip():
                    payload = create_nested_comment_payload(
                        story_id=story_id,
                        story_title=story_title,
                        comment_id=comment_id,
                        comment_text=comment_text,
                        parent_comment_id=parent_id,
                        parent_comment_text=parent_text,
                        payload_id=payload_id
                    )
                    outfile.write(json.dumps(payload) + "\n")
                    payload_count += 1
                    type3_count += 1
                    
                    if (type2_count + type3_count) % 5000 == 0:
                        print(f"  Generated {type2_count:,} Type 2 + {type3_count:,} Type 3 payloads...")
                    continue
            
            # Type 2: Direct comment (story + comment, no parent context)
            payload = create_direct_comment_payload(
                story_id=story_id,
                story_title=story_title,
                comment_id=comment_id,
                comment_text=comment_text,
                payload_id=payload_id
            )
            
            outfile.write(json.dumps(payload) + "\n")
            payload_count += 1
            type2_count += 1
            
            if (type2_count + type3_count) % 5000 == 0:
                print(f"  Generated {type2_count:,} Type 2 + {type3_count:,} Type 3 payloads...")
    
    # Summary
    total_payloads = type1_count + type2_count + type3_count
    print(f"\n{'='*70}")
    print(f"SUMMARY:")
    print(f"  Type 1 (story_heading):     {type1_count:>10,}")
    print(f"  Type 2 (direct_comment):    {type2_count:>10,}")
    print(f"  Type 3 (nested_comment):    {type3_count:>10,}")
    print(f"  {'─'*48}")
    print(f"  Total payloads:             {total_payloads:>10,}")
    print(f"  Expected (from CSV):        {len(comments_data):>10,}")
    print(f"  Match: {'✅ YES' if total_payloads == len(comments_data) else '❌ NO (check config)'}")
    print(f"  Output: {output_jsonl}")
    print(f"{'='*70}")


# STAGE 2: Aspect Sentiment System Prompt
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


def create_stage2_messages(comment_text: str, detected_aspects: list) -> list:
    """
    Create Stage 2 (aspect sentiment) messages.
    
    Args:
        comment_text: Comment text to analyze
        detected_aspects: List of detected aspects from Stage 1 with evidence
    
    Returns:
        Messages array for sentiment classification
    """
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


def prepare_stage2_payloads(
    stage1_results: Path,
    output_jsonl: Path,
    min_confidence: float = 0.7
):
    """
    Generate Stage 2 payloads from Stage 1 results.
    Filters by confidence threshold and creates stage2_messages.
    
    Args:
        stage1_results: Path to Stage 1 output JSONL
        output_jsonl: Path to Stage 2 input JSONL
        min_confidence: Minimum confidence to qualify for Stage 2
    """
    print(f"\nGenerating Stage 2 payloads from Stage 1 results...")
    print(f"  Filtering by confidence >= {min_confidence}")
    print(f"  Input: {stage1_results}")
    print(f"  Output: {output_jsonl}\n")
    
    stage2_count = 0
    qualified_count = 0
    no_aspects_count = 0
    below_confidence_count = 0
    
    with open(stage1_results, "r", encoding="utf-8") as infile, \
         open(output_jsonl, "w", encoding="utf-8") as outfile:
        
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
            stage2_count += 1
            
            if not aspects:
                no_aspects_count += 1
                if stage2_count <= 5 or stage2_count % 100 == 0:
                    print(f"[S2Prep {stage2_count}] {record['payload_id']}: No aspects, skip Stage 2")
                continue
            
            # Filter by confidence
            qualified_aspects = [
                a for a in aspects 
                if a.get("confidence", 0) >= min_confidence
            ]
            
            if not qualified_aspects:
                below_confidence_count += 1
                if stage2_count <= 5 or stage2_count % 100 == 0:
                    aspect_names = [a["aspect"] for a in aspects]
                    print(f"[S2Prep {stage2_count}] {record['payload_id']}: Aspects {aspect_names} below confidence {min_confidence}, skip Stage 2")
                continue
            
            # Generate Stage 2 messages
            comment_text = record.get("comment_text", "")
            stage2_messages = create_stage2_messages(comment_text, qualified_aspects)
            
            stage2_payload = {
                "payload_id": record["payload_id"],
                "payload_type": record.get("payload_type"),
                "story_id": record.get("story_id"),
                "comment_text": comment_text,
                "detected_aspects": qualified_aspects,
                "stage2_messages": stage2_messages,
                "stage1_elapsed": record.get("stage1_elapsed", 0)
            }
            
            outfile.write(json.dumps(stage2_payload) + "\n")
            
            qualified_count += 1
            if stage2_count <= 5 or qualified_count % 100 == 0:
                aspect_names = [a["aspect"] for a in qualified_aspects[:2]]
                print(f"[S2Prep {stage2_count}] {record['payload_id']}: ✅ {len(qualified_aspects)} aspects → Stage 2 ({', '.join(aspect_names)}...)")
    
    # Summary
    reduction_pct = 100 * (1 - qualified_count / max(1, stage2_count))
    print(f"\n{'='*60}")
    print(f"STAGE 2 PREP SUMMARY:")
    print(f"  Input records: {stage2_count:,}")
    print(f"  No aspects: {no_aspects_count:,}")
    print(f"  Below confidence {min_confidence}: {below_confidence_count:,}")
    print(f"  Qualified for Stage 2: {qualified_count:,}")
    print(f"  Filtering efficiency: {reduction_pct:.0f}% reduction")
    print(f"  Output: {output_jsonl}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare sentiment analysis payloads (Stage 1 and Stage 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
PAYLOAD TYPES (mutually exclusive per comment):
  Type 1 (optional): Story heading only → LLM
  Type 2 (default): Story + direct comment → LLM  
  Type 3 (optional): Story + parent comment + reply → LLM
  
Each comment generates exactly ONE payload type.
Use --include-story-headings to add Type 1 (story titles alone).
Use --include-type3 to create Type 3 for nested comments (requires parent_id column).
        """
    )
    
    # Stage 1 Mode: Generate Stage 1 payloads from CSV
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("hn_ai_filtered.csv"),
        help="Input filtered CSV (Stage 1 mode, default: hn_ai_filtered.csv)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sentiment_payloads.jsonl"),
        help="Output JSONL path (Stage 1 mode, default: sentiment_payloads.jsonl)"
    )
    
    parser.add_argument(
        "--max-payloads",
        type=int,
        default=0,
        help="Maximum payloads to generate (0 = unlimited, default: 0)"
    )
    
    parser.add_argument(
        "--include-story-headings",
        action="store_true",
        help="Include Type 1 (story_heading) payloads in addition to Type 2/3"
    )
    
    parser.add_argument(
        "--include-type3",
        action="store_true",
        help="Generate Type 3 (nested_comment) payloads when parent_id exists (requires parent_comment_text in CSV)"
    )
    
    # Stage 2 Mode: Generate Stage 2 payloads from Stage 1 results
    parser.add_argument(
        "--stage2-results",
        type=Path,
        default=None,
        help="Stage 1 results file (Stage 2 mode, enables Stage 2 payload generation)"
    )
    
    parser.add_argument(
        "--stage2-output",
        type=Path,
        default=Path("stage2_payloads.jsonl"),
        help="Stage 2 output path (Stage 2 mode, default: stage2_payloads.jsonl)"
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence for Stage 2 filtering (default: 0.7)"
    )
    
    args = parser.parse_args()
    
    # Mode: Stage 2 payload generation
    if args.stage2_results:
        if not args.stage2_results.is_file():
            print(f"ERROR: Stage 1 results file does not exist: {args.stage2_results}")
            return
        
        prepare_stage2_payloads(
            stage1_results=args.stage2_results,
            output_jsonl=args.stage2_output,
            min_confidence=args.min_confidence
        )
        
        print(f"\nStage 2 payloads saved to: {args.stage2_output}")
        return
    
    # Mode: Stage 1 payload generation (default)
    if not args.input.is_file():
        print(f"ERROR: Input file does not exist: {args.input}")
        return
    
    prepare_payloads(
        input_csv=args.input,
        output_jsonl=args.output,
        max_payloads=args.max_payloads,
        include_story_headings=args.include_story_headings,
        include_type3=args.include_type3
    )
    
    print(f"\nStage 1 payloads saved to: {args.output}")


if __name__ == "__main__":
    main()
