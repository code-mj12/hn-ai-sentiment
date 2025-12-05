"""Build LLM-ready sentiment analysis payloads from Hacker News threads.

Inputs: a CSV produced by `script_no_pandas.py` (default: hacker_news_ai_threads.csv
or the fallback hacker_news_ai_threads_new.csv). Each row contains either a root
story or a comment linked to its story.

Outputs:
* `sentiment_llm_payload.jsonl` - newline-delimited JSON documents for LLM
  inference with cleaned text, metadata, heuristic aspect hints, and an
  instruction string describing the sentiment/aspect taxonomy.
* Optional CSV mirror with the same information for auditing/training.

The script cleans HTML, strips code fences, removes emojis, normalizes text,
extracts lightweight metadata (model names, keyword-aspect hints), and embeds a
prompt template so you can feed each row directly to GPT-4, Claude, etc.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_INPUT = "hacker_news_ai_threads_new.csv"
DEFAULT_JSONL = "sentiment_llm_payload.jsonl"
DEFAULT_CSV = "sentiment_llm_payload.csv"
DEFAULT_MAX_RECORDS = None

EMOJI_RE = re.compile(
    "[\U0001F600-\U0001F64F]|"  # emoticons
    "[\U0001F300-\U0001F5FF]|"  # symbols & pictographs
    "[\U0001F680-\U0001F6FF]|"  # transport & map symbols
    "[\U0001F1E0-\U0001F1FF]"   # flags
)
CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```|`[^`]+`")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_RE = re.compile(r"\s+")

MODEL_KEYWORDS = {
    "openai": ["openai", "gpt", "chatgpt", "o3", "o1", "sora"],
    "anthropic": ["anthropic", "claude"],
    "google": ["gemini", "google", "bard"],
    "meta": ["llama", "metallama", "facebook"],
    "mistral": ["mistral", "mixtral"],
}

ASPECT_TAXONOMY = [
    {"key": "performance_speed", "description": "Latency, throughput, efficiency, resource use"},
    {"key": "accuracy_reliability", "description": "Correctness, hallucinations, evaluation quality"},
    {"key": "security", "description": "Vulnerabilities, exploits, safeguards"},
    {"key": "privacy", "description": "Data sharing, user privacy, surveillance"},
    {"key": "usability_ux", "description": "User experience, workflow, ease of use"},
    {"key": "cost_price", "description": "Pricing, licensing, monetization burden"},
    {"key": "ethics", "description": "Bias, fairness, responsible use, societal impact"},
    {"key": "regulation_policy", "description": "Law, policy, governance, compliance"},
    {"key": "community_tone", "description": "Tone of discourse, collaboration, reputation"},
    {"key": "business_model", "description": "Strategy, revenue, market dynamics"},
    {"key": "other", "description": "Anything else relevant"},
]

ASPECT_KEYWORDS = {
    "performance_speed": ["speed", "latency", "slow", "fast", "throughput", "optimize"],
    "accuracy_reliability": ["accurate", "accuracy", "hallucinate", "reliability", "correct", "bug"],
    "security": ["security", "exploit", "breach", "hack", "vulnerability"],
    "privacy": ["privacy", "data", "tracking", "surveillance"],
    "usability_ux": ["ux", "ui", "workflow", "experience", "interface", "friction"],
    "cost_price": ["cost", "pricing", "expensive", "cheap", "subscription", "tier"],
    "ethics": ["ethic", "bias", "fairness", "responsible", "harm"],
    "regulation_policy": ["regulation", "policy", "law", "compliance", "act", "bill"],
    "community_tone": ["community", "tone", "toxic", "friendly", "culture"],
    "business_model": ["business", "monetize", "revenue", "market", "competition"],
}

PROMPT_TEMPLATE = """You are an expert analyst. Given the text and metadata, output STRICT JSON with this schema:\n{{\n  \"id\": <int>,\n  \"sentiment\": \"positive|neutral|negative|mixed\",\n  \"aspects\": [\n    {{\n      \"aspect\": <string from allowed list>,\n      \"present\": true|false,\n      \"evidence\": <string|null>,\n      \"sentiment\": <sentiment|null>,\n      \"confidence\": <0-1>,\n      \"implicit\": true|false\n    }}\n  ]\n}}\n\nRules:\n1. Aspect list = {aspect_list}.\n2. Detect aspects first (binary), then rate sentiment only when present/implicit.\n3. Use \"other\" for valid but uncategorized aspects.\n4. Consider metadata (root story title, author, depth) for context.\n5. If the text is empty or only metadata, emit sentiment \"neutral\" with all aspects present=false.\n""".replace("{aspect_list}", ", ".join(a["key"] for a in ASPECT_TAXONOMY))


@dataclass
class Record:
    node_id: int
    root_story_id: int
    node_type: str
    depth: int
    cleaned_text: str
    normalized_text: str
    model_tags: List[str]
    aspect_hints: List[str]
    context: Dict[str, str]


def safe_text(value: object) -> str:
    """Return a clean string, collapsing NaN/None to empty."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        # Non-scalar types are fine; fall through to string conversion
        pass
    return str(value)


def strip_code(text: str) -> str:
    return CODE_BLOCK_RE.sub(" ", text)


def remove_emojis(text: str) -> str:
    return EMOJI_RE.sub(" ", text)


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = html.unescape(value)
    text = strip_code(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = remove_emojis(text)
    text = text.replace("\u200b", " ")
    text = text.replace("\xa0", " ")
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def normalize_text(value: str) -> str:
    text = value.lower()
    translator = str.maketrans({punct: " " for punct in string.punctuation})
    text = text.translate(translator)
    text = MULTISPACE_RE.sub(" ", text)
    return text.strip()


def detect_model_tags(text: str) -> List[str]:
    tokens = text.lower()
    tags = []
    for tag, keywords in MODEL_KEYWORDS.items():
        if any(keyword in tokens for keyword in keywords):
            tags.append(tag)
    return tags


def heuristic_aspects(text: str) -> List[str]:
    lowered = text.lower()
    hints = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            hints.append(aspect)
    return hints


def build_record(row: pd.Series) -> Record:
    text_source = safe_text(row.get("text_clean"))
    if not text_source:
        text_source = safe_text(row.get("text"))
    cleaned = clean_text(text_source)
    normalized = normalize_text(cleaned)
    node_id = row.get("node_id")
    root_story_id = row.get("root_story_id")
    try:
        node_id_int = int(node_id)
    except (TypeError, ValueError):
        node_id_int = -1
    try:
        root_story_id_int = int(root_story_id)
    except (TypeError, ValueError):
        root_story_id_int = -1
    try:
        depth_int = int(row.get("comment_depth") or 0)
    except (TypeError, ValueError):
        depth_int = 0
    parent_raw = row.get("parent_id")
    try:
        parent_id_int = int(parent_raw) if parent_raw not in (None, "") else None
    except (TypeError, ValueError):
        parent_id_int = None

    conversation_role = safe_text(row.get("conversation_role")) or (
        "story"
        if (row.get("node_type") or "").lower() == "story"
        else ("direct_reply" if depth_int <= 1 else "thread_reply")
    )

    return Record(
        node_id=node_id_int,
        root_story_id=root_story_id_int,
        node_type=(row.get("node_type") or "").lower(),
        depth=depth_int,
        cleaned_text=cleaned,
        normalized_text=normalized,
        model_tags=detect_model_tags(cleaned),
        aspect_hints=heuristic_aspects(cleaned),
        context={
            "root_title": safe_text(row.get("root_story_title")),
            "root_author": safe_text(row.get("root_story_author")),
            "root_url": safe_text(row.get("root_story_url")),
            "root_text": safe_text(row.get("root_story_text_clean")),
            "node_author": safe_text(row.get("node_author")),
            "node_time": safe_text(row.get("node_time_iso")),
            "parent_id": parent_id_int,
            "parent_author": safe_text(row.get("parent_author")),
            "parent_text": safe_text(row.get("parent_text_clean")),
            "parent_type": safe_text(row.get("parent_type")),
            "lineage_to_story": safe_text(row.get("lineage_to_story")),
            "conversation_role": conversation_role,
        },
    )


def make_llm_payload(record: Record) -> Dict:
    return {
        "id": record.node_id,
        "root_story_id": record.root_story_id,
        "node_type": record.node_type,
        "comment_depth": record.depth,
        "text": record.cleaned_text,
        "normalized_text": record.normalized_text,
        "model_tags": record.model_tags,
        "aspect_hints": record.aspect_hints,
        "context": record.context,
        "prompt": PROMPT_TEMPLATE,
        "taxonomy": ASPECT_TAXONOMY,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LLM payloads for sentiment/aspect analysis")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Thread CSV produced by script_no_pandas")
    parser.add_argument("--jsonl-output", default=DEFAULT_JSONL, help="Path for JSONL payload")
    parser.add_argument("--csv-output", default=DEFAULT_CSV, help="Path for optional CSV mirror")
    parser.add_argument(
        "--output-format",
        choices=["jsonl", "csv", "both"],
        default="jsonl",
        help="Choose which artifacts to emit (default: jsonl only)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=DEFAULT_MAX_RECORDS,
        help="Limit number of rows (after filtering) to speed up experiments",
    )
    parser.add_argument(
        "--include-stories",
        action="store_true",
        help="Include root stories themselves; default processes comments only",
    )
    parser.add_argument(
        "--min-text-length",
        type=int,
        default=10,
        help="Drop rows whose cleaned text is shorter than this many characters",
    )
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    required_cols = ["node_id", "root_story_id", "node_type", "comment_depth"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {path}")
    return df


def filter_records(df: pd.DataFrame, include_stories: bool, min_text_len: int) -> pd.DataFrame:
    base = df.copy()
    base["node_type_str"] = base["node_type"].fillna("").astype(str).str.lower()
    if not include_stories:
        base = base[base["node_type_str"] == "comment"]
    text_column = "text_clean" if "text_clean" in base.columns else "text"
    base["text_for_len"] = base[text_column].fillna("").astype(str)
    base = base[base["text_for_len"].str.len() >= min_text_len]
    base = base.drop(columns=["text_for_len", "node_type_str"], errors="ignore")
    return base.reset_index(drop=True)


def save_jsonl(payloads: List[Dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for obj in payloads:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Wrote {len(payloads)} records to {path}")


def save_csv(payloads: List[Dict], path: Path) -> None:
    df = pd.DataFrame(payloads)
    df.to_csv(path, index=False)
    print(f"Wrote CSV mirror to {path}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    df = load_dataframe(input_path)
    df = filter_records(df, include_stories=args.include_stories, min_text_len=args.min_text_length)
    if args.max_records:
        df = df.head(args.max_records)

    payloads = []
    for _, row in df.iterrows():
        record = build_record(row)
        payload = make_llm_payload(record)
        payloads.append(payload)

    output_format = args.output_format

    if output_format in ("jsonl", "both"):
        jsonl_path = Path(args.jsonl_output)
        save_jsonl(payloads, jsonl_path)
    else:
        print("Skipping JSONL output (--output-format=csv)")

    if output_format in ("csv", "both") and args.csv_output:
        save_csv(payloads, Path(args.csv_output))
    elif output_format == "csv" and not args.csv_output:
        raise ValueError("--output-format=csv requires --csv-output path")
    elif output_format == "both" and not args.csv_output:
        print("CSV path not provided; skipping CSV output despite --output-format=both")
    else:
        if output_format == "jsonl" and args.csv_output:
            print("Skipping CSV output (--output-format=jsonl)")

    print("\nPipeline complete. Use each JSONL row as the message payload for GPT-4, Claude, LLaMA, etc.")
    print("Each payload already references the taxonomy and includes heuristic aspect hints for active learning.")


if __name__ == "__main__":
    main()
