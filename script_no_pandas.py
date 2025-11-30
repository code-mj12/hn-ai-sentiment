"""Prepare Hacker News AI-related posts for sentiment analysis.

This script loads a CSV that already contains AI-related Hacker News stories
and comments (for example, the CSV produced by `fetch_hn_ai_posts.py`). It
reconstructs story/comment relationships, traverses nested threads, and
emits:
  * `hacker_news_ai_threads.csv` – flattened story/comment rows with lineage
    metadata for per-comment sentiment labeling.
  * `hacker_news_ai_story_blobs.json` – story-level documents that merge each
    story's title/body with all associated comment text for document-level
    sentiment analysis.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
from collections import defaultdict, deque
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import List, Tuple
from urllib import error as urllib_error, request as urllib_request

pd = importlib.import_module("pandas")

HN_ITEM_ENDPOINT = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
TAG_RE = re.compile(r"<[^>]+>")
EXPECTED_COLUMNS = [
    "id",
    "title",
    "text",
    "score",
    "author",
    "time",
    "type",
    "url",
    "parent",
    "descendants",
    "deleted",
    "dead",
    "poll",
    "kids",
    "parts",
]
DEFAULT_RAW_CSV = "hacker_news_ai_posts.csv"
DEFAULT_THREAD_CSV = "hacker_news_ai_threads.csv"
DEFAULT_STORY_JSON = "hacker_news_ai_story_blobs.json"
DEFAULT_STORY_CSV = "hacker_news_ai_story_blobs.csv"
DEFAULT_STORY_KEYWORDS = (
    "ai,artificial intelligence,machine learning,deep learning,llm,gpt,openai,chatgpt,anthropic,claude"
)


def parse_keyword_list(raw: str | None) -> List[str]:
    if not raw:
        return []
    return [token.strip().lower() for token in raw.split(",") if token.strip()]


def story_matches_keywords(text: str, keywords: List[str], match_mode: str) -> bool:
    if not keywords:
        return True
    text_lower = text.lower()
    if match_mode == "all":
        return all(keyword in text_lower for keyword in keywords)
    return any(keyword in text_lower for keyword in keywords)


def select_story_ids(df, keywords: List[str], match_mode: str, max_stories: int | None, sort_by: str) -> List[int]:
    story_df = df[df["type"].fillna("").astype(str).str.lower() == "story"].copy()
    if story_df.empty:
        return []

    combined_text = (
        story_df["title"].fillna("").astype(str)
        + " "
        + story_df["text"].fillna("").astype(str)
    )
    mask = combined_text.apply(lambda text: story_matches_keywords(text, keywords, match_mode))
    filtered = story_df[mask] if keywords else story_df

    if filtered.empty:
        return []

    if sort_by == "time":
        sort_series = pd.to_numeric(filtered["time"], errors="coerce").fillna(0)
    else:
        sort_series = pd.to_numeric(filtered["score"], errors="coerce").fillna(0)

    filtered = filtered.assign(__sort_value=sort_series)
    filtered = filtered.sort_values("__sort_value", ascending=False)

    if max_stories is not None:
        filtered = filtered.head(max_stories)

    return [int(_id) for _id in filtered["id"].dropna().astype(int).tolist()]


def filter_dataframe_to_story_ids(df, story_ids: List[int] | None):
    if not story_ids:
        return df
    story_set = set(story_ids)
    is_story = df["type"].fillna("").astype(str).str.lower() == "story"
    mask = (~is_story) | df["id"].isin(story_set)
    filtered = df[mask].copy()
    return filtered.reset_index(drop=True)


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = unescape(str(value))
    text = TAG_RE.sub(" ", text)
    return " ".join(text.split()).strip()


def format_timestamp(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        ts = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return ""


def fetch_hn_item(item_id: int | None, cache: dict) -> dict | None:
    if not item_id:
        return None
    if item_id in cache:
        return cache[item_id]
    url = HN_ITEM_ENDPOINT.format(item_id=int(item_id))
    try:
        with urllib_request.urlopen(url, timeout=10) as response:
            cache[item_id] = json.load(response)
    except (urllib_error.URLError, ValueError) as exc:
        print(f"Warning: failed to fetch item {item_id}: {exc}")
        cache[item_id] = None
    return cache[item_id]


def resolve_story(item: dict, item_lookup) -> Tuple[dict | None, int, List[int]]:
    if not item:
        return None, 0, []
    if item.get("type") == "story":
        return item, 0, []

    depth = 0
    lineage: List[int] = []
    current = item
    visited = set()

    while current and current.get("type") != "story":
        parent_id = current.get("parent")
        if not parent_id or parent_id in visited:
            return None, depth, lineage
        visited.add(parent_id)
        lineage.append(parent_id)
        parent_item = item_lookup(parent_id)
        if not parent_item:
            return None, depth, lineage
        current = parent_item
        depth += 1

    return current if current and current.get("type") == "story" else None, depth, lineage


def normalize_story(story_item: dict, source_label: str) -> dict:
    return {
        "story_id": story_item.get("id"),
        "title": clean_text(story_item.get("title")),
        "story_text_clean": clean_text(story_item.get("text")),
        "url": story_item.get("url", ""),
        "author": story_item.get("author") or story_item.get("by") or "",
        "score": story_item.get("score", 0) or 0,
        "time_iso": format_timestamp(story_item.get("time")),
        "descendants": story_item.get("descendants", ""),
        "source": source_label,
    }


def ensure_columns(df):
    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df


def load_raw_posts(csv_path: str | Path):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    print(f"Loading raw posts from {path}...")
    df = pd.read_csv(path)
    df = ensure_columns(df)
    print(f"Loaded {len(df)} rows from {path}")
    return df


def augment_with_comments(df, max_depth: int | None, max_comments: int | None, allowed_story_ids: List[int] | None = None):
    story_df = df[df["type"].fillna("").astype(str).str.lower() == "story"]
    if allowed_story_ids:
        story_set = set(int(sid) for sid in allowed_story_ids)
        story_df = story_df[story_df["id"].isin(story_set)]
    if story_df.empty:
        print("No stories found; skipping comment fetch.")
        return df

    existing_ids = set(df["id"].dropna().astype(int))
    new_records = []
    cache = {}

    for _, story in story_df.iterrows():
        story_id = story.get("id")
        if pd.isna(story_id):
            continue
        story_id = int(story_id)
        story_data = fetch_hn_item(story_id, cache) or {}
        initial_kids = story_data.get("kids") or []
        queue = deque((kid, 1) for kid in initial_kids)
        fetched = 0
        visited = set()

        while queue:
            comment_id, depth = queue.popleft()
            if comment_id in visited:
                continue
            visited.add(comment_id)
            comment = fetch_hn_item(comment_id, cache)
            if not comment or comment.get("type") != "comment":
                continue
            if comment_id in existing_ids:
                continue

            new_records.append({
                "id": comment.get("id"),
                "title": comment.get("title"),
                "text": comment.get("text"),
                "score": comment.get("score"),
                "author": comment.get("by"),
                "time": comment.get("time"),
                "type": comment.get("type"),
                "url": comment.get("url"),
                "parent": comment.get("parent"),
                "descendants": comment.get("descendants"),
                "deleted": comment.get("deleted"),
                "dead": comment.get("dead"),
                "poll": comment.get("poll"),
                "kids": comment.get("kids"),
                "parts": comment.get("parts"),
            })

            existing_ids.add(comment_id)
            fetched += 1
            if max_comments and fetched >= max_comments:
                break

            if max_depth is None or depth < max_depth:
                for kid in comment.get("kids") or []:
                    queue.append((kid, depth + 1))

        print(f"Fetched {fetched} comments for story {story_id}")

    if new_records:
        print(f"Appending {len(new_records)} fetched comments to dataset")
        df = pd.concat([df, pd.DataFrame(new_records)], ignore_index=True)
    else:
        print("No new comments fetched.")
    return ensure_columns(df)


def prepare_sentiment_rows(raw_df, allowed_story_ids: List[int] | None = None):
    if raw_df is None or raw_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    allowed_set = set(int(sid) for sid in allowed_story_ids) if allowed_story_ids else None

    records = raw_df.to_dict("records")
    item_index = {item.get("id"): item for item in records if item.get("id") is not None}
    fetched_cache: dict = {}

    def get_item(item_id):
        if item_id in item_index:
            return item_index[item_id]
        return fetch_hn_item(item_id, fetched_cache)

    thread_rows = []
    story_comment_map = defaultdict(list)
    story_metadata = {}

    for item in records:
        item_id = item.get("id")
        if item_id is None:
            continue

        story_item, depth, lineage = resolve_story(item, get_item)
        story_id = story_item.get("id") if story_item else None

        if allowed_set is not None and (story_id is None or story_id not in allowed_set):
            continue

        if story_item and story_id not in story_metadata:
            story_metadata[story_id] = normalize_story(
                story_item,
                source_label="query" if story_id in item_index else "api",
            )

        node_type = item.get("type")
        if node_type == "story":
            text_payload = " ".join(
                filter(None, [clean_text(item.get("title")), clean_text(item.get("text"))])
            )
        else:
            text_payload = clean_text(item.get("text"))

        row = {
            "root_story_id": story_id,
            "root_story_title": story_metadata.get(story_id, {}).get("title", "") if story_id else "",
            "root_story_author": story_metadata.get(story_id, {}).get("author", "") if story_id else "",
            "root_story_time_iso": story_metadata.get(story_id, {}).get("time_iso", "") if story_id else "",
            "root_story_score": story_metadata.get(story_id, {}).get("score", "") if story_id else "",
            "root_story_url": story_metadata.get(story_id, {}).get("url", "") if story_id else "",
            "node_id": item_id,
            "node_type": node_type,
            "node_author": item.get("author") or item.get("by") or "",
            "node_time_iso": format_timestamp(item.get("time")),
            "node_score": item.get("score", 0) or 0,
            "parent_id": item.get("parent"),
            "comment_depth": depth if node_type != "story" else 0,
            "lineage_to_story": " -> ".join(
                str(x) for x in (list(reversed(lineage)) + [item_id])
            ) if lineage else str(item_id),
            "text_clean": text_payload,
        }

        thread_rows.append(row)

        if story_id and node_type == "comment" and text_payload:
            story_comment_map[story_id].append(text_payload)

    story_summaries = []
    for story_id, metadata in story_metadata.items():
        comments = story_comment_map.get(story_id, [])
        comments_blob = " ".join(comments)
        story_summaries.append(
            {
                **metadata,
                "comment_count_in_sample": len(comments),
                "comment_text_blob": comments_blob,
                "story_plus_comments_text": " ".join(
                    filter(None, [metadata["title"], metadata["story_text_clean"], comments_blob])
                ).strip(),
            }
        )

    thread_df = pd.DataFrame(thread_rows)
    if not thread_df.empty:
        thread_df = thread_df.sort_values(
            by=["root_story_id", "comment_depth", "node_time_iso"],
            na_position="last",
        ).reset_index(drop=True)

    story_df = pd.DataFrame(story_summaries)
    if not story_df.empty:
        story_df = story_df.sort_values("score", ascending=False, na_position="last").reset_index(drop=True)

    return thread_df, story_df


def analyze_data(df, thread_df, story_df) -> None:
    if df is None or df.empty:
        print("No data to analyze")
        return

    print("\n=== Raw Data Overview ===")
    print(f"Total posts: {len(df)}")

    if "type" in df.columns:
        print("\nPosts by type:")
        type_counts = df["type"].fillna("unknown").value_counts()
        for type_name, count in type_counts.items():
            print(f"  {type_name}: {count}")

    if "score" in df.columns:
        scores = pd.to_numeric(df["score"], errors="coerce").dropna()
        if not scores.empty:
            print(f"\nAverage score: {scores.mean():.2f}")
            print(f"Highest score: {scores.max():.0f}")

    if "title" in df.columns:
        print("\nSample AI-related titles:")
        for title in df["title"].dropna().head(10):
            print(f"  - {title}")

    if thread_df is not None and not thread_df.empty:
        print(f"\nThread-aware rows prepared: {len(thread_df)}")
        for _, row in thread_df.head(5).iterrows():
            print(
                f"  Story {row.get('root_story_id')} <- {row.get('node_type')} {row.get('node_id')} (depth {row.get('comment_depth')})"
            )

    if story_df is not None and not story_df.empty:
        print(f"\nStory-level sentiment blobs: {len(story_df)}")
        for _, summary in story_df.head(3).iterrows():
            print(
                f"  Story {summary.get('story_id')} with {summary.get('comment_count_in_sample')} comments"
            )


def save_dataframe_to_csv(df, path: str | Path) -> str | None:
    if df is None or df.empty:
        return None
    target_path = Path(path)
    try:
        df.to_csv(target_path, index=False)
    except PermissionError as exc:
        fallback = target_path.with_name(f"{target_path.stem}_new{target_path.suffix}")
        print(
            f"Permission denied when writing {target_path}: {exc}. Saving to {fallback} instead."
        )
        df.to_csv(fallback, index=False)
        target_path = fallback
    print(f"Saved DataFrame to {target_path}")
    return str(target_path)


def save_dataframe_to_json(df, path: str | Path) -> str | None:
    if df is None or df.empty:
        return None
    target_path = Path(path)
    try:
        df.to_json(target_path, orient="records", indent=2, force_ascii=False)
    except PermissionError as exc:
        fallback = target_path.with_name(f"{target_path.stem}_new{target_path.suffix}")
        print(
            f"Permission denied when writing {target_path}: {exc}. Saving to {fallback} instead."
        )
        df.to_json(fallback, orient="records", indent=2, force_ascii=False)
        target_path = fallback
    print(f"Saved JSON to {target_path}")
    return str(target_path)


def save_story_summaries_to_json(df, path: str | Path) -> str | None:
    if df is None or df.empty:
        return None
    records = df.to_dict(orient="records")
    target_path = Path(path)
    try:
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except PermissionError as exc:
        fallback = target_path.with_name(f"{target_path.stem}_new{target_path.suffix}")
        print(
            f"Permission denied when writing {target_path}: {exc}. Saving to {fallback} instead."
        )
        with open(fallback, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        target_path = fallback
    print(f"Story sentiment blobs saved to {target_path}")
    return str(target_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prep Hacker News AI posts for sentiment analysis")
    parser.add_argument(
        "--input",
        default=DEFAULT_RAW_CSV,
        help="Path to the raw CSV exported from BigQuery",
    )
    parser.add_argument(
        "--threads-output",
        default=DEFAULT_THREAD_CSV,
        help="CSV path for the flattened thread dataset",
    )
    parser.add_argument(
        "--stories-output",
        default=DEFAULT_STORY_JSON,
        help="JSON path for story-level sentiment blobs",
    )
    parser.add_argument(
        "--stories-csv-output",
        default=DEFAULT_STORY_CSV,
        help="CSV path for story-level sentiment blobs",
    )
    parser.add_argument(
        "--raw-csv-output",
        default=None,
        help="Optional path to re-save the cleaned raw DataFrame",
    )
    parser.add_argument(
        "--raw-json-output",
        default=None,
        help="Optional path to persist the raw DataFrame as JSON",
    )
    parser.add_argument(
        "--fetch-comments",
        action="store_true",
        help="Fetch comment threads for each story via the Hacker News API",
    )
    parser.add_argument(
        "--max-comment-depth",
        type=int,
        default=6,
        help="Depth limit when fetching comments (default: 6)",
    )
    parser.add_argument(
        "--max-comments-per-story",
        type=int,
        default=500,
        help="Maximum comments to fetch per story (default: 500)",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Limit processing to the top N stories after keyword filtering",
    )
    parser.add_argument(
        "--story-keywords",
        default=DEFAULT_STORY_KEYWORDS,
        help=(
            "Comma-separated keywords that a story must mention (title or text) to be included."
            " Set to an empty string to disable this filter."
        ),
    )
    parser.add_argument(
        "--story-keyword-match",
        choices=("any", "all"),
        default="any",
        help="Require a story to match any vs. all supplied keywords",
    )
    parser.add_argument(
        "--story-sort-by",
        choices=("score", "time"),
        default="score",
        help="Sort criterion when selecting the top stories",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df_raw = load_raw_posts(args.input)
    story_keywords = parse_keyword_list(args.story_keywords)
    apply_story_filter = bool(story_keywords) or args.max_stories is not None
    selected_story_ids: List[int] | None = None

    if apply_story_filter:
        selected_story_ids = select_story_ids(
            df_raw,
            story_keywords,
            match_mode=args.story_keyword_match,
            max_stories=args.max_stories,
            sort_by=args.story_sort_by,
        )

        if not selected_story_ids:
            print("No stories matched the provided keyword/limit criteria. Exiting.")
            return

        print(
            f"Selected {len(selected_story_ids)} stories for processing (examples: {selected_story_ids[:5]})"
        )
        df_raw = filter_dataframe_to_story_ids(df_raw, selected_story_ids)

    if args.fetch_comments:
        df_raw = augment_with_comments(
            df_raw,
            max_depth=args.max_comment_depth,
            max_comments=args.max_comments_per_story,
            allowed_story_ids=selected_story_ids,
        )
    thread_df, story_df = prepare_sentiment_rows(df_raw, allowed_story_ids=selected_story_ids)
    analyze_data(df_raw, thread_df, story_df)

    outputs = []
    outputs.append(save_dataframe_to_csv(thread_df, args.threads_output))
    outputs.append(save_story_summaries_to_json(story_df, args.stories_output))
    outputs.append(save_dataframe_to_csv(story_df, args.stories_csv_output))

    if args.raw_csv_output:
        outputs.append(save_dataframe_to_csv(df_raw, args.raw_csv_output))
    if args.raw_json_output:
        outputs.append(save_dataframe_to_json(df_raw, args.raw_json_output))

    outputs = [path for path in outputs if path]
    if outputs:
        print("\nNext steps:")
        for path in outputs:
            print(f"  - {path}")
        print("Use the thread CSV for per-comment sentiment or the story JSON for document-level analysis.")


if __name__ == "__main__":
    main()
