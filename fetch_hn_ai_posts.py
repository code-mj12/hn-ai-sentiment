"""Fetch AI-related Hacker News items from BigQuery and export to disk.

This script keeps the BigQuery interaction separate from the downstream
sentiment preprocessing pipeline. Run it whenever you want to refresh the raw
Hacker News dataset; then run `script_no_pandas.py` (the prep script) to build
thread-aware sentiment inputs from the CSV this script produces.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import deque
from pathlib import Path
from urllib import error as urllib_error, request as urllib_request

from google.cloud import bigquery

pd = importlib.import_module("pandas")

KEYWORDS = [
    " ai ",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "llm",
    "gpt",
    "openai",
    "chatgpt",
    "claude",
    "anthropic",
]

DEFAULT_LIMIT = 1000
DEFAULT_PROJECT = "ai-lab-479123"
DEFAULT_CSV = "hacker_news_ai_posts.csv"
DEFAULT_JSON = "hacker_news_ai_posts.json"
DEFAULT_COMMENT_STORY_LIMIT = 10
DEFAULT_COMMENT_DEPTH = 6
DEFAULT_COMMENTS_PER_STORY = 500
HN_ITEM_ENDPOINT = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
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


def build_keyword_filter(field: str) -> str:
    clauses = [f"LOWER({field}) LIKE '%{kw}%'" for kw in KEYWORDS]
    return " OR\n          ".join(clauses)


def build_query(limit: int) -> str:
    title_filter = build_keyword_filter("title")
    text_filter = build_keyword_filter("text")
    return f"""
    WITH ai_items AS (
      SELECT
        id,
        title,
        text,
        score,
        `by` AS author,
        time,
        type,
        url,
        parent,
        descendants,
        deleted,
                dead
      FROM `bigquery-public-data.hacker_news.full`
      WHERE
        type IN ('story', 'comment')
        AND (
          {title_filter}
          OR
          {text_filter}
        )
        AND time >= UNIX_SECONDS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 730 DAY))
    )
    SELECT * FROM ai_items
    ORDER BY score DESC
    LIMIT {limit}
    """


def ensure_columns(df):
    for column in EXPECTED_COLUMNS:
        if column not in df.columns:
            df[column] = None
    return df


def fetch_hn_item(item_id: int | None, cache: dict) -> dict | None:
    if not item_id:
        return None
        return cache[item_id]
    url = HN_ITEM_ENDPOINT.format(item_id=int(item_id))
    try:
        with urllib_request.urlopen(url, timeout=10) as response:
            cache[item_id] = json.load(response)
    except (urllib_error.URLError, ValueError) as exc:
        print(f"Warning: failed to fetch item {item_id}: {exc}")
        cache[item_id] = None
    return cache[item_id]


def augment_with_comments(
    df,
    story_limit: int | None,
    max_depth: int | None,
    max_comments_per_story: int | None,
):
    story_df = df[df["type"].fillna("").astype(str).str.lower() == "story"].copy()
    if story_df.empty:
        print("No stories found; skipping comment fetch.")
        return df

    score_series = pd.to_numeric(story_df["score"], errors="coerce").fillna(0)
    story_df = story_df.assign(__score=score_series).sort_values("__score", ascending=False)
    if story_limit is not None and story_limit > 0:
        story_df = story_df.head(story_limit)

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

            new_records.append(
                {
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
                }
            )

            existing_ids.add(comment_id)
            fetched += 1
            if max_comments_per_story and fetched >= max_comments_per_story:
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


def fetch_posts(project_id: str, limit: int):
    print(f"Running BigQuery job (project={project_id}, limit={limit})...")
    client = bigquery.Client(project=project_id)
    query = build_query(limit)
    job = client.query(query)
    df = job.result().to_dataframe()

    df = ensure_columns(df)
    print(f"Retrieved {len(df)} rows from BigQuery")
    return df


def save_dataframe_to_csv(df, path: Path) -> str:
    df.to_csv(path, index=False)
    print(f"Saved CSV to {path}")
    return str(path)


def save_dataframe_to_json(df, path: Path) -> str:
    df.to_json(path, orient="records", indent=2, force_ascii=False)
    print(f"Saved JSON to {path}")
    return str(path)


def analyze_df(df) -> None:
    if df.empty:
        print("No rows returned.")
        return

    print("\n=== Query Summary ===")
    print(f"Total rows: {len(df)}")
    if "type" in df.columns:
        counts = df["type"].fillna("unknown").value_counts()
        for type_name, count in counts.items():
            print(f"  {type_name}: {count}")
    if "score" in df.columns:
        scores = pd.to_numeric(df["score"], errors="coerce").dropna()
        if not scores.empty:
            print(f"Average score: {scores.mean():.2f}")
            print(f"Top score: {scores.max():.0f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch AI-related Hacker News posts from BigQuery")
    parser.add_argument("--project-id", default=DEFAULT_PROJECT, help="Google Cloud project ID to bill the query")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Maximum number of rows to return")
    parser.add_argument("--csv-output", default=DEFAULT_CSV, help="Path to write the raw CSV")
    parser.add_argument("--json-output", default=DEFAULT_JSON, help="Path to write the raw JSON")
    parser.add_argument("--skip-json", action="store_true", help="If set, skip writing the JSON file")
    parser.add_argument(
        "--fetch-comments",
        action="store_true",
        help="Fetch live comment threads from the Hacker News API before writing outputs",
    )
    parser.add_argument(
        "--comment-story-limit",
        type=int,
        default=DEFAULT_COMMENT_STORY_LIMIT,
        help="Only fetch comments for the top N stories by score (default: 10)",
    )
    parser.add_argument(
        "--max-comment-depth",
        type=int,
        default=DEFAULT_COMMENT_DEPTH,
        help="Depth limit when fetching live comment trees",
    )
    parser.add_argument(
        "--max-comments-per-story",
        type=int,
        default=DEFAULT_COMMENTS_PER_STORY,
        help="Maximum number of comments to fetch per story",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = fetch_posts(args.project_id, args.limit)
    if args.fetch_comments:
        df = augment_with_comments(
            df,
            story_limit=args.comment_story_limit,
            max_depth=args.max_comment_depth,
            max_comments_per_story=args.max_comments_per_story,
        )
    analyze_df(df)

    outputs = [save_dataframe_to_csv(df, Path(args.csv_output))]
    if not args.skip_json:
        outputs.append(save_dataframe_to_json(df, Path(args.json_output)))

    print("\nNext steps:")
    print("  - Use script_no_pandas.py to convert the CSV into sentiment-ready formats.")
    for out in outputs:
        print(f"  - Generated: {out}")


if __name__ == "__main__":
    main()
