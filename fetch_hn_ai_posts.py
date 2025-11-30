"""Fetch AI-related Hacker News items from BigQuery and export to disk.

This script keeps the BigQuery interaction separate from the downstream
sentiment preprocessing pipeline. Run it whenever you want to refresh the raw
Hacker News dataset; then run `script_no_pandas.py` (the prep script) to build
thread-aware sentiment inputs from the CSV this script produces.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

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
        dead,
        poll,
        kids,
        parts
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = fetch_posts(args.project_id, args.limit)
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
