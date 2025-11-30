# Hacker News AI Sentiment Workflow

This project builds a compact, reproducible pipeline that:

1. Pulls a BigQuery export of AI-related Hacker News stories (`hacker_news_ai_posts.csv`).
2. Rehydrates each selected story with capped Hacker News comments.
3. Prepares model-ready JSONL payloads with sentiment/aspect prompts.
4. Calls OpenRouter in reasoning mode, one story at a time, while preserving batches.
5. Visualizes the resulting sentiments/aspects with matplotlib charts.

The current default run focuses on **10 stories max** (see `--max-stories`) with up to **50 comments per story**, keeping experiments inexpensive.

---

## Step-by-step workflow

### 0. (Optional) Refresh the base CSV
Use `fetch_hn_ai_posts.py` or rerun the BigQuery export to regenerate `hacker_news_ai_posts.csv` when you want newer stories.

### 1. Curate stories + fetch comments (`script_no_pandas.py`)
```bash
python script_no_pandas.py \
  --input hacker_news_ai_posts.csv \
  --fetch-comments --max-comments-per-story 50 --max-comment-depth 4 \
  --max-stories 10 \
  --threads-output hacker_news_ai_threads_new.csv \
  --stories-output hacker_news_ai_story_blobs.json \
  --stories-csv-output hacker_news_ai_story_blobs.csv
```
**Outputs**
- `hacker_news_ai_threads_new.csv`: flattened story/comment rows (one row per node).
- `hacker_news_ai_story_blobs.(json|csv)`: story summaries with `comment_count_in_sample`.

### 2. Build LLM payloads (`sentiment_preprocess.py`)
```bash
python sentiment_preprocess.py \
  --input hacker_news_ai_threads_new.csv \
  --jsonl-output sentiment_llm_payload.jsonl \
  --csv-output sentiment_llm_payload.csv \
  --max-records 500   # shrink further if desired
```
**Outputs**
- `sentiment_llm_payload.jsonl` (primary source for LLM calls).
- `sentiment_llm_payload.csv` (audit/debug mirror).

### 3. Run OpenRouter with per-story batching (`openrouter_sentiment_runner.py`)
```bash
python openrouter_sentiment_runner.py \
  --input sentiment_llm_payload.jsonl \
  --output sentiment_llm_results.jsonl \
  --per-story-batch-size 50 \
  --append-output \
  --sleep 1.0 \
  --limit 0          # process every payload for the selected stories
```
Key features:
- Stories are processed sequentially, and you get complete batches per root story.
- `--append-output` lets you accumulate multiple runs into the same JSONL file.
- Use `--story-ids` or `--max-stories` to explicitly choose which stories to score.

### 4. Visualize results (`sentiment_charts.py`)
```bash
python sentiment_charts.py \
  --input sentiment_llm_results.jsonl \
  --output-dir charts \
  --top-aspects 10
```
**Outputs** (PNG charts)
- `charts/sentiment_distribution.png`
- `charts/top_aspects.png`

---

## Data + artifact index
| File | Format | What it represents |
| --- | --- | --- |
| `hacker_news_ai_posts.csv` | CSV | Base BigQuery export of AI-tagged stories with ids, titles, scores, authors, timestamps, and url metadata. |
| `hacker_news_ai_posts.json` | JSON | Same story subset as above but stored row-per-line for ad-hoc inspection or rehydration utilities. |
| `hacker_news_ai_threads.csv` | CSV | Legacy flat thread dump from earlier runs; kept only for historical comparison/testing. |
| `hacker_news_ai_threads_new.csv` | CSV | Current canonical flattened tree where each row is a story/comment node enriched with lineage fields, depth, and cleaned text. |
| `hacker_news_ai_story_blobs.json` | JSON | Story-level summaries (one JSON object per story) that include metadata plus `comment_count_in_sample` and lightweight comment excerpts. |
| `hacker_news_ai_story_blobs.csv` | CSV | Tabular view of the same summaries for spreadsheet analysis or filters (e.g., require ≥10 sampled comments). |
| `sentiment_llm_payload.jsonl` | JSONL | Primary model-ready payloads generated from the thread CSV; each line carries prompt text, aspect hints, and identifiers. |
| `sentiment_llm_payload.csv` | CSV | Audit-friendly mirror of the payload data so you can sort/filter without a JSONL viewer. |
| `sentiment_llm_results.jsonl` | JSONL | Raw OpenRouter outputs capturing both initial and "are you sure" answers plus reasoning tokens per payload. |
| `charts/sentiment_distribution.png` | PNG | Visualization of the share of positive/neutral/negative sentiments across all processed payloads. |
| `charts/top_aspects.png` | PNG | Bar chart of the most commonly detected aspects/topics with their sentiment lean. |

---

## Tips & maintenance
- Keep `key` (OpenRouter API token) out of version control.
- When re-running step 3, use `--append-output` and consider `--story-ids` to avoid re-charging already-processed stories.
- If a Hacker News API call fails mid-fetch, simply rerun step 1; the script will continue fetching remaining stories.
- For debugging or quick tests, reduce `--max-records` (step 2) or `--per-story-batch-size` (step 3).
- After each full pass, re-run `sentiment_charts.py` to update the visual dashboards.
