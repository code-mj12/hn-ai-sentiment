# Hacker News AI Sentiment Workflow

This repository contains a reproducible workflow for analyzing how the Hacker News community reacts to AI-related posts:

1. Query BigQuery for AI-tagged stories/comments.
2. Rehydrate and clean the raw threads.
3. Build per-comment JSONL payloads with consistent prompts.
4. Call OpenRouter (or any OpenAI-compatible API) for aspect-aware sentiment labels.
5. Chart the resulting sentiments/aspects.

CSV artifacts are now **intentionally untracked** to keep the repo lean—run the steps below to regenerate them locally whenever you need fresh data.

---

## Requirements

| Tool | Why it is needed |
| --- | --- |
| Python 3.10+ | All scripts are CLI utilities written for modern Python. |
| `pip install pandas requests matplotlib google-cloud-bigquery` | Core libraries across the pipeline (pandas for preprocessing, requests for OpenRouter calls, matplotlib for charts, BigQuery client for the optional fetch step). |
| Google Cloud credentials | Required only when running `fetch_hn_ai_posts.py` (BigQuery access). |
| OpenRouter API key | Save your token in a file named `key` (default path used by the runner + probes). |

Feel free to create a virtual environment and install the dependencies in one go:

```bash
python -m venv .venv
. .venv/Scripts/activate        # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install pandas requests matplotlib google-cloud-bigquery
```

---

## End-to-end workflow

Each script lives at the repo root. All commands below assume you are inside `AI_lab_hacker_news/`.

### 1. (Optional) Refresh the raw export – `fetch_hn_ai_posts.py`

Pull a fresh sample of AI discussions from the public Hacker News BigQuery dataset:

```bash
python fetch_hn_ai_posts.py \
  --project ai-lab-479123 \
  --limit 1000 \
  --csv-output hacker_news_ai_posts.csv \
  --json-output hacker_news_ai_posts.json \
  --comment-story-limit 10 \
  --comment-depth 6 \
  --comments-per-story 500
```

This step produces **two files** (CSV + JSON). They are gitignored; keep them locally or upload to cloud storage if you want an archive.

### 2. Build thread-aware rows – `script_no_pandas.py`

```bash
python script_no_pandas.py \
  --input hacker_news_ai_posts.csv \
  --fetch-comments \
  --max-comments-per-story 50 \
  --max-comment-depth 4 \
  --max-stories 10 \
  --threads-output hacker_news_ai_threads.csv \
  --stories-output hacker_news_ai_story_blobs.json
```

Outputs:

- `hacker_news_ai_threads.csv` – every story/comment node with lineage, cleaned text, and parent metadata.
- `hacker_news_ai_story_blobs.json` – per-story metadata + concatenated comment blobs for document-level work.

### 3. Build LLM payloads – `sentiment_preprocess.py`

```bash
python sentiment_preprocess.py \
  --input hacker_news_ai_threads.csv \
  --jsonl-output sentiment_llm_payload.jsonl \
  --output-format jsonl \
  --max-records 500
```

Use `--output-format csv` or `--output-format both` if you truly need a CSV mirror. Default is JSONL-only to avoid duplicate artifacts.

### 4. Smoke-test a single payload – `single_llm_probe.py`

```bash
python single_llm_probe.py \
  --payload-index 0 \
  --model openrouter/auto \
  --input sentiment_llm_payload.jsonl \
  --api-key-path key
```

This prints the OpenRouter JSON reply plus the elapsed time so you can verify prompts or latency before batch runs.

### 5. Batch scoring with story chunks – `openrouter_sentiment_runner.py`

```bash
python openrouter_sentiment_runner.py \
  --input sentiment_llm_payload.jsonl \
  --output sentiment_llm_results.jsonl \
  --model openrouter/auto \
  --per-story-batch-size 50 \
  --append-output \
  --sleep 1.0 \
  --story-ids 40345775 \
  --limit 0
```

Tips:

- `--story-ids` lets you re-run or debug specific threads without re-scoring everything.
- `--append-output` keeps existing JSONL rows and appends new results.
- Each API call bundles every payload for the story (or chunk) to save on request count while still saving per-comment outputs.

### 6. Visualize results – `sentiment_charts.py`

```bash
python sentiment_charts.py \
  --input sentiment_llm_results.jsonl \
  --output-dir charts \
  --top-aspects 10
```

Generates `charts/sentiment_distribution.png` and `charts/top_aspects.png` (folder auto-created).

---

## Generated artifacts (gitignored)

| Path | Description |
| --- | --- |
| `hacker_news_ai_posts.(csv|json)` | Raw BigQuery export and newline-delimited JSON mirror. |
| `hacker_news_ai_threads.csv` | Flattened story/comment rows used for downstream processing. |
| `hacker_news_ai_story_blobs.json` | Story-level rollups (title + combined comments). |
| `sentiment_llm_payload.jsonl` | Primary payload file consumed by both the runner and probe. |
| `sentiment_llm_results.jsonl` | OpenRouter responses (per-comment JSON). |
| `charts/*.png` | Visual summaries created by `sentiment_charts.py`. |

Delete and regenerate these at any time; the scripts are deterministic when fed the same upstream data.

---

## Operational notes

- Store your OpenRouter key in `key` (or pass `--api-key-path`). Never commit it.
- If a fetch step aborts mid-run, simply re-run the script; deduplication is handled for you.
- Use `--max-records` in `sentiment_preprocess.py` and `--story-ids`/`--limit` in the runner when experimenting.
- The repo no longer ships CSV samples—this keeps Git history small and encourages fetching the freshest data before each analysis.

---

## Publishing to GitHub

After regenerating artifacts locally and verifying the steps above, commit the code changes (not the generated CSV/JSONL files) and push:

```bash
git status
git add README.md sentiment_preprocess.py
git commit -m "Document workflow and thin CSV artifacts"
git push origin main
```

Ensure your gitignore covers data outputs if you decide to change filenames.
