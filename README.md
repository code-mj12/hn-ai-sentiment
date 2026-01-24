# Hacker News AI Sentiment Analysis (Two-Stage Pipeline, v2)

This repo implements a two-stage, batched LLM workflow for Hacker News AI threads:

**Stage 1: Aspect Detection (batched, 10 per call)**
- Detect which aspects appear and capture evidence + confidence.
- Strict JSON array: one response per payload, no extras, no drops.

**Stage 2: Aspect Sentiment (batched, 10 per call)**
- For detected aspects, classify sentiment (positive/neutral/negative/mixed) + score + reasoning.
- Also strict JSON array, aligned to the batch ordering.

**Why two stages?**
- Reduces hallucinations (detect then classify)
- Skips sentiment when no aspects found (cheaper/faster)
- Evidence gives traceability

Data artifacts (CSV shards, payloads, results, charts) are **gitignored** by default.

---

## Requirements

| Tool | Why it is needed |
| --- | --- |
| Python 3.10+ | All scripts are CLI utilities written for modern Python. |
| `pip install pandas requests matplotlib` | CSV processing, API calls, charting (optional). |
| InnKube inference key | Save your token in a file named `key` (default path used by the inference runner). |
| HN data CSV shards | Place `hn_stories_*.csv` files in `data/` folder (from HN data dump). |

Feel free to use the base environment or create a virtual environment:

```bash
# Optional: Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install pandas requests matplotlib
```

---

## End-to-end workflow

All scripts live at the repo root. Commands assume you are inside `AI_lab_hacker_news/` with CSV shards in `data/` folder.

---

## Architecture (v2 Files)

| File | Purpose |
| --- | --- |
| **fetch_and_filter_ai_posts_v2.py** | Merge CSV shards + filter for AI keywords (writes `hn_ai_filtered_v2.csv`) |
| **prepare_sentiment_payloads_v2.py** | Generate Stage 1 payloads for v2 (`sentiment_payloads_v2.jsonl`) with 6-aspect taxonomy |
| **run_sentiment_analysis_v2.py** | Run complete two-stage pipeline (batched 10 per call) and merge final results |
| **sentiment_charts.py** | Generate charts from final results (optional; outputs are gitignored) |

---

## Quick Start (v2, batched)

```bash
# 1) Filter AI posts
python fetch_and_filter_ai_posts_v2.py --input-dir data --output hn_ai_filtered_v2.csv

# 2) Generate Stage 1 payloads (6 aspects)
python prepare_sentiment_payloads_v2.py --input hn_ai_filtered_v2.csv --output sentiment_payloads_v2.jsonl

# 3) Run complete two-stage pipeline (batched 10 per LLM call)
python run_sentiment_analysis_v2.py \
  --input sentiment_payloads_v2.jsonl \
  --output final_sentiment_results_v2.jsonl \
  --model qwen3-next-80b-a3b-instruct \
  --batch-size 10 \
  --min-confidence 0.7 \
  --timeout 120
```

The unified runner handles Stage 1, Stage 2 prep, Stage 2, and merge—now in **single batched requests** per stage (default batch size 10, no concurrency needed).

---

## Detailed Workflow

### Step 1: Filter AI Posts – `fetch_and_filter_ai_posts.py`

Merge CSV shards and extract AI-related discussions:

```bash
python fetch_and_filter_ai_posts.py \
  --input-dir data \
  --output hn_ai_filtered.csv \
  --max-rows 0
```

**Options:**
- `--input-dir`: Directory with `hn_stories_*.csv` shards
- `--output`: Output CSV path
- `--max-rows`: Limit rows (0 = unlimited)

**Output**: CSV with columns: comment_id, story_id, story_title, story_by, comment_time, comment_text, comment_by

---

### Step 2: Prepare Payloads – `prepare_sentiment_payloads_v2.py`

#### Mode 1: Generate Stage 1 payloads (default)

```bash
python prepare_sentiment_payloads_v2.py \
  --input hn_ai_filtered_v2.csv \
  --output sentiment_payloads_v2.jsonl \
  --max-payloads 0
```

**Output**: JSONL with `stage1_messages` field for aspect detection

#### Mode 2: Prepare Stage 2 payloads (optional, if running stages separately)

```bash
python prepare_sentiment_payloads.py \
  --stage2-results stage1_aspect_detection.jsonl \
  --stage2-output stage2_payloads.jsonl \
  --min-confidence 0.7
```

This mode filters Stage 1 results by confidence and generates `stage2_messages`.

---

### Step 3: Run Complete Pipeline – `run_sentiment_analysis_v2.py`

Execute the full two-stage pipeline in one command:

```bash
python run_sentiment_analysis_v2.py \
  --input sentiment_payloads_v2.jsonl \
  --output final_sentiment_results_v2.jsonl \
  --model qwen3-next-80b-a3b-instruct \
  --batch-size 10 \
  --min-confidence 0.7 \
  --timeout 120 \
  --max-payloads 0 \
  --sleep 0.0
```

**What this does (batched):**
1. **Stage 1 (batched)**: Loads `stage1_messages`, sends up to `--batch-size` payloads per request, strict JSON array length check.
2. **Stage 2 Prep**: Filters detected aspects by confidence ≥ `--min-confidence`.
3. **Stage 2 (batched)**: Sends up to `--batch-size` payloads per request for sentiment classification, strict JSON array length check.
4. **Merge**: Combines both stages into final output.

**Options:**
- `--input`: Input JSONL with stage1_messages
- `--output`: Final output path
- `--model`: Model to use (default: qwen3-next-80b-a3b-instruct)
- `--min-confidence`: Minimum confidence to qualify for Stage 2 (default: 0.7)
- `--timeout`: Request timeout in seconds (default: 120)
- `--max-payloads`: Max payloads to process (0 = unlimited)
- `--sleep`: Sleep between API calls in seconds
- `--no-resume`: Disable resume mode (normally continues from last position)

**Output**: JSONL with final sentiment results:
```json
{
  "payload_id": "payload_000042",
  "payload_type": "direct_comment",
  "story_id": "12345",
  "detected_aspects": [
    {
      "aspect": "performance_speed",
      "present": true,
      "evidence": "10x faster inference",
      "confidence": 0.95
    }
  ],
  "aspect_sentiments": [
    {
      "aspect": "performance_speed",
      "sentiment": "positive",
      "score": 0.85,
      "confidence": 0.9,
      "reasoning": "Clear positive language about speed improvement"
    }
  ],
  "overall_sentiment": "positive",
  "overall_score": 0.85,
  "stage1_elapsed": 5.2,
  "stage2_elapsed": 3.1,
  "total_elapsed": 8.3,
  "error": null
}
```

**Temporary Files** (automatically cleaned up after pipeline):
- `stage1_temp.jsonl`: Stage 1 raw results
- `stage2_payloads_temp.jsonl`: Stage 2 payloads after filtering
- `stage2_temp.jsonl`: Stage 2 raw results

---

### Optional: Visualization – `sentiment_charts.py`

Generate charts from final results:

```bash
python sentiment_charts.py \
  --input final_sentiment_results.jsonl \
  --output charts/
```

---

## Efficiency & Filtering

**Stage 2 Reduction (Typical Results)**:
- Input payloads: 1000
- Payloads with detected aspects: ~600 (40% filtered)
- Payloads above confidence threshold (0.7): ~300 (50% of qualified)
- **Total Stage 2 calls: ~300 (~70% reduction vs brute force)**

This filtering saves significant API costs while maintaining quality.

---

## Resume & Error Handling

All scripts support resumable execution:

```bash
# First run (processes payloads 1-500)
python run_sentiment_analysis.py --input sentiment_payloads.jsonl --output final_sentiment_results.jsonl --max-payloads 500

# Continue from where it left off (processes payloads 501+)
python run_sentiment_analysis.py --input sentiment_payloads.jsonl --output final_sentiment_results.jsonl
```

Add `--no-resume` to restart from scratch.

---

## Troubleshooting

**Error: API timeout**
```bash
# Increase timeout
python run_sentiment_analysis.py --input sentiment_payloads.jsonl --output final_sentiment_results.jsonl --timeout 180
```

**Error: "No messages found"**
- Verify payloads were generated correctly with `--max-payloads 5` first

**API errors in logs**
- Check your API key is correct in the `key` file
- Verify InnKube endpoint is accessible

---

## Performance Notes

- **Typical latency**: 4-7 seconds per Stage 1 payload, 2-5 seconds per Stage 2 payload
- **Cost-effective**: Stage 2 filtering saves 60-70% of API calls
- **Parallelization**: For larger datasets, split input into chunks and run multiple instances
- **Resume support**: Automatically skips already-processed payloads

---

## Aspect Taxonomy (12 aspects)
````
  --timeout 120 \
  --max-payloads 0
```

**Output**: Aspect-level sentiment classifications

Example:
```json
{
  "payload_id": "payload_000042",
  "stage2_result": {
    "aspect_sentiments": [
      {
        "aspect": "performance_speed",
        "sentiment": "positive",
        "sentiment_score": 0.85,
        "reasoning": "Enthusiastic about 10x speedup"
      }
    ],
    "overall_sentiment": "positive",
    "overall_score": 0.8
  }
}
```

### Step 6: Merge Results – `merge_results.py`

Combine Stage 1 + Stage 2 into final comprehensive output:

```bash
python merge_results.py \
  --stage1 stage1_aspect_detection.jsonl \
  --stage2 stage2_aspect_sentiment.jsonl \
  --output final_sentiment_results.jsonl
```

### Step 7: Visualize Results – `sentiment_charts.py`

```bash
python sentiment_charts.py \
  --input final_sentiment_results.jsonl \
  --output-dir charts \
  --top-aspects 10
```

Creates `charts/sentiment_distribution.png` and `charts/top_aspects.png`.

---

## Generated artifacts (gitignored)

| Path | Description |
| --- | --- |
| `data/hn_stories_*.csv` | Raw HN data dump CSV shards (input, not generated). |
| `hn_ai_filtered.csv` | Merged and filtered AI-related posts from Step 1. |
| `sentiment_payloads.jsonl` | Stage 1 payloads for aspect detection (Step 2). |
| `stage1_aspect_detection.jsonl` | Stage 1 results: detected aspects with evidence (Step 3). |
| `stage2_sentiment_payloads.jsonl` | Stage 2 payloads for sentiment classification (Step 4). |
| `stage2_aspect_sentiment.jsonl` | Stage 2 results: aspect-level sentiments (Step 5). |
| `final_sentiment_results.jsonl` | Merged Stage 1 + Stage 2 comprehensive results (Step 6). |
| `charts/*.png` | Visualization outputs from Step 7. |

All intermediate files are deterministic for the same input data.

---

## Operational notes

- Store InnKube inference key in `key` file (or pass `--api-key-path`). Never commit it.
- Endpoint: `https://llms.innkube.fim.uni-passau.de/v1/chat/completions` (OpenAI-compatible)
- Allowed models: `qwen3-next-80b-a3b-instruct` (default, recommended), `webthinker-qwq-32b`
- Place HN CSV shards (`hn_stories_*.csv`) in `data/` folder before running
- Expected CSV schema: `comment_id`, `story_id`, `story_title`, `story_by`, `comment_time`, `comment_text`, `comment_by`
- **Two-stage efficiency**: Stage 1 typically filters out 40-60% of payloads (no aspects detected), saving API calls
- Use `--min-confidence 0.7` in Step 4 to only process high-confidence aspects
- Both Stage 1 and Stage 2 support resume mode: re-run safely after interruption
- Recommended timeout: 120s (some payloads take 5-15s)
- For testing: use `--max-payloads 100` on Steps 2-5 before full runs
- Monitor progress: `tail -f stage1_aspect_detection.jsonl | wc -l`

## Why two stages?

**Single-stage problems:**
- LLM tries to detect aspects AND classify sentiment simultaneously
- Leads to hallucination (inventing aspects that aren't there)
- Wastes API calls on payloads with no relevant aspects
- Hard to debug: can't tell if error is in detection or classification

**Two-stage benefits:**
- **Accuracy**: LLM focuses on ONE task at a time (detection OR classification)
- **Efficiency**: Skip Stage 2 for payloads with no detected aspects (~40-60% savings)
- **Traceability**: Evidence field shows exact text span where aspect appears
- **Confidence filtering**: Set minimum confidence threshold (e.g., 0.7) to reduce false positives
- **Debuggability**: Can analyze Stage 1 results before running expensive Stage 2

## Project structure

```
AI_lab_hacker_news/
├── fetch_and_filter_ai_posts.py    # Step 1: Merge CSV shards + filter AI posts
├── prepare_sentiment_payloads.py   # Step 2: Generate 3 payload types
├── run_sentiment_analysis.py       # Step 3: Call InnKube for sentiment
├── sentiment_charts.py              # Step 4: Visualize results
├── key                              # InnKube API key (gitignored)
├── data/                            # HN CSV shards (gitignored)
│   ├── hn_stories_000000000000.csv
│   ├── hn_stories_000000000001.csv
│   └── ...
├── old/                             # Archived old scripts (gitignored)
│   ├── fetch_hn_ai_posts.py        # Old BigQuery version
│   ├── script_no_pandas.py         # Old thread builder
│   ├── sentiment_preprocess.py     # Old payload generator
│   ├── openrouter_sentiment_runner.py  # Old batch runner
│   └── single_llm_probe.py         # Old probe tool
└── charts/                          # Visualization outputs (gitignored)
```

---

## Publishing to GitHub

After running the workflow locally and verifying results, commit code changes (not data files) and push:

```bash
git status
git add fetch_and_filter_ai_posts.py prepare_sentiment_payloads.py run_sentiment_analysis.py
git add sentiment_charts.py README.md .gitignore
git commit -m "Refactor to 3-script architecture with CSV shard processing"
git push origin main
```

Data files are gitignored, so only code and documentation will be tracked.
