# Hacker News AI Sentiment Analysis Pipeline (V3)

A single-LLM-call pipeline for analyzing sentiment in Hacker News discussions about AI models.

## Pipeline Overview

```
CSV Data → Filter AI Posts → Generate Payloads → LLM Analysis → Confidence Filter → Charts
```

**V3 Optimization**: Combined aspect detection + sentiment into a single LLM call (50% fewer API calls than the old two-stage approach).

---

## Architecture

| File | Purpose |
|------|---------|
| `fetch_and_filter_ai_posts_v2.py` | Merge CSV shards + filter for AI keywords → `hn_ai_filtered_v2.csv` |
| `prepare_sentiment_payloads_v2.py` | Generate LLM payloads from filtered CSV → `sentiment_payloads_v2.jsonl` |
| `run_sentiment_analysis_v2.py` | Run batched LLM analysis (aspect + sentiment) → `final_sentiment_results_v2.jsonl` |
| `sentiment_charts.py` | Generate visualization charts → `charts_v2/` |
| `pipeline.py` | Orchestrate full pipeline (all steps in one command) |

---

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE FLOW (V3)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DATA PREPARATION                                                        │
│     ┌──────────────────┐      ┌──────────────────┐                         │
│     │ CSV Shards       │ ──── │ Filter AI Posts  │                         │
│     │ (hn_stories_*)   │      │ (keywords match) │                         │
│     └──────────────────┘      └────────┬─────────┘                         │
│                                        │                                    │
│                                        ▼                                    │
│                          ┌──────────────────────────┐                       │
│                          │ hn_ai_filtered_v2.csv    │                       │
│                          │ (stories + comments)     │                       │
│                          └────────────┬─────────────┘                       │
│                                       │                                     │
│  2. PAYLOAD GENERATION                │                                     │
│                                       ▼                                     │
│                          ┌──────────────────────────┐                       │
│                          │ prepare_sentiment_       │                       │
│                          │ payloads_v2.py           │                       │
│                          └────────────┬─────────────┘                       │
│                                       │                                     │
│                    ┌──────────────────┴──────────────────┐                  │
│                    ▼                                     ▼                  │
│          ┌─────────────────┐                   ┌─────────────────┐          │
│          │ direct_comment  │                   │ nested_comment  │          │
│          │ (story context) │                   │ (parent context)│          │
│          └─────────────────┘                   └─────────────────┘          │
│                    │                                     │                  │
│                    └──────────────────┬──────────────────┘                  │
│                                       ▼                                     │
│                          ┌──────────────────────────┐                       │
│                          │ sentiment_payloads_v2.   │                       │
│                          │ jsonl (5,905 payloads)   │                       │
│                          └────────────┬─────────────┘                       │
│                                       │                                     │
│  3. LLM ANALYSIS (Single Call)        │                                     │
│                                       ▼                                     │
│                          ┌──────────────────────────┐                       │
│                          │ run_sentiment_analysis_  │                       │
│                          │ v2.py (batched, 10/req)  │                       │
│                          └────────────┬─────────────┘                       │
│                                       │                                     │
│                                       ▼                                     │
│                    ┌──────────────────────────────────────┐                 │
│                    │         LLM PROMPT (per batch)       │                 │
│                    │  ┌────────────────────────────────┐  │                 │
│                    │  │ For each comment:              │  │                 │
│                    │  │ - Detect 6 aspects             │  │                 │
│                    │  │ - Rate each 1-5                │  │                 │
│                    │  │ - Classify sentiment           │  │                 │
│                    │  │ - Extract evidence             │  │                 │
│                    │  │ - Assign confidence            │  │                 │
│                    │  │ - Identify AI model mentioned  │  │                 │
│                    │  └────────────────────────────────┘  │                 │
│                    └──────────────────┬───────────────────┘                 │
│                                       │                                     │
│                                       ▼                                     │
│                          ┌──────────────────────────┐                       │
│                          │ Confidence Filter (≥0.7) │                       │
│                          └────────────┬─────────────┘                       │
│                                       │                                     │
│                                       ▼                                     │
│                          ┌──────────────────────────┐                       │
│                          │ final_sentiment_results_ │                       │
│                          │ v2.jsonl                 │                       │
│                          └────────────┬─────────────┘                       │
│                                       │                                     │
│  4. VISUALIZATION                     │                                     │
│                                       ▼                                     │
│                          ┌──────────────────────────┐                       │
│                          │ sentiment_charts.py      │                       │
│                          └────────────┬─────────────┘                       │
│                                       │                                     │
│                                       ▼                                     │
│          ┌────────────────────────────────────────────────────┐             │
│          │                    CHARTS                          │             │
│          │  - sentiment_distribution.png                      │             │
│          │  - top_aspects.png                                 │             │
│          │  - aspect_sentiment_mix.png                        │             │
│          │  - confidence_by_sentiment.png                     │             │
│          │  - aspect_confidence_leaderboard.png               │             │
│          │  - aspect_sentiment_by_model.png (AI model ratings)│             │
│          └────────────────────────────────────────────────────┘             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6-Aspect Taxonomy

| Aspect | Description | Examples |
|--------|-------------|----------|
| `coding_speed` | Speed, latency, execution time | "10x faster", "slow inference" |
| `coding_performance` | Quality, correctness, accuracy | "better code quality", "makes mistakes" |
| `cost_price` | Pricing, compute costs, ROI | "expensive API", "free tier" |
| `ethics` | Bias, fairness, transparency, safety | "hallucinations", "alignment issues" |
| `innovation` | Novelty, breakthroughs, SOTA | "revolutionary", "incremental improvement" |
| `skepticism_hype` | Doubt, feasibility concerns, overhype | "overhyped", "not practical yet" |

---

## Requirements

```bash
# Python 3.10+
pip install pandas requests matplotlib numpy

# InnKube API key (save in file named 'key')
echo "your-api-key" > key
```

---

## Quick Start

### Option 1: Full Pipeline (one command)

```bash
python pipeline.py
```

This runs all steps automatically.

### Option 2: Step by Step

```bash
# 1. Filter AI posts from CSV shards
python fetch_and_filter_ai_posts_v2.py --input-dir data --output hn_ai_filtered_v2.csv

# 2. Generate payloads
python prepare_sentiment_payloads_v2.py --input hn_ai_filtered_v2.csv --output sentiment_payloads_v2.jsonl

# 3. Run LLM analysis (use --max-payloads for testing)
python run_sentiment_analysis_v2.py \
  --input sentiment_payloads_v2.jsonl \
  --output final_sentiment_results_v2.jsonl \
  --batch-size 10 \
  --min-confidence 0.7 \
  --max-payloads 300

# 4. Generate charts
python sentiment_charts.py --input final_sentiment_results_v2.jsonl --output-dir charts_v2
```

---

## CLI Reference

### run_sentiment_analysis_v2.py

```bash
python run_sentiment_analysis_v2.py \
  --input sentiment_payloads_v2.jsonl \   # Input payloads
  --output final_sentiment_results_v2.jsonl \  # Final output
  --model qwen3-next-80b-a3b-instruct \   # LLM model
  --batch-size 10 \                        # Payloads per LLM request
  --min-confidence 0.7 \                   # Filter threshold
  --timeout 180 \                          # Request timeout (seconds)
  --max-payloads 300                       # Limit for testing (0=all)
```

### sentiment_charts.py

```bash
python sentiment_charts.py \
  --input final_sentiment_results_v2.jsonl \
  --output-dir charts_v2 \
  --top-aspects 10 \
  --dpi 120
```

---

## Output Format

Each record in `final_sentiment_results_v2.jsonl`:

```json
{
  "payload_id": "payload_000042",
  "payload_type": "direct_comment",
  "story_id": "12345",
  "comment_text": "Claude is much better than GPT-4 for coding...",
  "ai_model": "claude",
  "detected_aspects": [
    {
      "aspect": "coding_performance",
      "present": true,
      "rating": 4,
      "sentiment": "positive",
      "evidence": "much better for coding",
      "confidence": 0.9
    },
    {
      "aspect": "skepticism_hype",
      "present": true,
      "rating": 2,
      "sentiment": "negative",
      "evidence": "still makes mistakes",
      "confidence": 0.85
    }
  ],
  "stage1_elapsed": 8.5,
  "error": null
}
```

**Fields:**
- `ai_model`: Normalized AI model mentioned (gpt-4, claude, gemini, llama, etc.)
- `detected_aspects`: Array of aspects found with rating (1-5), sentiment, evidence
- `confidence`: Filters applied at 0.7 threshold

---

## Generated Charts

| Chart | Description |
|-------|-------------|
| `sentiment_distribution.png` | Overall sentiment breakdown (positive/negative/neutral/mixed) |
| `top_aspects.png` | Most frequently mentioned aspects |
| `aspect_sentiment_mix.png` | Stacked bar: sentiment distribution per aspect |
| `confidence_by_sentiment.png` | Confidence boxplot by sentiment category |
| `aspect_confidence_leaderboard.png` | Average confidence score per aspect |
| `aspect_sentiment_by_model.png` | AI model comparison: sentiment + aspect ratings heatmap |

---

## Project Structure

```
AI_lab_hacker_news/
├── fetch_and_filter_ai_posts_v2.py   # Step 1: Filter CSV shards
├── prepare_sentiment_payloads_v2.py  # Step 2: Generate payloads
├── run_sentiment_analysis_v2.py      # Step 3: LLM analysis (V3 optimized)
├── sentiment_charts.py               # Step 4: Visualization
├── pipeline.py                       # Full pipeline orchestration
├── key                               # InnKube API key (gitignored)
├── data/                             # CSV shards (gitignored)
│   └── hn_stories_*.csv
├── old/                              # Archived old scripts
├── charts_v2/                        # Generated charts (gitignored)
└── *.jsonl                           # Data artifacts (gitignored)
```

---

## Generated Artifacts (gitignored)

| File | Description |
|------|-------------|
| `hn_ai_filtered_v2.csv` | Filtered AI-related posts |
| `sentiment_payloads_v2.jsonl` | LLM input payloads (5,905 records) |
| `final_sentiment_results_v2_stage1.jsonl` | Raw LLM output |
| `final_sentiment_results_v2_filtered.jsonl` | After confidence filter |
| `final_sentiment_results_v2.jsonl` | Final merged results |
| `charts_v2/*.png` | Visualization charts |
| `charts_v2/analysis_summary.json` | Aggregated statistics |

---

## API Configuration

- **Endpoint**: `https://llms.innkube.fim.uni-passau.de/v1/chat/completions`
- **Models**: `qwen3-next-80b-a3b-instruct` (default), `webthinker-qwq-32b`
- **Key file**: Store API key in `key` file (never commit)

---

## Performance Notes

- **Batch size 10**: Optimal balance of throughput and reliability
- **~40 records/minute**: Typical processing speed
- **Confidence filter**: Reduces output by ~40% (removes low-quality detections)
- **Single LLM call**: V3 optimization reduces API calls by 50% vs two-stage

---

## Troubleshooting

**Timeout errors:**
```bash
python run_sentiment_analysis_v2.py --timeout 300  # Increase to 5 minutes
```

**Test with small sample:**
```bash
python run_sentiment_analysis_v2.py --max-payloads 20  # Process only 20 payloads
```

**Resume interrupted run:**
The pipeline automatically resumes from the last processed payload.

---

## License

MIT License
