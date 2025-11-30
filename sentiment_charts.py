"""Generate sentiment/aspect charts from OpenRouter outputs.

Reads the newline-delimited JSON written by ``openrouter_sentiment_runner.py``
(``sentiment_llm_results.jsonl`` by default), parses the model's final JSON
response, aggregates sentiment and aspect counts, and emits a couple of helpful
charts using matplotlib.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

DEFAULT_INPUT = "sentiment_llm_results.jsonl"
DEFAULT_OUTPUT_DIR = "charts"
DEFAULT_TOP_ASPECTS = 10


def parse_model_json(raw: Optional[str]) -> Optional[Dict]:
    if not raw:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Attempt to salvage by trimming to the first/last braces.
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            return None


def load_sentiment_payloads(path: Path) -> List[Dict]:
    payloads: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            parsed = parse_model_json(record.get("final_response") or record.get("initial_response"))
            if parsed is None:
                continue
            payloads.append(parsed)
    if not payloads:
        raise ValueError(f"No parsable sentiment payloads were found in {path}")
    return payloads


def aggregate(payloads: List[Dict]):
    sentiment_counts = Counter()
    aspect_counts = Counter()
    aspect_sentiment_counts: Dict[str, Counter] = defaultdict(Counter)

    for payload in payloads:
        sentiment = (payload.get("sentiment") or "").strip().lower()
        if sentiment:
            sentiment_counts[sentiment] += 1
        for aspect in payload.get("aspects", []) or []:
            aspect_key = (aspect.get("aspect") or "").strip()
            if not aspect_key:
                continue
            if aspect.get("present"):
                aspect_counts[aspect_key] += 1
                aspect_sentiment = (aspect.get("sentiment") or "").strip().lower()
                if aspect_sentiment:
                    aspect_sentiment_counts[aspect_key][aspect_sentiment] += 1
    return sentiment_counts, aspect_counts, aspect_sentiment_counts


def plot_sentiment_distribution(counter: Counter, output_dir: Path, dpi: int) -> Optional[Path]:
    if not counter:
        print("No sentiment data available for plotting.")
        return None
    labels = list(counter.keys())
    values = [counter[label] for label in labels]

    plt.figure(figsize=(6, 4), dpi=dpi)
    bars = plt.bar(labels, values, color="#4C72B0")
    plt.title("Sentiment distribution")
    plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values, strict=False):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha="center", va="bottom")

    output_path = output_dir / "sentiment_distribution.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_top_aspects(counter: Counter, output_dir: Path, top_n: int, dpi: int) -> Optional[Path]:
    if not counter:
        print("No aspect data available for plotting.")
        return None
    most_common = counter.most_common(top_n)
    labels = [item[0] for item in most_common]
    values = [item[1] for item in most_common]
    y_pos = list(range(len(labels)))

    plt.figure(figsize=(8, 4 + len(labels) * 0.2), dpi=dpi)
    bars = plt.barh(y_pos, values, color="#55A868")
    plt.yticks(y_pos, labels)
    plt.xlabel("Count")
    plt.title(f"Top {len(labels)} aspects (present=true)")
    plt.grid(axis="x", linestyle="--", alpha=0.3)
    for bar, value in zip(bars, values, strict=False):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, str(value), va="center", ha="left")

    output_path = output_dir / "top_aspects.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matplotlib charts from sentiment results")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="sentiment_llm_results.jsonl path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to save generated charts")
    parser.add_argument("--top-aspects", type=int, default=DEFAULT_TOP_ASPECTS, help="How many aspects to plot")
    parser.add_argument("--dpi", type=int, default=120, help="Chart DPI when saving")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_sentiment_payloads(results_path)
    sentiment_counts, aspect_counts, _ = aggregate(payloads)

    charts: List[Path] = []
    sentiment_chart = plot_sentiment_distribution(sentiment_counts, output_dir, args.dpi)
    if sentiment_chart:
        charts.append(sentiment_chart)
    aspect_chart = plot_top_aspects(aspect_counts, output_dir, args.top_aspects, args.dpi)
    if aspect_chart:
        charts.append(aspect_chart)

    if charts:
        print("Charts saved:")
        for chart_path in charts:
            print(f"  - {chart_path}")
    else:
        print("No charts were generated (insufficient data).")


if __name__ == "__main__":
    main()
