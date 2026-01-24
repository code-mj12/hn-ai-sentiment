"""Generate sentiment/aspect charts from unified pipeline outputs.

Reads the newline-delimited JSON written by ``run_sentiment_analysis.py``
(``final_sentiment_results.jsonl`` by default), aggregates sentiment and aspect
counts, and emits a couple of helpful charts using matplotlib.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

HN_ITEM_URL = "https://news.ycombinator.com/item?id={id}"

# Default to the v2 unified pipeline output
DEFAULT_INPUT = "final_sentiment_results_v2.jsonl"
DEFAULT_OUTPUT_DIR = "charts"
DEFAULT_TOP_ASPECTS = 10


def build_story_link(item_id: Optional[int]) -> Optional[str]:
    if item_id is None:
        return None
    return HN_ITEM_URL.format(id=item_id)


def load_results(path: Path) -> List[Dict]:
    """Load unified pipeline results from ``final_sentiment_results.jsonl``."""

    payloads: List[Dict] = []
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            payloads.append(record)

    if not payloads:
        raise ValueError(f"No parsable sentiment payloads were found in {path}")

    return payloads


@dataclass
class AggregationResult:
    sentiment_counts: Counter
    aspect_counts: Counter
    aspect_sentiment_counts: Dict[str, Counter]
    confidence_by_sentiment: Dict[str, List[float]]
    confidence_by_aspect: Dict[str, List[float]]
    aspect_sentiment_by_model: Dict[str, Dict[str, Counter]]  # {model: {aspect: Counter}}


def aggregate(payloads: List[Dict]) -> AggregationResult:
    sentiment_counts: Counter = Counter()
    aspect_counts: Counter = Counter()
    aspect_sentiment_counts: Dict[str, Counter] = defaultdict(Counter)
    confidence_by_sentiment: Dict[str, List[float]] = defaultdict(list)
    confidence_by_aspect: Dict[str, List[float]] = defaultdict(list)
    aspect_sentiment_by_model: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for payload in payloads:
        sentiment = (payload.get("overall_sentiment") or "").strip().lower()
        if sentiment:
            sentiment_counts[sentiment] += 1

        model = (payload.get("model") or "unknown").strip().lower()

        detected_aspects = payload.get("detected_aspects") or []
        for aspect in detected_aspects:
            aspect_key = (aspect.get("aspect") or "").strip()
            if not aspect_key:
                continue
            aspect_counts[aspect_key] += 1
            confidence = aspect.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence_by_aspect[aspect_key].append(confidence)

        aspect_sentiments = payload.get("aspect_sentiments") or []
        for aspect in aspect_sentiments:
            aspect_key = (aspect.get("aspect") or "").strip()
            aspect_sentiment = (aspect.get("sentiment") or "").strip().lower()
            if not aspect_key or not aspect_sentiment:
                continue
            aspect_sentiment_counts[aspect_key][aspect_sentiment] += 1
            # Also track by model
            aspect_sentiment_by_model[model][aspect_key][aspect_sentiment] += 1
            confidence = aspect.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence_by_sentiment[aspect_sentiment].append(confidence)
                confidence_by_aspect[aspect_key].append(confidence)

    return AggregationResult(
        sentiment_counts=sentiment_counts,
        aspect_counts=aspect_counts,
        aspect_sentiment_counts=aspect_sentiment_counts,
        confidence_by_sentiment=confidence_by_sentiment,
        confidence_by_aspect=confidence_by_aspect,
        aspect_sentiment_by_model=dict(aspect_sentiment_by_model),
    )


def build_summary(payloads: List[Dict], agg: AggregationResult, top_aspects: int) -> Dict:
    unique_story_ids = {payload.get("story_id") for payload in payloads if payload.get("story_id")}
    unique_root_ids = {payload.get("root_story_id") for payload in payloads if payload.get("root_story_id")}
    total_comments = len(payloads)
    total_aspect_mentions = sum(agg.aspect_counts.values())

    story_links = []
    for story_id in sorted(unique_story_ids):
        link = build_story_link(story_id)
        if link:
            story_links.append({"story_id": story_id, "url": link})

    root_story_links = []
    for root_id in sorted(unique_root_ids):
        link = build_story_link(root_id)
        if link:
            root_story_links.append({"root_story_id": root_id, "url": link})

    summary = {
        "comments_scored": total_comments,
        "unique_stories": len(unique_story_ids),
        "unique_root_stories": len(unique_root_ids),
        "sentiment_counts": dict(agg.sentiment_counts),
        "top_aspects": agg.aspect_counts.most_common(top_aspects),
        "avg_aspects_per_comment": (total_aspect_mentions / total_comments) if total_comments else 0,
        "story_links": story_links,
        "root_story_links": root_story_links,
    }
    return summary


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
    for bar, value in zip(bars, values):
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
    for bar, value in zip(bars, values):
        plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, str(value), va="center", ha="left")

    output_path = output_dir / "top_aspects.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path


def plot_aspect_sentiment_mix(
    aspect_counts: Counter,
    aspect_sentiment_counts: Dict[str, Counter],
    output_dir: Path,
    top_n: int,
    dpi: int,
) -> Optional[Path]:
    ordered_aspects = [name for name, _ in aspect_counts.most_common(top_n)]
    if not ordered_aspects:
        print("Not enough aspect data to build sentiment mix chart.")
        return None

    sentiments = ["positive", "negative", "mixed", "neutral"]
    matrix = np.zeros((len(sentiments), len(ordered_aspects)))
    for col, aspect in enumerate(ordered_aspects):
        counts = aspect_sentiment_counts.get(aspect, {})
        for row, category in enumerate(sentiments):
            matrix[row, col] = counts.get(category, 0)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
    bottom = np.zeros(len(ordered_aspects))
    for row, sentiment in enumerate(sentiments):
        ax.bar(
            ordered_aspects,
            matrix[row],
            bottom=bottom,
            label=sentiment.capitalize(),
        )
        bottom += matrix[row]
    plt.xticks(rotation=30, ha="right")
    ax.set_ylabel("Mentions")
    ax.set_title("Aspect sentiment mix (top aspects)")
    ax.legend()
    plt.tight_layout()
    output_path = output_dir / "aspect_sentiment_mix.png"
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_confidence_by_sentiment(
    confidence_by_sentiment: Dict[str, List[float]], output_dir: Path, dpi: int
) -> Optional[Path]:
    filtered = [(sentiment, values) for sentiment, values in confidence_by_sentiment.items() if values]
    if not filtered:
        print("No confidence data available to build sentiment boxplot.")
        return None

    labels = [item[0].capitalize() for item in filtered]
    data = [item[1] for item in filtered]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=dpi)
    ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Aspect confidence by sentiment")
    plt.tight_layout()
    output_path = output_dir / "confidence_by_sentiment.png"
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_aspect_confidence_leaderboard(
    confidence_by_aspect: Dict[str, List[float]], output_dir: Path, top_n: int, dpi: int
) -> Optional[Path]:
    averages = {
        aspect: (sum(values) / len(values))
        for aspect, values in confidence_by_aspect.items()
        if values
    }
    if not averages:
        print("No aspect confidence scores available.")
        return None

    leaderboard = sorted(averages.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    labels = [item[0] for item in leaderboard]
    values = [item[1] for item in leaderboard]

    fig, ax = plt.subplots(figsize=(8, 4 + len(labels) * 0.2), dpi=dpi)
    bars = ax.barh(labels, values, color="#4C72B0")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Average confidence")
    ax.set_title("Top aspect confidences")
    for bar, value in zip(bars, values):
        ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f"{value:.2f}", va="center")
    plt.tight_layout()
    output_path = output_dir / "aspect_confidence_leaderboard.png"
    plt.savefig(output_path)
    plt.close(fig)
    return output_path


def plot_aspect_sentiment_by_model(
    aspect_sentiment_by_model: Dict[str, Dict[str, Counter]],
    output_dir: Path,
    top_n: int,
    dpi: int,
) -> Optional[Path]:
    """Plot average sentiment score for each aspect, broken down by model.
    
    aspect_sentiment_by_model: {model: {aspect: Counter of sentiments}}
    """
    if not aspect_sentiment_by_model:
        return None
    
    # Calculate average sentiment score for each (model, aspect) pair
    model_aspect_scores: Dict[str, Dict[str, float]] = {}
    for model, aspect_dict in aspect_sentiment_by_model.items():
        model_aspect_scores[model] = {}
        for aspect, sentiment_counter in aspect_dict.items():
            # sentiment_counter maps "positive"|"neutral"|"negative"|"mixed" -> count
            total = sum(sentiment_counter.values())
            if total == 0:
                continue
            # Score: positive=+1, neutral=0, mixed=0.5, negative=-1
            score_map = {"positive": 1.0, "neutral": 0.0, "mixed": 0.5, "negative": -1.0}
            weighted_score = sum(
                score_map.get(sent, 0) * count for sent, count in sentiment_counter.items()
            ) / total
            model_aspect_scores[model][aspect] = weighted_score
    
    # Aggregate top aspects across all models
    all_aspects = Counter()
    for aspect_dict in model_aspect_scores.values():
        all_aspects.update(aspect_dict.keys())
    top_aspects = [asp for asp, _ in all_aspects.most_common(top_n)]
    
    if not top_aspects:
        return None
    
    # Prepare data: rows=aspects, columns=models
    models = sorted(model_aspect_scores.keys())
    data = []
    for aspect in top_aspects:
        row = [model_aspect_scores[model].get(aspect, 0.0) for model in models]
        data.append(row)
    
    data_array = np.array(data)
    
    # Create heatmap-like chart
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.5), max(5, len(top_aspects) * 0.4)), dpi=dpi)
    
    # Use imshow for heatmap
    im = ax.imshow(data_array, cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(top_aspects)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticklabels(top_aspects)
    ax.set_xlabel("Model")
    ax.set_ylabel("Aspect")
    ax.set_title(f"Aspect Sentiment by Model (Top {top_n})")
    
    # Add text annotations
    for i in range(len(top_aspects)):
        for j in range(len(models)):
            value = data_array[i, j]
            text = ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im, ax=ax, label="Sentiment Score")
    plt.tight_layout()
    
    output_path = output_dir / "aspect_sentiment_by_model.png"
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate matplotlib charts from sentiment results")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="final_sentiment_results.jsonl path")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Where to save generated charts")
    parser.add_argument("--top-aspects", type=int, default=DEFAULT_TOP_ASPECTS, help="How many aspects to plot")
    parser.add_argument("--dpi", type=int, default=120, help="Chart DPI when saving")
    parser.add_argument(
        "--summary-json",
        default="analysis_summary.json",
        help="Summary JSON filename (saved inside the output directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_results(results_path)
    aggregation = aggregate(payloads)

    charts: List[Path] = []
    sentiment_chart = plot_sentiment_distribution(aggregation.sentiment_counts, output_dir, args.dpi)
    if sentiment_chart:
        charts.append(sentiment_chart)
    aspect_chart = plot_top_aspects(aggregation.aspect_counts, output_dir, args.top_aspects, args.dpi)
    if aspect_chart:
        charts.append(aspect_chart)
    mix_chart = plot_aspect_sentiment_mix(
        aggregation.aspect_counts,
        aggregation.aspect_sentiment_counts,
        output_dir,
        args.top_aspects,
        args.dpi,
    )
    if mix_chart:
        charts.append(mix_chart)
    confidence_box = plot_confidence_by_sentiment(aggregation.confidence_by_sentiment, output_dir, args.dpi)
    if confidence_box:
        charts.append(confidence_box)
    confidence_leaderboard = plot_aspect_confidence_leaderboard(
        aggregation.confidence_by_aspect,
        output_dir,
        args.top_aspects,
        args.dpi,
    )
    if confidence_leaderboard:
        charts.append(confidence_leaderboard)
    by_model_chart = plot_aspect_sentiment_by_model(
        aggregation.aspect_sentiment_by_model,
        output_dir,
        args.top_aspects,
        args.dpi,
    )
    if by_model_chart:
        charts.append(by_model_chart)

    if charts:
        print("Charts saved:")
        for chart_path in charts:
            print(f"  - {chart_path}")
    else:
        print("No charts were generated (insufficient data).")

    summary = build_summary(payloads, aggregation, args.top_aspects)
    summary_path = output_dir / args.summary_json
    with summary_path.open("w", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
