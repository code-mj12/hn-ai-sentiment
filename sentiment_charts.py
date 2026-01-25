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
    aspect_ratings_by_model: Dict[str, Dict[str, List[int]]]  # {model: {aspect: [ratings]}}


def aggregate(payloads: List[Dict]) -> AggregationResult:
    sentiment_counts: Counter = Counter()
    aspect_counts: Counter = Counter()
    aspect_sentiment_counts: Dict[str, Counter] = defaultdict(Counter)
    confidence_by_sentiment: Dict[str, List[float]] = defaultdict(list)
    confidence_by_aspect: Dict[str, List[float]] = defaultdict(list)
    aspect_sentiment_by_model: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    aspect_ratings_by_model: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: defaultdict(list))

    for payload in payloads:
        model = (payload.get("ai_model") or payload.get("model") or "").strip().lower() or "unknown"

        # Process detected aspects (V3: sentiment is inside each aspect)
        detected_aspects = payload.get("detected_aspects") or []
        for aspect in detected_aspects:
            aspect_key = (aspect.get("aspect") or "").strip()
            if not aspect_key:
                continue
            aspect_counts[aspect_key] += 1
            confidence = aspect.get("confidence")
            if isinstance(confidence, (int, float)):
                confidence_by_aspect[aspect_key].append(confidence)
            # Collect rating for model-aspect
            rating = aspect.get("rating")
            if isinstance(rating, (int, float)):
                aspect_ratings_by_model[model][aspect_key].append(int(rating))

            # V3: Sentiment is now inside detected_aspects (not stage2_result)
            aspect_sentiment = (aspect.get("sentiment") or "").strip().lower()
            if aspect_sentiment:
                sentiment_counts[aspect_sentiment] += 1
                aspect_sentiment_counts[aspect_key][aspect_sentiment] += 1
                aspect_sentiment_by_model[model][aspect_key][aspect_sentiment] += 1
                if isinstance(confidence, (int, float)):
                    confidence_by_sentiment[aspect_sentiment].append(confidence)

    return AggregationResult(
        sentiment_counts=sentiment_counts,
        aspect_counts=aspect_counts,
        aspect_sentiment_counts=aspect_sentiment_counts,
        confidence_by_sentiment=confidence_by_sentiment,
        confidence_by_aspect=confidence_by_aspect,
        aspect_sentiment_by_model=dict(aspect_sentiment_by_model),
        aspect_ratings_by_model=dict(aspect_ratings_by_model),
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
    aspect_ratings_by_model: Dict[str, Dict[str, List[int]]],
    output_dir: Path,
    top_n: int,
    dpi: int,
    min_model_count: int = 10,
) -> Optional[Path]:
    """Plot detailed model analysis with sentiment and aspect ratings.

    aspect_sentiment_by_model: {model: {aspect: Counter of sentiments}}
    aspect_ratings_by_model: {model: {aspect: [ratings]}}
    min_model_count: Only show models with at least this many mentions
    """
    if not aspect_sentiment_by_model:
        return None

    # Count total mentions per model, sentiment totals
    model_sentiment_totals: Dict[str, Counter] = {}
    model_total_counts: Dict[str, int] = {}

    for model, aspect_dict in aspect_sentiment_by_model.items():
        # Skip "unknown" model
        if model == "unknown":
            continue
        sentiment_counter: Counter = Counter()
        for aspect, sent_counts in aspect_dict.items():
            sentiment_counter.update(sent_counts)
        total = sum(sentiment_counter.values())
        if total >= min_model_count:
            model_sentiment_totals[model] = sentiment_counter
            model_total_counts[model] = total

    if not model_sentiment_totals:
        print(f"No models (excluding 'unknown') with >= {min_model_count} mentions for chart.")
        return None

    # Sort models by total count (descending)
    sorted_models = sorted(
        model_total_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )
    models = [m for m, _ in sorted_models]

    if not models:
        return None

    # Create figure with 2 subplots: sentiment bars + aspect ratings heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=dpi,
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    # === LEFT: Sentiment grouped bar chart ===
    sentiments = ["positive", "negative", "neutral", "mixed"]
    colors = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#95a5a6", "mixed": "#f39c12"}

    data = {sent: [] for sent in sentiments}
    for model in models:
        counter = model_sentiment_totals[model]
        total = sum(counter.values())
        for sent in sentiments:
            pct = (counter.get(sent, 0) / total * 100) if total > 0 else 0
            data[sent].append(pct)

    x = np.arange(len(models))
    width = 0.2

    for i, sent in enumerate(sentiments):
        offset = (i - 1.5) * width
        bars = ax1.bar(x + offset, data[sent], width, label=sent.capitalize(), color=colors[sent])
        for bar, val in zip(bars, data[sent]):
            if val > 8:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8)

    model_labels = [f"{m}\n(n={model_total_counts[m]})" for m in models]
    ax1.set_ylabel('Percentage (%)')
    ax1.set_xlabel('AI Model')
    ax1.set_title('Sentiment Distribution by Model')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels, rotation=25, ha='right')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)

    # === RIGHT: Aspect RATINGS heatmap per model (1-5 scale) ===
    all_aspects = ["coding_speed", "coding_performance", "cost_price", "ethics", "innovation", "skepticism_hype"]

    # Build heatmap data (average rating per aspect per model)
    heatmap_data = []
    for model in models:
        model_ratings = aspect_ratings_by_model.get(model, {})
        row = []
        for asp in all_aspects:
            ratings = model_ratings.get(asp, [])
            avg_rating = sum(ratings) / len(ratings) if ratings else 0
            row.append(avg_rating)
        heatmap_data.append(row)

    heatmap_array = np.array(heatmap_data)

    # Use RdYlGn colormap: red=low rating, green=high rating
    im = ax2.imshow(heatmap_array, cmap="RdYlGn", aspect="auto", vmin=1, vmax=5)

    ax2.set_xticks(range(len(all_aspects)))
    ax2.set_yticks(range(len(models)))
    ax2.set_xticklabels([a.replace("_", "\n") for a in all_aspects], rotation=0, ha='center', fontsize=8)
    ax2.set_yticklabels(models)
    ax2.set_xlabel('Aspect')
    ax2.set_title('Avg Aspect Rating by Model (1-5)')

    # Add rating annotations
    for i in range(len(models)):
        for j in range(len(all_aspects)):
            val = heatmap_array[i, j]
            if val > 0:
                color = "white" if val < 2.5 or val > 4 else "black"
                ax2.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=9, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax2, label="Rating (1=Poor, 5=Excellent)", shrink=0.8)
    cbar.set_ticks([1, 2, 3, 4, 5])

    plt.suptitle(f'AI Model Analysis (min {min_model_count} mentions, excluding "unknown")',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_path = output_dir / "aspect_sentiment_by_model.png"
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
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
        aggregation.aspect_ratings_by_model,
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
