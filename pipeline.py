#!/usr/bin/env python3
"""
Final orchestration: Run concurrent pipeline + model tagging + charts
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"📌 {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ {description}")
    return result.returncode

def main():
    workspace = Path("/Users/majidkhan/Desktop/Passau/AI_lab_hacker_news")
    
    # Step 1: Prepare payloads
    run_command(
        "cd " + str(workspace) + " && python3 prepare_sentiment_payloads_v2.py --input hn_ai_filtered_v2.csv --output sentiment_payloads_v2.jsonl",
        "Step 1/3: Prepare sentiment payloads"
    )
    
    # Step 2: Run v2 sentiment analysis (batched)
    run_command(
        "cd " + str(workspace) + " && python3 run_sentiment_analysis_v2.py --input sentiment_payloads_v2.jsonl --output final_sentiment_results_v2.jsonl --batch-size 10 --min-confidence 0.7",
        "Step 2/3: Run sentiment analysis (v2 batched)"
    )
    
    # Step 3: Generate charts
    run_command(
        "cd " + str(workspace) + " && python3 sentiment_charts.py --input final_sentiment_results_v2.jsonl --output-dir charts_v2",
        "Step 3/3: Generate visualization charts"
    )
    
    print(f"\n{'='*70}")
    print(f"✅ PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutputs:")
    print(f"  Sentiment results: final_sentiment_results_v2.jsonl")
    print(f"  Charts: charts_v2/")
    print(f"\nKey file:")
    p = workspace / "final_sentiment_results_v2.jsonl"
    if p.exists():
        size = p.stat().st_size / 1024
        lines = sum(1 for _ in open(p))
        print(f"  {p.name:40s}: {lines:6d} records ({size:>8.1f} KB)")

if __name__ == "__main__":
    main()
