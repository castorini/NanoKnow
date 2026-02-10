#!/usr/bin/env python3
"""
Analyze how answer frequency in the pre-training corpus affects model accuracy.

Reads evaluation results and frequency data (qrels), stratifies questions by
how many FineWeb documents contain their answer, and reports accuracy per bucket.

Usage:
    python scripts/analyze_frequency.py \
        --eval_dir output/evals/ \
        --qrel_file qrels/squad_in_corpus.txt \
        --dataset squad \
        --output output/frequency_analysis.json
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
from collections import Counter, defaultdict

# Allow running without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache")


BUCKETS = [
    ("Rare (1-5)", 1, 5),
    ("Low (6-20)", 6, 20),
    ("Medium (21-50)", 21, 50),
    ("High (51+)", 51, float("inf")),
]


def bucket_for(frequency: int) -> str:
    for name, lo, hi in BUCKETS:
        if lo <= frequency <= hi:
            return name
    return "not_found"


def load_frequency_from_qrel(qrel_path: str) -> dict:
    """Count how many documents each question ID maps to in the qrel file.

    Expects TREC-style lines: ``qid 0 docid relevance``
    or CSV-style lines: ``qid, question, answer, docid, offset``
    """
    freq = Counter()
    with open(qrel_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",") if "," in line else line.split()
            qid = parts[0].strip()
            freq[qid] += 1
    print(f"Loaded frequencies for {len(freq)} question IDs")
    return dict(freq)


def load_question_id_mapping(dataset: str) -> dict:
    """Map question ID → question text for SQuAD / NQ."""
    from datasets import load_dataset

    if dataset == "squad":
        ds = load_dataset("rajpurkar/squad", split="validation")
        return {item["id"]: item["question"] for item in ds}
    else:  # nq
        ds = load_dataset("google-research-datasets/nq_open", split="validation")
        return {str(i): item["question"] for i, item in enumerate(ds)}


def analyze_eval_file(
    filepath: str, question_to_freq: dict
) -> dict:
    """Analyze a single evaluation pkl file."""
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    analysis = {}
    for condition, cond_data in data.items():
        if not isinstance(cond_data, dict) or "results" not in cond_data:
            continue

        bucket_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for result in cond_data["results"]:
            question = result.get("question", "")
            correct = result.get("correct", False)
            frequency = question_to_freq.get(question, 0)
            bucket = bucket_for(frequency)

            bucket_stats[bucket]["total"] += 1
            if correct:
                bucket_stats[bucket]["correct"] += 1

        analysis[condition] = {
            bucket: {
                "accuracy": s["correct"] / s["total"] if s["total"] else 0,
                "correct": s["correct"],
                "total": s["total"],
            }
            for bucket, s in bucket_stats.items()
        }

    return analysis


def main():
    parser = argparse.ArgumentParser(description="Frequency analysis")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory with eval pkl files")
    parser.add_argument("--qrel_file", type=str, required=True, help="Qrel file with frequency info")
    parser.add_argument("--dataset", type=str, choices=["squad", "nq"], required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Load frequency data
    qrel_freq = load_frequency_from_qrel(args.qrel_file)
    id_to_question = load_question_id_mapping(args.dataset)

    # Map question text → frequency
    question_to_freq = {}
    for qid, freq in qrel_freq.items():
        if qid in id_to_question:
            question_to_freq[id_to_question[qid]] = freq
    print(f"Mapped {len(question_to_freq)} questions to frequencies")

    # Analyze all eval files
    all_results = {}
    for filename in sorted(os.listdir(args.eval_dir)):
        if not filename.endswith(".pkl") or args.dataset not in filename.lower():
            continue

        filepath = os.path.join(args.eval_dir, filename)
        model_name = filename.replace(".pkl", "")
        print(f"\nAnalyzing: {filename}")

        try:
            analysis = analyze_eval_file(filepath, question_to_freq)
            all_results[model_name] = analysis
        except Exception as e:
            print(f"  Error: {e}")

    # Print summary
    print(f"\n{'='*80}")
    print("FREQUENCY ANALYSIS SUMMARY")
    print(f"{'='*80}")

    bucket_names = [name for name, _, _ in BUCKETS]
    for model, analysis in sorted(all_results.items()):
        for condition, buckets in analysis.items():
            if "no_context" in condition:
                print(f"\n{model} / {condition}:")
                for bn in bucket_names:
                    if bn in buckets:
                        s = buckets[bn]
                        print(f"  {bn}: {s['accuracy']*100:.1f}% ({s['correct']}/{s['total']})")

    # Save
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
