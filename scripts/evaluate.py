#!/usr/bin/env python3
"""
Evaluate nanochat checkpoints on in-corpus and out-of-corpus questions.

Runs up to four experimental conditions:
  1. In-corpus, closed-book (parametric knowledge only)
  2. In-corpus, w/ FineWeb context (RAG with pre-training data)
  3. In-corpus, w/ original context (SQuAD only)
  4. Out-of-corpus, w/ original context (SQuAD only)

Usage:
    python scripts/evaluate.py \
        --model pankajmathur/nanochat-d34-sft-hf \
        --projection output/squad_stage2.pkl \
        --dataset squad \
        --fineweb_path /path/to/fineweb-edu-100b-shuffle \
        --output output/eval_squad.pkl
"""

import os
import sys
import argparse
import pickle
from tqdm import tqdm

# Allow running without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache")

from nanoknow.evaluator import NanoChatEvaluator, get_fineweb_context


def load_squad_contexts():
    """Load original SQuAD contexts for the validation set."""
    from datasets import load_dataset

    dataset = load_dataset("rajpurkar/squad", split="validation")
    contexts = {}
    for item in dataset:
        key = item["question"]
        contexts[key] = item["context"]
    return contexts


def run_condition(evaluator, questions, condition_name, context_fn=None):
    """Run evaluation for a single condition."""
    results = []
    correct = 0

    for q in tqdm(questions, desc=condition_name):
        question = q["original_question"]
        answers = q["original_answers"]

        context = context_fn(q) if context_fn else None

        if context_fn and context is None:
            continue

        prompt = evaluator.format_prompt(question, context)
        prediction = evaluator.generate(prompt)
        is_correct = evaluator.exact_match(prediction, answers)

        results.append({
            "question": question,
            "answers": answers,
            "prediction": prediction,
            "correct": is_correct,
            "context_length": len(context.split()) if context else 0,
        })
        if is_correct:
            correct += 1

    total = len(results)
    accuracy = correct / total if total else 0
    print(f"  {condition_name}: {accuracy:.1%} ({correct}/{total})")

    return {
        "condition": condition_name,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate nanochat checkpoints")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--projection", type=str, required=True, help="Stage 2 projection pkl")
    parser.add_argument("--dataset", type=str, choices=["squad", "nq"], required=True)
    parser.add_argument("--fineweb_path", type=str, default=None, help="Path to FineWeb parquet shards")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Load projection data
    print(f"Loading projection: {args.projection}")
    with open(args.projection, "rb") as f:
        data = pickle.load(f)

    in_corpus = [r for r in data["results"] if r.get("verified")]
    out_corpus = [r for r in data["results"] if not r.get("verified")]
    print(f"In-corpus: {len(in_corpus)}, Out-of-corpus: {len(out_corpus)}")

    if args.limit:
        in_corpus = in_corpus[: args.limit]
        out_corpus = out_corpus[: args.limit]

    evaluator = NanoChatEvaluator(args.model)
    all_results = {}

    # Condition 1: Closed-book
    print("\n--- In-corpus, Closed-Book ---")
    all_results["in_corpus_no_context"] = run_condition(
        evaluator, in_corpus, "in_corpus_no_context"
    )

    # Condition 2: FineWeb context
    if args.fineweb_path:
        print("\n--- In-corpus, w/ FineWeb Context ---")

        def fineweb_ctx(q):
            doc_id = q.get("doc_id")
            answer = q.get("matched_answer")
            if doc_id and answer:
                return get_fineweb_context(doc_id, answer, args.fineweb_path)
            return None

        all_results["in_corpus_fineweb_context"] = run_condition(
            evaluator, in_corpus, "in_corpus_fineweb_context", fineweb_ctx
        )

    # Condition 3 & 4: Original context (SQuAD only)
    if args.dataset == "squad":
        print("\n--- Loading SQuAD original contexts ---")
        squad_contexts = load_squad_contexts()

        def orig_ctx(q):
            return squad_contexts.get(q["original_question"])

        print("\n--- In-corpus, w/ Original Context ---")
        all_results["in_corpus_original_context"] = run_condition(
            evaluator, in_corpus, "in_corpus_original_context", orig_ctx
        )

        print("\n--- Out-of-corpus, w/ Original Context ---")
        all_results["out_corpus_original_context"] = run_condition(
            evaluator, out_corpus, "out_corpus_original_context", orig_ctx
        )

    # Save
    with open(args.output, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
