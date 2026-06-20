#!/usr/bin/env python3
"""
Project QA benchmarks onto a pre-training corpus.

Full NanoKnow pipeline:
  Stage 1: BM25 retrieval + answer string matching
  Stage 2: LLM-based verification to filter coincidental matches

Usage:
    # Stage 1 only (fast, no GPU required)
    python scripts/project.py --dataset squad --stage 1 --output output/squad_stage1.pkl

    # Stage 2 (requires GPU for LLM judge)
    python scripts/project.py --stage 2 --input output/squad_stage1.pkl --output output/squad_stage2.pkl

    # Both stages in one run
    python scripts/project.py --dataset squad --stage both --output output/squad_projected.pkl
"""

import os
import sys
import argparse
import pickle
from tqdm import tqdm

# Allow running without pip install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/hf_cache")


def strip_full_text(result):
    """Drop bulky document text while preserving Stage 1 match metadata."""
    for doc in result.get("matching_docs", []):
        doc.pop("full_text", None)
    return result


def run_stage1(args):
    """Stage 1: BM25 retrieval + answer string matching."""
    from datasets import load_dataset
    from nanoknow.retriever import BM25Retriever, PyseriniRestRetriever

    # Load dataset
    if args.dataset == "nq":
        dataset = load_dataset("google-research-datasets/nq_open", split="validation")
        questions = [d["question"] for d in dataset]
        answers = [d["answer"] for d in dataset]
    elif args.dataset == "squad":
        dataset = load_dataset("rajpurkar/squad", split="validation")
        questions = [d["question"] for d in dataset]
        answers = [d["answers"]["text"] for d in dataset]
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    if args.limit:
        questions = questions[: args.limit]
        answers = answers[: args.limit]

    print(f"\nStage 1: Processing {len(questions)} {args.dataset.upper()} questions")

    if args.retriever == "api":
        retriever = PyseriniRestRetriever(
            index_path=args.index_path,
            api_base_url=args.api_base_url,
            api_token_env=args.api_token_env,
            top_k=args.top_k,
            window_size=args.window_size,
        )
    else:
        retriever = BM25Retriever(
            index_path=args.index_path,
            top_k=args.top_k,
            window_size=args.window_size,
        )

    results = []
    has_answer_count = 0
    partial_path = f"{args.output}.partial"

    def save_checkpoint():
        completed = len(results)
        checkpoint_data = {
            "dataset": args.dataset,
            "split": "validation",
            "stage": 1,
            "top_k": args.top_k,
            "window_size": args.window_size,
            "total": len(questions),
            "completed": completed,
            "has_answer": has_answer_count,
            "has_answer_rate": has_answer_count / completed if completed else 0,
            "results": results,
        }
        tmp_path = f"{partial_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        os.replace(tmp_path, partial_path)

    if os.path.exists(partial_path):
        print(f"Loading Stage 1 checkpoint from {partial_path}...")
        with open(partial_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        expected = {
            "dataset": args.dataset,
            "split": "validation",
            "top_k": args.top_k,
            "window_size": args.window_size,
        }
        actual = {k: checkpoint_data.get(k) for k in expected}
        if actual != expected:
            raise ValueError(
                f"Checkpoint metadata mismatch: expected {expected}, got {actual}"
            )
        results = checkpoint_data["results"]
        if len(results) > len(questions):
            raise ValueError(
                f"Checkpoint has {len(results)} results for {len(questions)} questions"
            )
        if not args.store_full_text:
            for result in results:
                strip_full_text(result)
        has_answer_count = sum(1 for r in results if r.get("has_answer"))
        print(f"Resuming Stage 1: {len(results)}/{len(questions)} questions complete")
        if not args.store_full_text:
            save_checkpoint()

    start = len(results)

    for i, (q, ans) in enumerate(
        tqdm(
            zip(questions[start:], answers[start:]),
            total=len(questions),
            initial=start,
            desc="Stage 1",
        ),
        start=start,
    ):
        if isinstance(ans, str):
            ans = [ans]

        result = retriever.search(q, ans)
        if not args.store_full_text:
            strip_full_text(result)
        result["id"] = i
        results.append(result)

        if result["has_answer"]:
            has_answer_count += 1

        if len(results) % 25 == 0:
            save_checkpoint()

    total = len(results)
    print(f"\nStage 1 complete: {has_answer_count}/{total} ({has_answer_count/total:.1%}) have answer matches")

    output_data = {
        "dataset": args.dataset,
        "split": "validation",
        "stage": 1,
        "top_k": args.top_k,
        "window_size": args.window_size,
        "total": total,
        "has_answer": has_answer_count,
        "has_answer_rate": has_answer_count / total,
        "results": results,
    }

    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"Saved to {args.output}")

    return output_data


def run_stage2(args, stage1_data=None):
    """Stage 2: LLM-based verification."""
    from nanoknow.verifier import LLMVerifier

    if stage1_data is None:
        print(f"Loading Stage 1 results from {args.input}...")
        with open(args.input, "rb") as f:
            stage1_data = pickle.load(f)

    to_verify = [
        r for r in stage1_data["results"]
        if r.get("has_answer") and r.get("matching_docs")
    ]
    print(f"\nStage 2: Verifying {len(to_verify)} has-answer questions with LLM")

    total = len(stage1_data["results"])
    has_answer = stage1_data["has_answer"]
    partial_path = f"{args.output}.partial"
    verified_results = []
    verified_by_id = {}

    if os.path.exists(partial_path):
        print(f"Loading Stage 2 checkpoint from {partial_path}...")
        with open(partial_path, "rb") as f:
            checkpoint_data = pickle.load(f)
        expected = {
            "dataset": stage1_data["dataset"],
            "split": stage1_data["split"],
            "model": args.model,
            "total": total,
            "has_answer": has_answer,
        }
        actual = {k: checkpoint_data.get(k) for k in expected}
        if actual != expected:
            raise ValueError(
                f"Checkpoint metadata mismatch: expected {expected}, got {actual}"
            )
        verified_results = checkpoint_data.get("verified_results", [])
        verified_by_id = {r["id"]: r for r in verified_results}
        print(
            "Resuming Stage 2: "
            f"{len(verified_by_id)}/{len(to_verify)} has-answer questions complete"
        )

    def save_checkpoint():
        verified_count = sum(1 for r in verified_results if r.get("verified"))
        checkpoint_data = {
            "dataset": stage1_data["dataset"],
            "split": stage1_data["split"],
            "model": args.model,
            "total": total,
            "has_answer": has_answer,
            "has_answer_rate": has_answer / total,
            "to_verify": len(to_verify),
            "completed": len(verified_by_id),
            "llm_verified": verified_count,
            "verification_rate": verified_count / total,
            "verified_results": verified_results,
        }
        tmp_path = f"{partial_path}.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(checkpoint_data, f)
        os.replace(tmp_path, partial_path)

    verifier = LLMVerifier(model_name=args.model)
    remaining = [r for r in to_verify if r["id"] not in verified_by_id]

    for r in tqdm(
        remaining,
        total=len(to_verify),
        initial=len(verified_by_id),
        desc="Stage 2",
    ):
        verification = verifier.verify(
            r["original_question"],
            r["original_answers"],
            r["matching_docs"],
        )

        result = {
            "id": r["id"],
            "has_answer": True,
            "has_answer_docs": r["has_answer_docs"],
            "original_question": r["original_question"],
            "original_answers": r["original_answers"],
            **verification,
        }
        verified_results.append(result)
        verified_by_id[result["id"]] = result
        save_checkpoint()

    # Merge with non-has-answer results
    full_results = []

    for r in stage1_data["results"]:
        if r["id"] in verified_by_id:
            full_results.append(verified_by_id[r["id"]])
        else:
            full_results.append({
                "id": r["id"],
                "has_answer": False,
                "verified": False,
                "original_question": r["original_question"],
                "original_answers": r["original_answers"],
            })

    verified_count = sum(1 for r in verified_results if r.get("verified"))
    print(f"\nStage 2 complete:")
    print(f"  String match:  {has_answer}/{total} ({has_answer/total:.1%})")
    print(f"  LLM verified:  {verified_count}/{total} ({verified_count/total:.1%})")
    print(f"  Filtered out:  {has_answer - verified_count} coincidental matches")

    output_data = {
        "dataset": stage1_data["dataset"],
        "split": stage1_data["split"],
        "model": args.model,
        "total": total,
        "has_answer": has_answer,
        "llm_verified": verified_count,
        "has_answer_rate": has_answer / total,
        "verification_rate": verified_count / total,
        "results": full_results,
    }

    with open(args.output, "wb") as f:
        pickle.dump(output_data, f)
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"Saved to {args.output}")

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Project QA benchmarks onto pre-training corpora"
    )
    parser.add_argument(
        "--dataset", type=str, choices=["nq", "squad"],
        help="Dataset to project (required for stage 1)",
    )
    parser.add_argument(
        "--stage", type=str, choices=["1", "2", "both"], default="both",
        help="Which stage(s) to run",
    )
    parser.add_argument("--input", type=str, help="Input file (for stage 2)")
    parser.add_argument("--output", type=str, required=True, help="Output file")
    parser.add_argument(
        "--index_path", type=str,
        default="/home/tardis/shared/llms/hub/datasets--karpathy--fineweb-edu-100b-shuffle/index",
        help="Path to the Lucene index, or REST API index name when --retriever api",
    )
    parser.add_argument(
        "--retriever", type=str, choices=["local", "api"], default="local",
        help="Retriever backend for stage 1",
    )
    parser.add_argument(
        "--api_base_url", type=str, default="http://99.251.12.72:8081",
        help="Pyserini REST API base URL",
    )
    parser.add_argument(
        "--api_token_env", type=str, default="PYSERINI_API_TOKEN",
        help="Environment variable containing the Pyserini REST API token",
    )
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument(
        "--store_full_text",
        action="store_true",
        help="Store full matched document text in Stage 1 outputs",
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    if args.stage in ("1", "both") and not args.dataset:
        parser.error("--dataset is required for stage 1")
    if args.stage == "2" and not args.input:
        parser.error("--input is required for stage 2")

    if args.stage == "1":
        run_stage1(args)
    elif args.stage == "2":
        run_stage2(args)
    else:  # both
        stage1_data = run_stage1(args)
        # swap output for stage2
        stage1_output = args.output.replace(".pkl", "_stage1.pkl")
        os.rename(args.output, stage1_output)
        args.input = stage1_output
        run_stage2(args, stage1_data=stage1_data)


if __name__ == "__main__":
    main()
