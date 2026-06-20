#!/usr/bin/env python3
"""Merge Stage 2 shard outputs into the standard Stage 2 pickle format."""

import argparse
import os
import pickle


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Merge sharded Stage 2 verification outputs"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Original Stage 1 pickle used by all Stage 2 shards",
    )
    parser.add_argument(
        "--shards",
        nargs="+",
        required=True,
        help="Stage 2 shard pickle outputs",
    )
    parser.add_argument("--output", required=True, help="Merged Stage 2 pickle")
    args = parser.parse_args()

    stage1_data = load_pickle(args.input)
    shard_data = [load_pickle(path) for path in args.shards]

    if not shard_data:
        raise ValueError("At least one shard is required")

    expected = {
        "dataset": stage1_data["dataset"],
        "split": stage1_data["split"],
        "total": len(stage1_data["results"]),
        "has_answer": stage1_data["has_answer"],
    }
    model = shard_data[0].get("model")
    shard_count = shard_data[0].get("stage2_shard_count")

    if shard_count != len(args.shards):
        raise ValueError(
            f"Expected {shard_count} shard files, got {len(args.shards)}"
        )

    seen_shard_ids = set()
    verified_by_id = {}
    for path, data in zip(args.shards, shard_data):
        actual = {k: data.get(k) for k in expected}
        if actual != expected:
            raise ValueError(
                f"Shard metadata mismatch in {path}: expected {expected}, got {actual}"
            )
        if data.get("model") != model:
            raise ValueError(f"Shard model mismatch in {path}")
        if data.get("stage2_shard_count") != shard_count:
            raise ValueError(f"Shard count mismatch in {path}")

        shard_id = data.get("stage2_shard_id")
        if shard_id in seen_shard_ids:
            raise ValueError(f"Duplicate shard id {shard_id}")
        seen_shard_ids.add(shard_id)

        for result in data.get("results", []):
            result_id = result["id"]
            if result_id in verified_by_id:
                raise ValueError(f"Duplicate verified result id {result_id}")
            verified_by_id[result_id] = result

    expected_shard_ids = set(range(shard_count))
    if seen_shard_ids != expected_shard_ids:
        raise ValueError(
            f"Missing shard ids: expected {expected_shard_ids}, got {seen_shard_ids}"
        )

    full_results = []
    for row in stage1_data["results"]:
        if row["id"] in verified_by_id:
            full_results.append(verified_by_id[row["id"]])
        else:
            full_results.append({
                "id": row["id"],
                "has_answer": False,
                "verified": False,
                "original_question": row["original_question"],
                "original_answers": row["original_answers"],
            })

    total = len(full_results)
    has_answer = stage1_data["has_answer"]
    llm_verified = sum(1 for r in verified_by_id.values() if r.get("verified"))

    output_data = {
        "dataset": stage1_data["dataset"],
        "split": stage1_data["split"],
        "model": model,
        "total": total,
        "has_answer": has_answer,
        "llm_verified": llm_verified,
        "has_answer_rate": has_answer / total,
        "verification_rate": llm_verified / total,
        "results": full_results,
    }

    tmp_path = f"{args.output}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(output_data, f)
    os.replace(tmp_path, args.output)

    print(f"Saved merged Stage 2 output to {args.output}")
    print(f"  dataset: {output_data['dataset']}")
    print(f"  split: {output_data['split']}")
    print(f"  model: {output_data['model']}")
    print(f"  total: {total}")
    print(f"  has_answer: {has_answer}")
    print(f"  llm_verified: {llm_verified}")
    print(f"  verification_rate: {output_data['verification_rate']}")


if __name__ == "__main__":
    main()
