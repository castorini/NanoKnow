import argparse
import json
import pickle
import re
from pathlib import Path


def clean_json_output(text):
    if not isinstance(text, str):
        return {"correct": False, "explanation": "MISSING_JUDGE_OUTPUT"}

    if "</think>" in text:
        text = text.split("</think>")[-1]

    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"correct": False, "explanation": "JSON_PARSE_ERROR"}


def compute_condition_scores(predictions):
    total = len(predictions)
    if total == 0:
        return {
            "count": 0,
            "exact_match_accuracy": 0.0,
            "llm_judge_accuracy": 0.0,
        }

    em_correct = sum(bool(pred.get("exact_match_score", False)) for pred in predictions)
    judge_correct = sum(
        bool(clean_json_output(pred.get("llm_judge_score")).get("correct", False))
        for pred in predictions
    )

    return {
        "count": total,
        "exact_match_accuracy": em_correct / total,
        "llm_judge_accuracy": judge_correct / total,
    }


def load_scored_results(input_file):
    with open(input_file, "rb") as handle:
        return pickle.load(handle)


def main():
    parser = argparse.ArgumentParser(
        description="Compute eval accuracies for every condition in a scored pickle."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to a *_scored.pkl file containing exact-match and LLM-judge scores.",
    )
    args = parser.parse_args()

    data = load_scored_results(args.input_file)

    summary = {}
    for condition_name, condition_data in data.items():
        predictions = condition_data.get("results", [])
        summary[condition_name] = compute_condition_scores(predictions)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
