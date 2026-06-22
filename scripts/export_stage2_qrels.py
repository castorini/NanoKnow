#!/usr/bin/env python3
"""Export NanoKnow topics, answers, and qrels from a final Stage 2 pickle."""

import argparse
import json
import os
import pickle
import re
from pathlib import Path


CORPUS_RE = re.compile(r"^[a-z0-9_-]+$")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def validate_corpus_slug(corpus):
    if not CORPUS_RE.fullmatch(corpus):
        raise ValueError(
            "--corpus must contain only lowercase letters, digits, hyphens, "
            f"and underscores: {corpus!r}"
        )


def corpus_dir_name(corpus):
    if corpus == "fineweb":
        return "fineweb-edu"
    return corpus


def validate_stage2_data(data, dataset):
    if data.get("dataset") != dataset:
        raise ValueError(
            f"Dataset mismatch: --dataset={dataset!r}, pickle has "
            f"{data.get('dataset')!r}"
        )
    if "stage2_shard_id" in data:
        raise ValueError("Shard Stage 2 pickle detected; merge shards before export")

    required = ["dataset", "total", "llm_verified", "results"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Stage 2 pickle is missing required keys: {missing}")

    results = data["results"]
    if not isinstance(results, list):
        raise ValueError("Stage 2 pickle field 'results' must be a list")
    if len(results) != data["total"]:
        raise ValueError(
            f"Expected len(results) == total, got {len(results)} and {data['total']}"
        )

    for index, row in enumerate(results):
        missing_row_keys = [
            key
            for key in ["id", "original_question", "original_answers"]
            if key not in row
        ]
        if missing_row_keys:
            raise ValueError(f"Row {index} is missing keys: {missing_row_keys}")
        if not isinstance(row["original_answers"], list):
            raise ValueError(f"Row {index} original_answers must be a list")
        if row.get("verified") is True:
            verified_docs = row.get("verified_docs")
            if verified_docs is not None:
                if not isinstance(verified_docs, list):
                    raise ValueError(f"Row {index} verified_docs must be a list")
                missing_doc_ids = [
                    doc_index
                    for doc_index, doc in enumerate(verified_docs)
                    if not isinstance(doc, dict) or not doc.get("doc_id")
                ]
                if missing_doc_ids:
                    raise ValueError(
                        f"Row {index} verified_docs missing doc_id at "
                        f"positions: {missing_doc_ids[:5]}"
                    )
            elif not row.get("doc_id"):
                raise ValueError(f"Row {index} is verified but missing doc_id")

    return results


def build_outputs(results, dataset, corpus):
    answers_lines = []
    supported_topic_lines = []
    unsupported_topic_lines = []
    qrels_lines = []

    for row in results:
        qid = str(row["id"])
        question = str(row["original_question"])
        answers = [str(answer) for answer in row["original_answers"]]

        answers_lines.append(
            json.dumps({"qid": qid, "answer": answers}, ensure_ascii=False)
        )

        if row.get("verified") is True:
            supported_topic_lines.append(f"{qid}\t{question}")
            verified_docs = row.get("verified_docs")
            if verified_docs is None:
                verified_docs = [{"doc_id": row["doc_id"]}]
            for doc in verified_docs:
                qrels_lines.append(f"{qid} Q0 {doc['doc_id']} 1")
        else:
            unsupported_topic_lines.append(f"{qid}\t{question}")

    filenames = {
        f"answers.nanoknow-{dataset}.jsonl": answers_lines,
        f"topics.nanoknow-{dataset}-{corpus}.supported.tsv": supported_topic_lines,
        f"topics.nanoknow-{dataset}-{corpus}.unsupported.tsv": unsupported_topic_lines,
        f"qrels.nanoknow-{dataset}-{corpus}.supported.txt": qrels_lines,
    }

    return {
        name: "\n".join(lines) + ("\n" if lines else "")
        for name, lines in filenames.items()
    }


def write_output(path, content, overwrite):
    encoded = content.encode("utf-8")
    if path.exists():
        existing = path.read_bytes()
        if existing == encoded:
            return "unchanged"
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite differing file without --overwrite: {path}"
            )

    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_bytes(encoded)
    os.replace(tmp_path, path)
    return "written"


def count_lines(content):
    return content.count("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Export NanoKnow topics, answers, and qrels from Stage 2 output"
    )
    parser.add_argument("--input", required=True, help="Final merged Stage 2 pickle")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["nq", "squad"],
        help="Dataset name used in output filenames",
    )
    parser.add_argument(
        "--corpus",
        required=True,
        help="Corpus slug used in topic/qrels filenames, e.g. climbmix",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Dataset output directory. Default: questions-and-qrels/<dataset>. "
            "Answers are written here; corpus-specific topics/qrels are "
            "written under <output-dir>/<corpus-folder>."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files whose content differs",
    )
    args = parser.parse_args()

    validate_corpus_slug(args.corpus)
    data = load_pickle(args.input)
    results = validate_stage2_data(data, args.dataset)

    dataset_output_dir = Path(args.output_dir or f"questions-and-qrels/{args.dataset}")
    corpus_output_dir = dataset_output_dir / corpus_dir_name(args.corpus)
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    corpus_output_dir.mkdir(parents=True, exist_ok=True)

    outputs = build_outputs(results, args.dataset, args.corpus)
    statuses = {}
    for filename, content in outputs.items():
        output_dir = dataset_output_dir if filename.startswith("answers.") else corpus_output_dir
        statuses[filename] = write_output(
            output_dir / filename,
            content,
            overwrite=args.overwrite,
        )

    supported = sum(1 for row in results if row.get("verified") is True)
    unsupported = len(results) - supported
    qrels_name = f"qrels.nanoknow-{args.dataset}-{args.corpus}.supported.txt"

    print("Export complete")
    print(f"  dataset: {args.dataset}")
    print(f"  corpus: {args.corpus}")
    print(f"  total: {len(results)}")
    print(f"  supported: {supported}")
    print(f"  unsupported: {unsupported}")
    print(f"  qrels: {count_lines(outputs[qrels_name])}")
    print(f"  dataset_output_dir: {dataset_output_dir}")
    print(f"  corpus_output_dir: {corpus_output_dir}")
    for filename, status in statuses.items():
        output_dir = dataset_output_dir if filename.startswith("answers.") else corpus_output_dir
        print(f"  {status}: {output_dir / filename}")


if __name__ == "__main__":
    main()
