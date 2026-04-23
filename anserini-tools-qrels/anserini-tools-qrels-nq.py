#!/usr/bin/env python3
"""
Generate Anserini-compatible topics, qrels, and answers files for NanoKnow NQ.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

DEFAULT_SUPPORTED_QRELS = REPO_ROOT / "qrels" / "nq_supported.txt"
DEFAULT_PICKLE = Path(
    "/u201/njedidi/past_projects/nanochat_tests_nour/"
    "nq_fineweb_eval_val_stage2.pkl"
)
DEFAULT_ANSWERS = SCRIPT_DIR / "answers.nanoknow-nq.jsonl"
DEFAULT_SUPPORTED_TOPICS = SCRIPT_DIR / "topics.nanoknow-nq.supported.tsv"
DEFAULT_UNSUPPORTED_TOPICS = SCRIPT_DIR / "topics.nanoknow-nq.unsupported.tsv"
DEFAULT_TREC_QRELS = SCRIPT_DIR / "qrels.nanoknow-nq.supported.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build NQ-only Anserini topics/qrels files from NanoKnow inputs."
    )
    parser.add_argument(
        "--supported-qrels",
        type=Path,
        default=DEFAULT_SUPPORTED_QRELS,
        help="Path to nq_supported.txt",
    )
    parser.add_argument(
        "--pickle-path",
        type=Path,
        default=DEFAULT_PICKLE,
        help='Path to the Stage 2 pickle containing a top-level "results" key',
    )
    parser.add_argument(
        "--answers-out",
        type=Path,
        default=DEFAULT_ANSWERS,
        help="Where to write answers.nanoknow-nq.jsonl",
    )
    parser.add_argument(
        "--supported-topics-out",
        type=Path,
        default=DEFAULT_SUPPORTED_TOPICS,
        help="Where to write topics.nanoknow-nq.supported.tsv",
    )
    parser.add_argument(
        "--unsupported-topics-out",
        type=Path,
        default=DEFAULT_UNSUPPORTED_TOPICS,
        help="Where to write topics.nanoknow-nq.unsupported.tsv",
    )
    parser.add_argument(
        "--qrels-out",
        type=Path,
        default=DEFAULT_TREC_QRELS,
        help="Where to write qrels.nanoknow-nq.supported.txt",
    )
    return parser.parse_args()


def load_results(pickle_path: Path) -> list[dict]:
    with pickle_path.open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict) or "results" not in data:
        raise ValueError(f'Expected a dict with a top-level "results" key: {pickle_path}')

    results = data["results"]
    if not isinstance(results, list):
        raise ValueError(f'"results" must be a list: {pickle_path}')
    if not results:
        raise ValueError(f'No rows found under "results" in {pickle_path}')
    return results


def load_questions(results: list[dict]) -> dict[str, str]:
    questions: dict[str, str] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        qid = item.get("id")
        question = item.get("original_question")
        if qid is None or question is None:
            continue
        questions[str(qid)] = str(question).replace("\t", " ").strip()

    if not questions:
        raise ValueError("No questions found in pickle results")
    return questions


def load_answers(results: list[dict]) -> dict[str, list[str]]:
    answers: dict[str, list[str]] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        qid = item.get("id")
        original_answers = item.get("original_answers")
        if qid is None or original_answers is None:
            continue
        if not isinstance(original_answers, list):
            raise ValueError(f"original_answers must be a list for qid {qid}")
        answers[str(qid)] = [str(answer) for answer in original_answers]

    if not answers:
        raise ValueError("No answers found in pickle results")
    return answers


def assert_question_matches_line(qid: str, line: str, questions: dict[str, str]) -> None:
    if qid not in questions:
        raise KeyError(f"Question text for qid {qid} not found in pickle results")

    qid_prefix = f"{qid},"
    assert line.startswith(qid_prefix), (
        f"Unexpected qid prefix for qid {qid}\n"
        f"Expected line to start with: {qid_prefix!r}\n"
        f"Actual line: {line!r}"
    )

    after_qid = line[len(qid_prefix) :].lstrip(" ")
    expected_question = questions[qid]
    assert after_qid.startswith(expected_question), (
        f"Question mismatch for qid {qid}\n"
        f"Expected question: {expected_question!r}\n"
        f"Actual line: {line!r}"
    )

    post_question = after_qid[len(expected_question) :]
    assert post_question.lstrip(" ").startswith(", "), (
        f"Unexpected formatting after question for qid {qid}\n"
        f"Expected whitespace followed by ', ' after the question text\n"
        f"Actual line: {line!r}"
    )


def iter_supported_pairs(supported_qrels: Path, questions: dict[str, str]):
    with supported_qrels.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                prefix, docid, _offset = line.rsplit(", ", 2)
                qid, _rest = prefix.split(", ", 1)
            except ValueError as exc:
                raise ValueError(f"Could not parse qrels line: {line}") from exc

            assert_question_matches_line(qid, line, questions)
            yield qid, docid


def iter_unsupported_qids(results: list[dict], questions: dict[str, str]):
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("contaminated") is not False:
            continue
        qid = item.get("id")
        if qid is None:
            continue
        qid_str = str(qid)
        if qid_str not in questions:
            raise KeyError(f"Question text for qid {qid_str} not found in pickle results")
        yield qid_str


def write_topics(questions: dict[str, str], qids_in_order: list[str], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for qid in qids_in_order:
            if qid in seen:
                continue
            if qid not in questions:
                raise KeyError(f"Question text for qid {qid} not found in pickle results")
            handle.write(f"{qid}\t{questions[qid]}\n")
            seen.add(qid)
            count += 1

    return count


def write_qrels(qid_docids: list[tuple[str, str]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for qid, docid in qid_docids:
            handle.write(f"{qid} Q0 {docid} 1\n")
    return len(qid_docids)


def write_answers(answers: dict[str, list[str]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for qid in sorted(answers, key=lambda value: int(value)):
            record = {"qid": qid, "answer": answers[qid]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def verify_topics_file(questions: dict[str, str], qids_in_order: list[str], topics_path: Path) -> int:
    expected_qids: list[str] = []
    seen_expected: set[str] = set()
    for qid in qids_in_order:
        if qid not in seen_expected:
            expected_qids.append(qid)
            seen_expected.add(qid)

    observed_qids: list[str] = []
    seen_observed: set[str] = set()

    with topics_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            parts = line.split("\t")
            assert len(parts) == 2, (
                f"Malformed topics row at line {line_number}: expected 2 TSV columns, got {len(parts)}"
            )

            qid, question = parts
            assert qid not in seen_observed, (
                f"Duplicate qid {qid} found in generated topics file at line {line_number}"
            )
            assert qid in questions, (
                f"Generated topics file contains qid {qid} not present in pickle results"
            )
            assert question == questions[qid], (
                f"Post-hoc topics verification failed for qid {qid}\n"
                f"Expected question from pickle: {questions[qid]!r}\n"
                f"Observed question in topics file: {question!r}"
            )

            seen_observed.add(qid)
            observed_qids.append(qid)

    assert observed_qids == expected_qids, (
        "Generated topics qids do not match the expected unique qids/order derived from the source"
    )
    return len(observed_qids)


def verify_qrels_file(qid_docids: list[tuple[str, str]], qrels_path: Path) -> int:
    observed_pairs: list[tuple[str, str]] = []

    with qrels_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\n")
            parts = line.split()
            assert len(parts) == 4, (
                f"Malformed qrels row at line {line_number}: expected 4 columns, got {len(parts)}"
            )

            qid, q0, docid, relevance = parts
            assert q0 == "Q0", (
                f"Malformed qrels row at line {line_number}: expected second column 'Q0', got {q0!r}"
            )
            assert relevance == "1", (
                f"Malformed qrels row at line {line_number}: expected relevance '1', got {relevance!r}"
            )
            observed_pairs.append((qid, docid))

    assert observed_pairs == qid_docids, (
        "Generated qrels do not exactly match the qid/docid pairs from nq_supported.txt"
    )
    return len(observed_pairs)


def verify_answers_file(answers: dict[str, list[str]], answers_path: Path) -> int:
    expected_qids = sorted(answers, key=lambda value: int(value))
    observed_qids: list[str] = []

    with answers_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            record = json.loads(raw_line)
            assert isinstance(record, dict), (
                f"Malformed answers row at line {line_number}: expected JSON object"
            )
            assert set(record.keys()) == {"qid", "answer"}, (
                f"Malformed answers row at line {line_number}: expected keys {{'qid', 'answer'}}"
            )

            qid = str(record["qid"])
            observed_answer = record["answer"]
            assert qid in answers, (
                f"Generated answers file contains qid {qid} not present in pickle results"
            )
            assert observed_answer == answers[qid], (
                f"Post-hoc answers verification failed for qid {qid}\n"
                f"Expected answers from pickle: {answers[qid]!r}\n"
                f"Observed answers in JSONL: {observed_answer!r}"
            )
            observed_qids.append(qid)

    assert observed_qids == expected_qids, (
        "Generated answers qids do not match the expected qids/order derived from pickle results"
    )
    return len(observed_qids)


def main() -> None:
    args = parse_args()

    results = load_results(args.pickle_path)
    questions = load_questions(results)
    answers = load_answers(results)

    qid_docids = list(iter_supported_pairs(args.supported_qrels, questions))
    supported_qids_in_order = [qid for qid, _docid in qid_docids]
    unsupported_qids_in_order = list(iter_unsupported_qids(results, questions))

    answer_count = write_answers(answers, args.answers_out)
    supported_topic_count = write_topics(
        questions, supported_qids_in_order, args.supported_topics_out
    )
    unsupported_topic_count = write_topics(
        questions, unsupported_qids_in_order, args.unsupported_topics_out
    )
    qrel_count = write_qrels(qid_docids, args.qrels_out)

    verified_answer_count = verify_answers_file(answers, args.answers_out)
    verified_supported_topic_count = verify_topics_file(
        questions, supported_qids_in_order, args.supported_topics_out
    )
    verified_unsupported_topic_count = verify_topics_file(
        questions, unsupported_qids_in_order, args.unsupported_topics_out
    )
    verified_qrel_count = verify_qrels_file(qid_docids, args.qrels_out)

    print(f"Wrote {answer_count} answers to {args.answers_out}")
    print(f"Wrote {supported_topic_count} supported topics to {args.supported_topics_out}")
    print(f"Wrote {unsupported_topic_count} unsupported topics to {args.unsupported_topics_out}")
    print(f"Wrote {qrel_count} qrels to {args.qrels_out}")
    print(
        f"Verified {verified_answer_count} generated answers against pickle answers in "
        f"{args.answers_out}"
    )
    print(
        f"Verified {verified_supported_topic_count} generated supported topics against pickle "
        f"questions in {args.supported_topics_out}"
    )
    print(
        f"Verified {verified_unsupported_topic_count} generated unsupported topics against "
        f"pickle questions in {args.unsupported_topics_out}"
    )
    print(
        f"Verified {verified_qrel_count} generated qrels against qid/docid pairs in "
        f"{args.supported_qrels}"
    )


if __name__ == "__main__":
    main()
