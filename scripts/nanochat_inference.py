import json
import os 
import argparse
import pickle
import random
import torch
from tqdm import tqdm

import sys
# from nanochat.common import autodetect_device_type

from pyserini.eval.evaluate_dpr_retrieval import has_answers, SimpleTokenizer, _normalize
from pyserini.search.lucene import LuceneSearcher

random.seed(42)


def resolve_nanochat_build_model(nanochat_dir=None):
    nanochat_path = nanochat_dir or os.environ.get("NANOCHAT_DIR")

    if nanochat_path:
        nanochat_path = os.path.abspath(os.path.expanduser(nanochat_path))
        if not os.path.isdir(nanochat_path):
            raise FileNotFoundError(
                f"nanochat directory not found: {nanochat_path}. "
                "Pass --nanochat-dir /path/to/nanochat or set NANOCHAT_DIR."
            )
        import_root = nanochat_path
        if os.path.exists(os.path.join(nanochat_path, "checkpoint_manager.py")):
            import_root = os.path.dirname(nanochat_path)
        if import_root not in sys.path:
            sys.path.insert(0, import_root)

    try:
        from nanochat.checkpoint_manager import build_model
    except ImportError as exc:
        setup_hint = (
            "Could not import nanochat. Install nanochat in this Python "
            "environment, pass --nanochat-dir /path/to/nanochat, or set "
            "NANOCHAT_DIR=/path/to/nanochat."
        )
        if nanochat_path:
            setup_hint += f" Tried nanochat directory: {nanochat_path}."
        raise ImportError(setup_hint) from exc

    return build_model


def load_text_row(docid, searcher):
    doc = searcher.doc(docid)
    if doc is None:
        raise ValueError(f"Document not found in FineWeb index: {docid}")
    return json.loads(doc.raw())["text"]


def load_unified_results_from_qrels(qrels_dir: str, dataset: str) -> list[dict]:
    qrels_path = os.path.join(qrels_dir, f"qrels.nanoknow-{dataset}-fineweb.supported.txt")
    answers_path = os.path.join(qrels_dir, f"answers.nanoknow-{dataset}.jsonl")
    supported_topics_path = os.path.join(qrels_dir, f"topics.nanoknow-{dataset}-fineweb.supported.tsv")
    unsupported_topics_path = os.path.join(qrels_dir, f"topics.nanoknow-{dataset}-fineweb.unsupported.tsv")

    for required_path in [qrels_path, answers_path, supported_topics_path, unsupported_topics_path]:
        if not os.path.exists(required_path):
            raise FileNotFoundError(f"Required input file not found: {required_path}")

    def load_topics(path):
        topics = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.rstrip("\n")
                if not line:
                    continue
                qid, question = line.split("\t", 1)
                topics[str(qid)] = question
        return topics

    def load_answers(path):
        answers = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                qid = str(record["qid"])
                answer_list = record["answer"]
                if not isinstance(answer_list, list):
                    raise ValueError(f"Expected answer list for qid {qid}")
                answers[qid] = [str(answer) for answer in answer_list]
        return answers

    def load_qrels(path):
        qrels = {}
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                qid, _q0, docid, _score = line.split()
                qrels.setdefault(str(qid), []).append(docid)
        return qrels

    supported_topics = load_topics(supported_topics_path)
    unsupported_topics = load_topics(unsupported_topics_path)
    answers = load_answers(answers_path)
    supported_qrels = load_qrels(qrels_path)

    overlap = set(supported_topics) & set(unsupported_topics)
    if overlap:
        raise ValueError(f"Found qids in both supported and unsupported topics: {sorted(overlap)[:5]}")

    missing_supported_answers = sorted(set(supported_topics) - set(answers))
    if missing_supported_answers:
        raise ValueError(f"Missing answers for supported qids: {missing_supported_answers[:5]}")

    missing_unsupported_answers = sorted(set(unsupported_topics) - set(answers))
    if missing_unsupported_answers:
        raise ValueError(f"Missing answers for unsupported qids: {missing_unsupported_answers[:5]}")

    missing_supported_qrels = sorted(set(supported_topics) - set(supported_qrels))
    if missing_supported_qrels:
        raise ValueError(f"Missing qrels for supported qids: {missing_supported_qrels[:5]}")

    extra_supported_qrels = sorted(set(supported_qrels) - set(supported_topics))
    if extra_supported_qrels:
        raise ValueError(f"Found qrels for qids not present in supported topics: {extra_supported_qrels[:5]}")

    unified_results = []

    for qid, question in supported_topics.items():
        unified_results.append({
            "id": qid,
            "original_question": question,
            "original_answers": answers[qid],
            "contaminated": True,
            "doc_ids": supported_qrels[qid],
        })

    for qid, question in unsupported_topics.items():
        unified_results.append({
            "id": qid,
            "original_question": question,
            "original_answers": answers[qid],
            "contaminated": False,
            "doc_ids": [],
        })

    unified_results.sort(key=lambda item: int(item["id"]))
    return unified_results

def load_inference_engine(checkpoint_dir, step, device, nanochat_dir=None):
    build_model = resolve_nanochat_build_model(nanochat_dir)
    device_obj = torch.device(device)
    model, tokenizer, _ = build_model(
        checkpoint_dir=checkpoint_dir,
        phase='eval',
        step=step,
        device=device_obj
    )
    return model, tokenizer, device_obj

def generate(model, tokenizer, prompt, max_tokens=64):
    kwargs = dict(max_tokens=max_tokens, temperature=0, seed=random.randint(0, 2**31 - 1))
    conversation = {"messages": [{"role": "user", "content": prompt}]}
    prompt_tokens, _ = tokenizer.render_conversation(conversation)
    
    generated_tokens = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    stream = model.generate(prompt_tokens, **kwargs)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for token in stream:
            generated_tokens.append(token)
            if token == tokenizer.encode_special("<|assistant_end|>"):
                stream.close()   
                break
                
    return tokenizer.decode(generated_tokens)

def extract_surrounding_text(text, answers, window_size=200):
    """
    Extract surrounding context around the first matched answer,
    limited to window_size words total (≈ window_size/2 on each side).
    """
    if not text:
        return ""

    text_lower = text.lower()
    words = text.split()

    for ans in answers:
        if not ans:
            continue

        ans_lower = ans.lower()
        pos = text_lower.find(ans_lower)
        if pos == -1:
            continue

        # Find word index corresponding to character position
        char_count = 0
        word_idx = 0
        for i, w in enumerate(words):
            if char_count >= pos:
                word_idx = i
                break
            char_count += len(w) + 1  # +1 for space

        half = window_size // 2
        start = max(0, word_idx - half)
        end = min(len(words), word_idx + half)

        return " ".join(words[start:end])

    return ""


def format_prompt(question: str, context: str = None) -> str:
    if context:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    else:
        prompt = f"Question: {question}\n\nAnswer:"
    return prompt

def run_generation(model, tokenizer, question, context=None):
    # Returns both the prompt string and the generated text
    prompt = format_prompt(question=question, context=context)
    prediction = generate(model, tokenizer, prompt=prompt)
    return prompt, prediction

def run_zero_shot(model, tokenizer, results, use_contaminated=True):
    data = results
    outputs = []

    for result in tqdm(data, desc="Zero-shot eval"):
        # ---- only change is this line ----
        if result.get("contaminated") != use_contaminated:
            continue

        question = result["original_question"]
        prompt, prediction = run_generation(model, tokenizer, question, context=None)

        outputs.append({
            "mode": "zs",
            "question": question,
            "prompt": prompt,
            "answers": result["original_answers"],
            "prediction": prediction,
            "context_length": 0
        })

    return outputs

def run_rag(model, tokenizer, results, fineweb_searcher):
    data = results
    outputs = []

    for result in tqdm(data, desc="RAG eval"):
        if not result.get("contaminated"):
            continue

        doc_ids = result.get("doc_ids", [])
        if not doc_ids:
            continue

        answer_passage = ""
        context = ""
        for docid in doc_ids:
            answer_passage = load_text_row(docid, fineweb_searcher)
            context = extract_surrounding_text(
                answer_passage,
                answers=result["original_answers"]
            )
            if context:
                break

        if not context:
            context = answer_passage

        question = result["original_question"]
        prompt, prediction = run_generation(model, tokenizer, question, context=context)

        outputs.append({
            "mode": "rag_fineweb_context",
            "question": question,
            "prompt": prompt,
            "answers": result["original_answers"],
            "prediction": prediction,
            "context_length": len(context) if context else 0
        })

    return outputs

from datasets import load_dataset

def run_rag_original_context(model, tokenizer, results, use_contaminated=True):
    print("Loading SQuAD validation set...")
    squad_val = load_dataset("rajpurkar/squad", split="validation")

    data = results
    outputs = []

    for result in tqdm(data, desc="RAG (gold context) eval"):
        # ---- only change is this line ----
        if result.get("contaminated") != use_contaminated:
            continue

        qid = int(result["id"])
        squad_example = squad_val[qid]
        question = result["original_question"]
        # assert squad_example["question"] == question, (
        #     f"SQuAD question mismatch for qid {qid}: "
        #     f"{squad_example['question']!r} != {question!r}"
        # )

        squad_answers = squad_example["answers"]["text"]
        # assert squad_answers == result["original_answers"], (
        #     f"SQuAD answers mismatch for qid {qid}: "
        #     f"{squad_answers!r} != {result['original_answers']!r}"
        # )

        context = squad_example["context"]

        prompt, prediction = run_generation(model, tokenizer, question, context=context)
        outputs.append({
            "mode": "rag_gold_context",
            "question": question,
            "prompt": prompt,
            "answers": result["original_answers"],
            "prediction": prediction,
            "context_length": len(context)
        })

    return outputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--dataset", type=str, choices=["squad", "nq"], default="nq")
    parser.add_argument("--qrels_dir", type=str, required=True)
    parser.add_argument("--fineweb_index_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--nanochat-dir",
        type=str,
        default=None,
        help="Path to a local nanochat checkout. Can also be set with NANOCHAT_DIR.",
    )
    args = parser.parse_args()

    # Load Model
    model, tokenizer, device = load_inference_engine(
        args.checkpoint_dir,
        args.step,
        args.device,
        nanochat_dir=args.nanochat_dir,
    )
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{os.path.basename(args.checkpoint_dir.rstrip('/'))}_{args.dataset}"
    unified_results = load_unified_results_from_qrels(args.qrels_dir, args.dataset)
    print(f"Loading FineWeb index from {args.fineweb_index_path}...")
    fineweb_searcher = LuceneSearcher(args.fineweb_index_path)
    print(f"FineWeb index loaded ({fineweb_searcher.num_docs:,} documents)")
    if args.dataset == "squad":
        all_results = {
            "supported_closed_book": {},
            "supported_w_fineweb_context": {},
            "supported_w_original_context": {},
            "unsupported_closed_book": {},
            "unsupported_w_original_context": {},
        }

        all_results["supported_closed_book"]["results"] = run_zero_shot(
                model, tokenizer, unified_results, use_contaminated=True
            )

        all_results["unsupported_closed_book"]["results"] = run_zero_shot(
                model, tokenizer, unified_results, use_contaminated=False
            )

        all_results["supported_w_fineweb_context"]["results"] = run_rag(
                model, tokenizer, unified_results, fineweb_searcher
            )

        all_results["supported_w_original_context"]["results"] = run_rag_original_context(
                model, tokenizer, unified_results, use_contaminated=True
            )

        all_results["unsupported_w_original_context"]["results"] = run_rag_original_context(
                model, tokenizer, unified_results, use_contaminated=False
            )

        with open(output_path, "wb") as f:
            pickle.dump(all_results, f)
    else:
        all_results = {
            "supported_closed_book": {},
            "supported_w_fineweb_context": {},
            "unsupported_closed_book": {}
        }

        all_results["supported_closed_book"]["results"] = run_zero_shot(
            model, tokenizer, unified_results
        )

        all_results["unsupported_closed_book"]["results"] = run_zero_shot(
                model, tokenizer, unified_results, use_contaminated=False
            )

        all_results["supported_w_fineweb_context"]["results"] = run_rag(
            model, tokenizer, unified_results, fineweb_searcher
        )

        with open(output_path, "wb") as f:
            pickle.dump(all_results, f)
