# NanoKnow

**Project QA benchmarks onto LLM pre-training corpora.**

NanoKnow identifies which benchmark questions have answers in a model's training data, enabling controlled studies of parametric knowledge vs. retrieval-augmented generation (RAG).

🎉 NanoKnow was accepted to SIGIR '26!

Arxiv: https://arxiv.org/abs/2602.20122

## Overview

Given a QA benchmark and a pre-training corpus, NanoKnow produces **relevance judgments (qrels)** that partition questions into:

- **Supported**: The answer exists in the training data (the model could have memorized it).
- **Unsupported**: The answer does not appear in the training data.

The pipeline has three stages:

1. **BM25 Retrieval** — Search the corpus for candidate documents using the question as a query.
2. **Answer String Matching** — Filter to documents that contain the gold answer as a substring.
3. **LLM Verification** — Use an LLM judge to filter out coincidental matches (e.g., "Paris" in a passage about Paris, Texas).

## Pre-built Qrels

We provide pre-built qrels for [nanochat](https://github.com/karpathy/nanochat) models trained on [karpathy/fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle):

| Dataset | Questions | Supported | Unsupported |
|---------|-----------|-----------|---------------|
| SQuAD   | 10,570    | 7,490 (71%) | 3,080 (29%) |
| NQ-Open | 3,610     | 2,389 (66%) | 1,221 (34%) |

The pre-built files are organized by dataset under `questions-and-qrels/`:

```text
questions-and-qrels/
├── nq/
│   ├── answers.nanoknow-nq.jsonl
│   ├── qrels.nanoknow-nq.supported.txt
│   ├── topics.nanoknow-nq.supported.tsv
│   └── topics.nanoknow-nq.unsupported.tsv
└── squad/
    ├── answers.nanoknow-squad.jsonl
    ├── qrels.nanoknow-squad.supported.txt
    ├── topics.nanoknow-squad.supported.tsv
    └── topics.nanoknow-squad.unsupported.tsv
```

Each dataset directory contains:

- `topics.nanoknow-<dataset>.supported.tsv`: supported questions as `qid<TAB>question`.
- `topics.nanoknow-<dataset>.unsupported.tsv`: unsupported questions as `qid<TAB>question`.
- `answers.nanoknow-<dataset>.jsonl`: gold answers as one JSON object per line, e.g. `{"qid": "0", "answer": ["14 December 1972 UTC", "December 1972"]}`.
- `qrels.nanoknow-<dataset>.supported.txt`: TREC-format qrels for supported questions, e.g. `0 Q0 shard_01177_50695 1`.

## Installation

```bash
pip install -r requirements.txt
```

For BM25 retrieval, you also need Java 11+:
```bash
# Ubuntu/Debian
sudo apt install openjdk-11-jdk
```

## FineWeb-Edu Lucene Index

We release a pre-built Lucene index over [karpathy/fineweb-edu-100b-shuffle](https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle) (326 GB):

**Download**: [LingweiGu/NanoKnow-Fineweb-Edu-Index](https://huggingface.co/datasets/LingweiGu/NanoKnow-Fineweb-Edu-Index)

```bash
huggingface-cli download LingweiGu/NanoKnow-Fineweb-Edu-Index --repo-type dataset --local-dir ./fineweb-edu-index
```

To build the index yourself using [Anserini](https://github.com/castorini/anserini):

```bash
bin/run.sh io.anserini.index.IndexCollection \
  -collection FinewebCollection \
  -input /path/to/corpus \
  -index /output/directory \
  -generator DefaultLuceneDocumentGenerator \
  -threads 16
```

## Usage

### Project a new benchmark

```bash
# Stage 1: BM25 retrieval + answer matching (CPU only)
python scripts/project.py \
    --dataset squad \
    --stage 1 \
    --index_path /path/to/lucene-index \
    --output output/squad_stage1.pkl

# Stage 2: LLM verification (requires GPU)
python scripts/project.py \
    --stage 2 \
    --input output/squad_stage1.pkl \
    --output output/squad_stage2.pkl

# Or run both stages together
python scripts/project.py \
    --dataset squad \
    --stage both \
    --index_path /path/to/lucene-index \
    --output output/squad_projected.pkl
```

### Evaluate a nanochat checkpoint

#### Step 1: Run inference with a checkpoint

```bash
NANOCHAT_DIR="${NANOCHAT_DIR:-/path/to/nanochat}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/path/to/nanochat-checkpoint}"
STEP="${STEP:?Set STEP to the checkpoint step to evaluate}"
DATASET="${DATASET:-nq}"
QRELS_DIR="${QRELS_DIR:-questions-and-qrels/${DATASET}}"
FINEWEB_INDEX_PATH="${FINEWEB_INDEX_PATH:-/path/to/fineweb-index}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
DEVICE="${DEVICE:-cuda}"

python scripts/nanochat_inference.py \
    --nanochat-dir "${NANOCHAT_DIR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --step "${STEP}" \
    --dataset "${DATASET}" \
    --qrels_dir "${QRELS_DIR}" \
    --fineweb_index_path "${FINEWEB_INDEX_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --device "${DEVICE}"
```

If nanochat is already importable in your Python environment, `--nanochat-dir` can be omitted. For repeated runs, you can set `NANOCHAT_DIR=/path/to/nanochat` instead of passing the argument each time.

The output file is named from the checkpoint directory basename and dataset, for example `output/karpathy_nanochat_d32_nq`.

#### Step 2: Score predictions with the evaluator

```bash
python scripts/evaluate_model_predictions.py \
    --input_file output/karpathy_nanochat_d32_nq \
    --output_file nanochat_evaluations/karpathy_nanochat_d32_nq_scored.pkl
```

#### Step 3: Summarize evaluation scores

```bash
python scripts/get_eval_scores.py nanochat_evaluations/karpathy_nanochat_d32_nq_scored.pkl
```

Example output:

```json
{
  "contaminated_fineweb_context": {
    "count": 2389,
    "exact_match_accuracy": 0.46923398911678527,
    "llm_judge_accuracy": 0.46672247802427796
  },
  "contaminated_no_context": {
    "count": 2389,
    "exact_match_accuracy": 0.19589786521557137,
    "llm_judge_accuracy": 0.2293846797823357
  },
  "non_contaminated_no_context": {
    "count": 1221,
    "exact_match_accuracy": 0.00819000819000819,
    "llm_judge_accuracy": 0.04914004914004914
  }
}
```

Replication check: on June 1, 2026, we re-ran the NQ evaluation for
`karpathy_nanochat_d32` at step `650`. Inference completed in
`36m56s`, scoring completed in `25m40s`, and the reproduced scores were:

```json
{
  "contaminated_fineweb_context": {
    "count": 2389,
    "exact_match_accuracy": 0.4679782335705316,
    "llm_judge_accuracy": 0.47216408539137716
  },
  "contaminated_no_context": {
    "count": 2389,
    "exact_match_accuracy": 0.19715362076182502,
    "llm_judge_accuracy": 0.22980326496442027
  },
  "non_contaminated_no_context": {
    "count": 1221,
    "exact_match_accuracy": 0.00819000819000819,
    "llm_judge_accuracy": 0.04504504504504504
  }
}
```

## Repository Structure

```
NanoKnow/
├── nanoknow/                  # Core library
│   ├── retriever.py           # Stage 1: BM25 retrieval + answer matching
│   ├── verifier.py            # Stage 2: LLM-based verification
│   └── evaluator.py           # Evaluation utilities
├── scripts/                   # Runnable scripts
│   ├── project.py             # Run the projection pipeline
│   ├── nanochat_inference.py  # Run nanochat checkpoint inference
│   ├── evaluate_model_predictions.py  # Score predictions
│   └── get_eval_scores.py     # Summarize scored evaluation results
├── questions-and-qrels/       # Pre-built benchmark questions, answers, and qrels
│   ├── nq/
│   │   ├── answers.nanoknow-nq.jsonl
│   │   ├── qrels.nanoknow-nq.supported.txt
│   │   ├── topics.nanoknow-nq.supported.tsv
│   │   └── topics.nanoknow-nq.unsupported.tsv
│   └── squad/
│       ├── answers.nanoknow-squad.jsonl
│       ├── qrels.nanoknow-squad.supported.txt
│       ├── topics.nanoknow-squad.supported.tsv
│       └── topics.nanoknow-squad.unsupported.tsv
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## nanochat Checkpoints

We evaluated eight checkpoints across three model scales:

| Scale | Checkpoints |
|-------|------------|
| d20 (~561M params) | `sampathchanda/nanochat-d20`, `shu127/nanochat-d20`, `pankajmathur/nanochat-d20` |
| d32 (~1.9B params) | `karpathy/nanochat-d32`, `Antigma/nanochat-d32` |
| d34 (~2.2B params) | `renatocastro33/nanochat-d34-sft`, `victoremnm/nanochat-d34-sft`, `pankajmathur/nanochat-d34-sft-hf` |

## Citation

```bibtex
@article{gu2026nanoknow,
  title={NanoKnow: How to Know What Your Language Model Knows},
  author={Gu, Lingwei and Jedidi, Nour and Lin, Jimmy},
  journal={arXiv preprint arXiv:2602.20122},
  year={2026}
}
```

## License

Apache 2.0
