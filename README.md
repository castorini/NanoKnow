# NanoKnow

**Project QA benchmarks onto LLM pre-training corpora.**

NanoKnow identifies which benchmark questions have answers in a model's training data, enabling controlled studies of parametric knowledge vs. retrieval-augmented generation (RAG).

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

Qrels are in `qrels/` in CSV format:
```
qid, question, answer, doc_id, answer_offset
```

## Installation

```bash
pip install -r requirements.txt
```

For BM25 retrieval, you also need Java 11+:
```bash
# Ubuntu/Debian
sudo apt install openjdk-11-jdk
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

```bash
python scripts/evaluate.py \
    --model pankajmathur/nanochat-d34-sft-hf \
    --projection output/squad_stage2.pkl \
    --dataset squad \
    --fineweb_path /path/to/fineweb-edu-100b-shuffle \
    --output output/eval_results.pkl
```

To evaluate the model predictions, pass ```output/eval_results.pkl``` to the following script:

```bash
python scripts/evaluate_model_predictions.py \
    --input_file output/eval_results.pkl 
```

This will produce a ```eval_results_scored.pkl``` file which produces a binary evaluation for each of
the model's predictions for a given condition (e.g., closed-book). To get an overall accuracy score, 
simply take the sum of the true values across all predictions, for example as follows:
```
def clean_json_output(text):
    if "</think>" in text:
        text = text.split("</think>")[-1]

    # 2. Strip markdown code blocks (```json ... ```) and whitespace
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE)

    # 3. Return the parsed dict directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"correct": False, "explanation": "JSON_PARSE_ERROR"}

# To provide an example, we set condition_name to be "supported_no_context"
# Please replace with the condition you are interested in evaluating
condition_name = "supported_no_context"
predictions = data[condition_name]["results"]
n = len(predictions)
em_acc = sum(d['exact_match_score'] for d in predictions) / n
judge_acc = sum(
        clean_json_output(d['llm_judge_score'])['correct']
        for d in predictions
    ) / n
```

### Analyze frequency effects

```bash
python scripts/analyze_frequency.py \
    --eval_dir output/ \
    --qrel_file qrels/squad_supported.txt \
    --dataset squad \
    --output output/frequency_analysis.json
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

## Repository Structure

```
NanoKnow/
├── nanoknow/                  # Core library
│   ├── retriever.py           # Stage 1: BM25 retrieval + answer matching
│   ├── verifier.py            # Stage 2: LLM-based verification
│   └── evaluator.py           # Nanochat evaluation utilities
├── scripts/                   # Runnable scripts
│   ├── project.py             # Run the projection pipeline
│   ├── evaluate.py            # Evaluate nanochat checkpoints
│   └── analyze_frequency.py   # Frequency analysis
├── qrels/                     # Pre-built relevance judgments
│   ├── squad_supported.txt    # SQuAD supported (7,490 questions)
│   ├── squad_unsupported.txt  # SQuAD unsupported (3,080 questions)
│   ├── nq_supported.txt       # NQ supported (2,389 questions)
│   └── nq_unsupported.txt      # NQ unsupported (1,221 questions)
├── requirements.txt
├── LICENSE
└── README.md
```

## Nanochat Checkpoints

We evaluated eight checkpoints across three model scales:

| Scale | Checkpoints |
|-------|------------|
| d20 (~561M params) | `sampathchanda/nanochat-d20`, `shu127/nanochat-d20`, `pankajmathur/nanochat-d20` |
| d32 (~1B params) | `karpathy/nanochat-d32`, `Antigma/nanochat-d32` |
| d34-sft (~1.4B params) | `renatocastro33/nanochat-d34-sft`, `victoremnm/nanochat-d34-sft`, `pankajmathur/nanochat-d34-sft-hf` |

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
