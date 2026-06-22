# Creating NQ and SQuAD Projections on ClimbMix-400B

This guide explains how to reproduce our projection of NQ and SQuAD onto ClimbMix-400B.

The projection pipeline was:

1. Stage 1: retrieve top-1000 documents from ClimbMix-400B with BM25 through the Pyserini REST API.
2. Stage 1: keep only documents with answer-string matches and store compact context snippets.
3. Stage 2: verify every matched document with `Qwen/Qwen3.5-9B`.
4. Merge Stage 2 shards.
5. Export NanoKnow topics, answers, and qrels.

## Setup

```bash
cd NanoKnow
mkdir -p output

export PYSERINI_API_TOKEN="<your-token>"
```

## Stage 1: NQ

```bash
python scripts/project.py \
  --dataset nq \
  --stage 1 \
  --retriever api \
  --index_path climbmix-400b \
  --top_k 1000 \
  --window_size 512 \
  --output output/nq_climbmix_stage1_top1000.pkl
```

Final Stage 1 NQ statistics:

- Dataset: `google-research-datasets/nq_open`, validation split
- Questions: `3,610`
- Retrieved per query: `top_k=1000`
- Queries with answer-string matches: `3,164`
- Matching documents: `613,845`
- Stored document text: context snippets only

## Stage 1: SQuAD

```bash
python scripts/project.py \
  --dataset squad \
  --stage 1 \
  --retriever api \
  --index_path climbmix-400b \
  --top_k 1000 \
  --window_size 512 \
  --output output/squad_climbmix_top1000_stage1.pkl
```

Final Stage 1 SQuAD statistics:

- Dataset: `rajpurkar/squad`, validation split
- Questions: `10,570`
- Retrieved per query: `top_k=1000`
- Queries with answer-string matches: `9,464`
- Matching documents: `1,441,408`
- Stored document text: context snippets only

## Stage 2: LLM Verification

Stage 2 used `Qwen/Qwen3.5-9B` as the LLM judge. We disabled early stopping so that every Stage 1 matched document was judged, not only the first verified document per question.

NQ was run in 3 shards:

```bash
for shard in 0 1 2; do
  python scripts/project.py \
    --stage 2 \
    --input output/nq_climbmix_stage1_top1000.pkl \
    --output "output/nq_climbmix_stage2_qwen35_9b_noearly.shard_${shard}_of_3.pkl" \
    --model Qwen/Qwen3.5-9B \
    --stage2_shard_count 3 \
    --stage2_shard_id "${shard}" \
    --no-stage2_early_stop
done
```

SQuAD was run in 6 shards:

```bash
for shard in 0 1 2 3 4 5; do
  python scripts/project.py \
    --stage 2 \
    --input output/squad_climbmix_top1000_stage1.pkl \
    --output "output/squad_climbmix_stage2_qwen35_9b_noearly.shard_${shard}_of_6.pkl" \
    --model Qwen/Qwen3.5-9B \
    --stage2_shard_count 6 \
    --stage2_shard_id "${shard}" \
    --no-stage2_early_stop
done
```

## Merge Stage 2 Shards

Merge NQ:

```bash
python scripts/merge_stage2_shards.py \
  --input output/nq_climbmix_stage1_top1000.pkl \
  --shards \
    output/nq_climbmix_stage2_qwen35_9b_noearly.shard_0_of_3.pkl \
    output/nq_climbmix_stage2_qwen35_9b_noearly.shard_1_of_3.pkl \
    output/nq_climbmix_stage2_qwen35_9b_noearly.shard_2_of_3.pkl \
  --output output/nq_climbmix_stage2_qwen35_9b_noearly.pkl
```

Merge SQuAD:

```bash
python scripts/merge_stage2_shards.py \
  --input output/squad_climbmix_top1000_stage1.pkl \
  --shards \
    output/squad_climbmix_stage2_qwen35_9b_noearly.shard_0_of_6.pkl \
    output/squad_climbmix_stage2_qwen35_9b_noearly.shard_1_of_6.pkl \
    output/squad_climbmix_stage2_qwen35_9b_noearly.shard_2_of_6.pkl \
    output/squad_climbmix_stage2_qwen35_9b_noearly.shard_3_of_6.pkl \
    output/squad_climbmix_stage2_qwen35_9b_noearly.shard_4_of_6.pkl \
    output/squad_climbmix_stage2_qwen35_9b_noearly.shard_5_of_6.pkl \
  --output output/squad_climbmix_stage2_qwen35_9b_noearly.pkl
```

Final Stage 2 statistics:

| Dataset | Questions | Stage 1 answer-string matches | LLM-verified questions | Verified docs |
|---------|-----------|--------------------------------|------------------------|---------------|
| NQ-Open | 3,610 | 3,164 | 3,021 | 323,986 |
| SQuAD | 10,570 | 9,464 | 9,071 | 572,213 |

## Export NanoKnow Files

Export NQ:

```bash
python scripts/export_stage2_qrels.py \
  --input output/nq_climbmix_stage2_qwen35_9b_noearly.pkl \
  --dataset nq \
  --corpus climbmix \
  --output-dir questions-and-qrels/nq \
  --overwrite
```

Export SQuAD:

```bash
python scripts/export_stage2_qrels.py \
  --input output/squad_climbmix_stage2_qwen35_9b_noearly.pkl \
  --dataset squad \
  --corpus climbmix \
  --output-dir questions-and-qrels/squad \
  --overwrite
```

Final exported files:

```text
questions-and-qrels/nq/answers.nanoknow-nq.jsonl
questions-and-qrels/nq/climbmix/topics.nanoknow-nq-climbmix.supported.tsv
questions-and-qrels/nq/climbmix/topics.nanoknow-nq-climbmix.unsupported.tsv
questions-and-qrels/nq/climbmix/qrels.nanoknow-nq-climbmix.supported.txt

questions-and-qrels/squad/answers.nanoknow-squad.jsonl
questions-and-qrels/squad/climbmix/topics.nanoknow-squad-climbmix.supported.tsv
questions-and-qrels/squad/climbmix/topics.nanoknow-squad-climbmix.unsupported.tsv
questions-and-qrels/squad/climbmix/qrels.nanoknow-squad-climbmix.supported.txt
```

Final exported counts:

| Dataset | Answers | Supported topics | Unsupported topics | Qrels |
|---------|---------|------------------|--------------------|-------|
| NQ-Open | 3,610 | 3,021 | 589 | 323,986 |
| SQuAD | 10,570 | 9,071 | 1,499 | 572,213 |
