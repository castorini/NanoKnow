"""
Evaluate nanochat checkpoints on supported / unsupported questions.

Supports four conditions:
  1. Closed-book (no context)
  2. w/ FineWeb context (from pre-training data)
  3. w/ Original context (SQuAD only)
  4. Unsupported + Original context (SQuAD only)
"""

import re
from typing import Dict, List, Optional


class NanoChatEvaluator:
    """Load a nanochat checkpoint and generate answers."""

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def format_prompt(
        self, question: str, context: Optional[str] = None
    ) -> str:
        if context:
            return f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return f"Question: {question}\n\nAnswer:"

    def generate(self, prompt: str, max_new_tokens: int = 64) -> str:
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        return response.strip()

    @staticmethod
    def normalize(text: str) -> str:
        """Normalize text for exact-match comparison."""
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", "", text)
        return " ".join(text.split())

    def exact_match(self, prediction: str, gold_answers: List[str]) -> bool:
        """Check if prediction matches any gold answer (substring match)."""
        pred = self.normalize(prediction)
        for gold in gold_answers:
            g = self.normalize(gold)
            if g in pred or pred in g:
                return True
        return False


def get_fineweb_context(
    doc_id: str,
    matched_answer: str,
    fineweb_path: str,
    max_words: int = 200,
) -> Optional[str]:
    """Extract a context window from a FineWeb-Edu parquet shard.

    Args:
        doc_id: Document ID in format ``shard_XXXXX_YYYYY``.
        matched_answer: The answer string to center the window on.
        fineweb_path: Base path to the FineWeb-Edu parquet shards.
        max_words: Maximum number of words in the context window.
    """
    try:
        import duckdb

        parts = doc_id.split("_")
        shard_num = int(parts[1])
        row_idx = int(parts[2])
        parquet_file = f"{fineweb_path}/shard_{shard_num:05d}.parquet"

        con = duckdb.connect()
        query = f"SELECT text FROM read_parquet('{parquet_file}') LIMIT 1 OFFSET {row_idx}"
        result = con.execute(query).fetchone()
        con.close()

        if not result:
            return None

        text = result[0]
        pos = text.lower().find(matched_answer.lower())
        if pos == -1:
            return None

        words = text.split()
        char_count, word_idx = 0, 0
        for i, w in enumerate(words):
            if char_count >= pos:
                word_idx = i
                break
            char_count += len(w) + 1

        half = max_words // 2
        start = max(0, word_idx - half)
        end = min(len(words), word_idx + half)
        return " ".join(words[start:end])
    except Exception as e:
        print(f"Error getting FineWeb context for {doc_id}: {e}")
        return None
