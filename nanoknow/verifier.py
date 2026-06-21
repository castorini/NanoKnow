"""
Stage 2: LLM-based verification of answer string matches.

Uses an LLM judge to filter out coincidental matches—e.g., "Paris" appearing
in a passage about Paris, Texas rather than Paris, France.
"""

from typing import Dict, List


class LLMVerifier:
    """Verify whether a document genuinely answers a question using an LLM."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
    ):
        import os

        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        print(f"Loading LLM verifier: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=1,
            trust_remote_code=True,
            max_model_len=max_model_len,
            dtype="bfloat16",
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )
        self.sampling_params = SamplingParams(
            temperature=0,
            max_tokens=512,
            skip_special_tokens=True,
        )
        self.model_name = model_name

    def _format_prompt(
        self, question: str, answers: List[str], doc_text: str
    ) -> str:
        answers_str = ", ".join(f'"{a}"' for a in answers[:5])

        chat = [
            {
                "role": "system",
                "content": (
                    "You verify whether answer knowledge exists in text. "
                    "Check if test Q&A pairs appear in the pre-training corpus. "
                    "Be STRICT. Only mark TRUE if the context DIRECTLY answers "
                    "the question."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION: {question}\n"
                    f"EXPECTED ANSWERS: {answers_str}\n\n"
                    f"DOCUMENT:\n{doc_text}\n\n"
                    "Does this document contain information that directly "
                    "answers the question?\n"
                    "Reply TRUE: [reason] or COINCIDENTAL: [reason]"
                ),
            },
        ]
        try:
            return self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )

    def _generate(self, prompt: str) -> str:
        return self._generate_many([prompt])[0]

    def _generate_many(self, prompts: List[str]) -> List[str]:
        outputs = self.model.generate(prompts, self.sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]

    def _verified_doc_result(self, doc: Dict, doc_index: int, response: str) -> Dict:
        return {
            "verified_at_doc": doc_index + 1,
            "doc_id": doc["doc_id"],
            "matched_answer": doc["matched_answer"],
            "match_position": doc["match_position"],
            "total_words": doc["total_words"],
            "context_snippet": doc["context_snippet"],
            "reason": response.strip(),
        }

    def verify(
        self,
        question: str,
        answers: List[str],
        matching_docs: List[Dict],
        early_stop: bool = True,
    ) -> Dict:
        """Verify has-answer documents with the LLM judge.

        Iterates through matching_docs and verifies documents with an LLM judge.

        Returns:
            Dict with 'verified' bool and verified doc metadata. With early_stop,
            returns after the first confirmed match. Otherwise verifies all docs.
        """
        if not matching_docs:
            return {
                "verified": False,
                "docs_checked": 0,
                "verified_docs": [],
            }

        verified_docs = []
        prompts = [
            self._format_prompt(question, answers, doc.get("context_snippet", ""))
            for doc in matching_docs
        ]

        if early_stop:
            responses = []
            response_iter = None
        else:
            responses = self._generate_many(prompts)
            response_iter = iter(responses)

        for i, doc in enumerate(matching_docs):
            if early_stop:
                response = self._generate(prompts[i])
            else:
                response = next(response_iter)

            # Strip thinking tags if present
            if "</think>" in response:
                response = response.split("</think>")[-1]

            clean = response.strip().upper()
            if clean.startswith("TRUE") or clean.startswith("YES"):
                verified_doc = self._verified_doc_result(doc, i, response)
                verified_docs.append(verified_doc)

                if early_stop:
                    return {
                        "verified": True,
                        "docs_checked": i + 1,
                        "verified_docs": verified_docs,
                        **verified_doc,
                    }

        if verified_docs:
            return {
                "verified": True,
                "docs_checked": len(matching_docs),
                "verified_docs": verified_docs,
                **verified_docs[0],
            }

        return {
            "verified": False,
            "docs_checked": len(matching_docs),
            "verified_docs": [],
        }
