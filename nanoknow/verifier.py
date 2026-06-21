"""
Stage 2: LLM-based verification of answer string matches.

Uses an LLM judge to filter out coincidental matches—e.g., "Paris" appearing
in a passage about Paris, Texas rather than Paris, France.
"""

from typing import Dict, List


class LLMVerifier:
    """Verify whether a document genuinely answers a question using an LLM."""

    def __init__(self, model_name: str = "Qwen/Qwen3-8B"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading LLM verifier: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self.model_name = model_name

    def _format_prompt(
        self, question: str, answers: List[str], doc_text: str
    ) -> str:
        answers_str = ", ".join(f'"{a}"' for a in answers[:5])

        # Smart truncation: keep passage around the answer
        if len(doc_text) > 2500:
            doc_lower = doc_text.lower()
            answer_pos, matched = -1, None
            for ans in answers:
                pos = doc_lower.find(ans.lower())
                if pos != -1:
                    answer_pos, matched = pos, ans
                    break

            if answer_pos > 1200 and matched:
                head = doc_text[:800]
                start = max(0, answer_pos - 400)
                end = min(len(doc_text), answer_pos + len(matched) + 400)
                doc_text = head + "\n[...]\n" + doc_text[start:end] + "..."
            else:
                doc_text = doc_text[:2500] + "..."

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
        import torch

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[1] :], skip_special_tokens=True
        )
        return response.strip()

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
        verified_docs = []
        docs_checked = 0

        for i, doc in enumerate(matching_docs):
            docs_checked = i + 1
            text = doc.get("context_snippet", "")
            prompt = self._format_prompt(question, answers, text)
            response = self._generate(prompt)

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
                        "docs_checked": docs_checked,
                        "verified_docs": verified_docs,
                        **verified_doc,
                    }

        if verified_docs:
            return {
                "verified": True,
                "docs_checked": docs_checked,
                "verified_docs": verified_docs,
                **verified_docs[0],
            }

        return {
            "verified": False,
            "docs_checked": docs_checked,
            "verified_docs": [],
        }
