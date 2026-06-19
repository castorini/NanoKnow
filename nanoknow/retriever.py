"""
Stage 1: BM25 retrieval + answer string matching.

Given a question and its gold answers, retrieve candidate documents from
a Lucene index and check which ones contain the answer string.
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from typing import Dict, List, Optional


def extract_context(
    full_text: str,
    answers: List[str],
    window_size: int = 256,
) -> Optional[Dict]:
    """Extract a context window around the first answer match in the text.

    Args:
        full_text: The full document text.
        answers: List of gold answer strings to search for.
        window_size: Total number of words to include in the context window.

    Returns:
        Dict with context, matched_answer, match_position, total_words,
        or None if no answer is found.
    """
    text_lower = full_text.lower()
    words = full_text.split()

    for answer in answers:
        pos = text_lower.find(answer.lower())
        if pos != -1:
            # Map character offset to word index
            char_count = 0
            word_idx = 0
            for i, w in enumerate(words):
                char_count += len(w) + 1
                if char_count >= pos:
                    word_idx = i
                    break

            half = window_size // 2
            start = max(0, word_idx - half)
            end = min(len(words), word_idx + half)

            return {
                "context": " ".join(words[start:end]),
                "matched_answer": answer,
                "match_position": pos,
                "total_words": len(words),
            }
    return None


class BM25Retriever:
    """Retrieve documents from a Lucene index and check for answer matches."""

    def __init__(
        self,
        index_path: str,
        top_k: int = 100,
        window_size: int = 256,
    ):
        self.top_k = top_k
        self.window_size = window_size

        from pyserini.search.lucene import LuceneSearcher

        print(f"Loading BM25 index from {index_path}...")
        self.searcher = LuceneSearcher(index_path)
        print(f"Index loaded ({self.searcher.num_docs:,} documents)")

    def get_doc_text(self, doc_id: str) -> Optional[str]:
        """Retrieve document text from the index by doc ID."""
        doc = self.searcher.doc(doc_id)
        if doc is None:
            return None
        try:
            raw = json.loads(doc.raw())
            return raw.get("contents", raw.get("text", ""))
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def check_answer_in_text(text: str, answers: List[str]) -> List[str]:
        """Check if any answer appears as a substring (case-insensitive)."""
        if not text:
            return []
        text_lower = text.lower()
        return [a for a in answers if a.lower() in text_lower]

    def search(self, question: str, answers: List[str]) -> Dict:
        """BM25 search + answer string matching.

        Args:
            question: The query string.
            answers: List of gold answer strings.

        Returns:
            Dict with has_answer flag, matching_docs list, and metadata.
        """
        hits = self.searcher.search(question, k=self.top_k)

        base = {
            "original_question": question,
            "original_answers": answers,
            "docs_checked": len(hits),
        }

        if not hits:
            return {**base, "has_answer": False, "has_answer_docs": 0}

        matching_docs = []
        for hit in hits:
            text = self.get_doc_text(hit.docid)
            if text is None:
                continue

            found = self.check_answer_in_text(text, answers)
            if found:
                ctx = extract_context(text, found, self.window_size)
                matching_docs.append({
                    "doc_id": hit.docid,
                    "score": hit.score,
                    "found_answers": found,
                    "matched_answer": ctx["matched_answer"] if ctx else found[0],
                    "match_position": ctx["match_position"] if ctx else -1,
                    "total_words": ctx["total_words"] if ctx else len(text.split()),
                    "context_snippet": ctx["context"] if ctx else text[:500],
                    "full_text": text,
                })

        if not matching_docs:
            return {**base, "has_answer": False, "has_answer_docs": 0}

        return {
            **base,
            "has_answer": True,
            "has_answer_docs": len(matching_docs),
            "matching_docs": matching_docs,
        }


class PyseriniRestRetriever:
    """Retrieve documents from the Pyserini REST API and check answer matches."""

    def __init__(
        self,
        index_path: str,
        api_base_url: str,
        api_token_env: str = "PYSERINI_API_TOKEN",
        top_k: int = 100,
        window_size: int = 256,
    ):
        self.index_path = index_path
        self.api_base_url = api_base_url.rstrip("/")
        self.top_k = top_k
        self.window_size = window_size
        self.token = os.environ.get(api_token_env)

        if not self.token:
            raise ValueError(f"Set {api_token_env} to use the Pyserini REST API")

        print(f"Using Pyserini REST API index {index_path} at {self.api_base_url}")

    def _get_json(self, path: str, params: Optional[Dict] = None) -> Dict:
        query = f"?{urlencode(params)}" if params else ""
        url = f"{self.api_base_url}{path}{query}"
        request = Request(url, headers={"Authorization": f"Bearer {self.token}"})

        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Pyserini REST API request failed ({e.code}): {detail}") from e
        except URLError as e:
            raise RuntimeError(f"Pyserini REST API request failed: {e.reason}") from e

    @staticmethod
    def _doc_to_text(doc) -> Optional[str]:
        if doc is None:
            return None
        if isinstance(doc, str):
            try:
                return PyseriniRestRetriever._doc_to_text(json.loads(doc))
            except json.JSONDecodeError:
                return doc
        if isinstance(doc, dict):
            for key in ("contents", "text", "segment"):
                value = doc.get(key)
                if isinstance(value, str):
                    return value
            return json.dumps(doc)
        return str(doc)

    @staticmethod
    def check_answer_in_text(text: str, answers: List[str]) -> List[str]:
        """Check if any answer appears as a substring (case-insensitive)."""
        if not text:
            return []
        text_lower = text.lower()
        return [a for a in answers if a.lower() in text_lower]

    def search(self, question: str, answers: List[str]) -> Dict:
        index = quote(self.index_path, safe="")
        data = self._get_json(f"/v1/{index}/search", {"query": question, "hits": self.top_k})
        hits = data.get("candidates", [])

        base = {
            "original_question": question,
            "original_answers": answers,
            "docs_checked": len(hits),
        }

        if not hits:
            return {**base, "has_answer": False, "has_answer_docs": 0}

        matching_docs = []
        for hit in hits:
            text = self._doc_to_text(hit.get("doc"))
            if text is None:
                doc_id = quote(hit["docid"], safe="")
                doc_data = self._get_json(f"/v1/{index}/doc/{doc_id}")
                text = self._doc_to_text(doc_data.get("doc"))
            if text is None:
                continue

            found = self.check_answer_in_text(text, answers)
            if found:
                ctx = extract_context(text, found, self.window_size)
                matching_docs.append({
                    "doc_id": hit["docid"],
                    "score": hit.get("score"),
                    "found_answers": found,
                    "matched_answer": ctx["matched_answer"] if ctx else found[0],
                    "match_position": ctx["match_position"] if ctx else -1,
                    "total_words": ctx["total_words"] if ctx else len(text.split()),
                    "context_snippet": ctx["context"] if ctx else text[:500],
                    "full_text": text,
                })

        if not matching_docs:
            return {**base, "has_answer": False, "has_answer_docs": 0}

        return {
            **base,
            "has_answer": True,
            "has_answer_docs": len(matching_docs),
            "matching_docs": matching_docs,
        }
