from typing import List
from collections import defaultdict
from pydantic import Field
import re
import logging

from langchain.schema import BaseRetriever, Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from constants import COMMON_WORDS


class HybridRetriever(BaseRetriever):
    bm25_retriever: BaseRetriever = Field(...)
    semantic_retriever: BaseRetriever = Field(...)
    k_bm25: int = Field(default=3)
    k_semantic: int = Field(default=5)
    k_final: int = Field(default=None)
    use_bm25: bool = Field(default=True)

    def is_identifier_word(self, word: str) -> bool:
        return bool(re.fullmatch(r"[\w\-]+", word)) and not word.isalpha()

    def get_uncommon_or_identifier_words(self, query: str) -> List[str]:
        words = re.findall(r"[\w\-]+", query.lower())
        selected = []
        for w in words:
            if len(w) <= 2 or w in COMMON_WORDS:
                continue
            looks_like_identifier = any([
                re.search(r"\d", w),
                "_" in w,
                "-" in w,
                any(c.isupper() for c in w),
            ])
            is_uncommon = w not in COMMON_WORDS and len(w) > 5
            if looks_like_identifier or is_uncommon:
                selected.append(w)
        return selected

    def semantic_has_exact_match(self, semantic_docs: List[Document], word: str) -> bool:
        pattern = r"\b" + re.escape(word) + r"\b"
        for doc in semantic_docs:
            if re.search(pattern, doc.page_content, flags=re.IGNORECASE):
                return True
        return False

    def _get_semantic_docs_with_scores(self, query: str) -> List[Document]:
        """Retrieve semantic docs and attach scores to metadata (MultiVectorRetriever-style)."""
        results = []
        try:
            # Prefer normalized similarity (relevance) if available
            if hasattr(self.semantic_retriever, "vectorstore"):
                raw = self.semantic_retriever.vectorstore.similarity_search_with_relevance_scores(
                    query, k=self.k_semantic
                )
            else:
                raw = self.semantic_retriever.vectorstore.similarity_search_with_score(
                    query, k=self.k_semantic
                )
            results = raw
        except Exception as e:
            logging.warning(f"Falling back to raw semantic retrieval: {e}")
            docs = self.semantic_retriever.get_relevant_documents(query)
            results = [(doc, None) for doc in docs]

        # Attach score to metadata (LangChain MultiVectorRetriever style)
        id_to_docs = defaultdict(list)
        for doc, score in results:
            doc.metadata = dict(doc.metadata or {})
            doc.metadata["score"] = score
            doc_id = doc.metadata.get("doc_id") or id(doc)
            id_to_docs[doc_id].append(doc)

        # Return flattened docs (no docstore like MultiVectorRetriever)
        semantic_docs = [d for group in id_to_docs.values() for d in group]
        return semantic_docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """Retrieve hybrid documents with scores appended to metadata."""
        semantic_docs = self._get_semantic_docs_with_scores(query)
        special_words = self.get_uncommon_or_identifier_words(query)

        add_bm25 = any(
            not self.semantic_has_exact_match(semantic_docs, word)
            for word in special_words
        )

        if self.use_bm25 and add_bm25:
            bm25_docs = self.bm25_retriever.get_relevant_documents(query)[: self.k_bm25]
            for doc in bm25_docs:
                doc.metadata = dict(doc.metadata or {})
                doc.metadata.setdefault("score", 0.0)
            seen = set()
            final_docs = []
            for doc in bm25_docs + semantic_docs:
                if doc.page_content not in seen:
                    final_docs.append(doc)
                    seen.add(doc.page_content)
        else:
            final_docs = semantic_docs

        if self.k_final:
            final_docs = final_docs[: self.k_final]

        for doc in final_docs:
            logging.info(
                f"[Retriever] Source={doc.metadata.get('source','?')} | Score={doc.metadata.get('score')}"
            )

        return final_docs

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
