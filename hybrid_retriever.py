from langchain.schema import BaseRetriever, Document
from typing import List
from pydantic import Field
import re

class HybridRetriever(BaseRetriever):
    bm25_retriever: BaseRetriever = Field(...)
    semantic_retriever: BaseRetriever = Field(...)
    k_bm25: int = Field(default=3)
    k_semantic: int = Field(default=5)
    k_final: int = Field(default=None)

    def is_identifier_query(self, query: str) -> bool:
        # identifier: letters/digits/_/- only, no spaces
        return bool(re.fullmatch(r"[\w\-]+", query))

    def _get_relevant_documents(self, query: str) -> List[Document]:
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:self.k_bm25]
        semantic_docs = self.semantic_retriever.get_relevant_documents(query)[:self.k_semantic]

        # exact matches (string contains)
        exact_matches = [
            doc for doc in bm25_docs
            if query.lower() in doc.page_content.lower()
        ]

        # 1. If it's an identifier-like query → prioritize exact match only
        if self.is_identifier_query(query) and exact_matches:
            return exact_matches[: self.k_final or self.k_bm25]

        # 2. Otherwise → combine exact, bm25, and semantic
        seen = set()
        final_docs = []

        # Exact first
        for doc in exact_matches:
            if doc.page_content not in seen:
                final_docs.append(doc)
                seen.add(doc.page_content)

        # BM25 next
        for doc in bm25_docs:
            if doc.page_content not in seen:
                final_docs.append(doc)
                seen.add(doc.page_content)

        # Semantic last
        for doc in semantic_docs:
            if doc.page_content not in seen:
                final_docs.append(doc)
                seen.add(doc.page_content)

        if self.k_final:
            final_docs = final_docs[:self.k_final]

        return final_docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
