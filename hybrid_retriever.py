from langchain.schema import BaseRetriever, Document
from typing import List
from pydantic import Field
import logging

class HybridRetriever(BaseRetriever):
    bm25_retriever: BaseRetriever = Field(...)
    semantic_retriever: BaseRetriever = Field(...)
    k_bm25: int = Field(default=3)
    k_semantic: int = Field(default=5)
    k_final: int = Field(default=None)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        bm25_docs = self.bm25_retriever.get_relevant_documents(query)[:self.k_bm25]
        semantic_docs = self.semantic_retriever.get_relevant_documents(query)[:self.k_semantic]

        # If at least one exact match found, prioritize full text
        exact_matches = [doc for doc in bm25_docs if query.lower() in doc.page_content.lower()]
    
        if exact_matches:  
            return exact_matches[:self.k_final or self.k_bm25]
        
        seen = set()
        docs = []
        for doc in bm25_docs + semantic_docs:  # BM25 first, then semantic
            if doc.page_content not in seen:
                docs.append(doc)
                seen.add(doc.page_content)
        
        if self.k_final:
            docs = docs[:self.k_final]
            
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
