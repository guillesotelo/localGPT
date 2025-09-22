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
        
        logging.info('\n ')
        logging.info('\n Full Text Search')
        for doc in bm25_docs:
            logging.info(f"Document: {doc.metadata.get('source', 'Unknown Source')}")
            logging.info(f"Content: {doc.page_content[:100]}")
        logging.info('\n ')
        logging.info('\n ')

        # Merge (naive concat — you could deduplicate or rerank here)
        docs = bm25_docs + semantic_docs

        if self.k_final:
            docs = docs[:self.k_final]
        return docs

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
