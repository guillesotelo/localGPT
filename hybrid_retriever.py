from typing import List
from collections import defaultdict
from pydantic import Field
import re
import logging
import sqlite3
import json

from langchain.schema import BaseRetriever, Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from constants import COMMON_WORDS


def group_and_order_by_document(docs: List[Document]) -> List[Document]:
    grouped = defaultdict(list)

    for doc in docs:
        doc_id = doc.metadata.get("doc_id")
        if doc_id:
            grouped[doc_id].append(doc)
        else:
            grouped[None].append(doc)

    ordered = []

    for doc_id, chunks in grouped.items():
        if doc_id is None:
            ordered.extend(chunks)
        else:
            chunks.sort(key=lambda d: d.metadata.get("chunk_seq", 0))
            ordered.extend(chunks)

    return ordered


def search_fts(query, k, db_path):
    """Full text search"""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT page_content, source, chunk_id, metadata_json, bm25(docs) AS score
            FROM docs
            WHERE docs MATCH ?
            ORDER BY score
            LIMIT ?;
        """, (query, k))

        rows = cur.fetchall()
        conn.close()
        
        if not rows:
            return []

        docs = []
        
        # Normalize BM25 scores to 0–1 (higher = better)
        scores = [row["score"] for row in rows]
        min_s, max_s = min(scores), max(scores)
        if max_s == min_s:
            norm_scores = [1.0 for _ in scores]
        else:
            norm_scores = [(max_s - s) / (max_s - min_s) for s in scores]


        docs = []
        for row, norm_score in zip(rows, norm_scores):
            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            meta["bm25_score_raw"] = row["score"]
            meta["score"] = norm_score  # normalized score
            meta.setdefault("source", row["source"])
            meta.setdefault("chunk_id", row["chunk_id"])
            docs.append(Document(page_content=row["page_content"], metadata=meta))

        return docs
    
    except:
        return []


def get_uncommon_or_identifier_words(query: str) -> List[str]:
    words = re.findall(r"\b[\w\-]+\b", query)
    selected = []
    for w in words:
        lw = w.lower()

        looks_like_identifier = (
            bool(re.search(r"[_\-\d]", w))
            or (w.isupper() and len(w) >= 2)
        )
        
        is_uncommon = lw not in COMMON_WORDS and len(w) > 2

        if looks_like_identifier or is_uncommon:
            selected.append(w)
    return selected


def results_contain_relevance_words(query: str, docs: List[Document]) -> bool:
    words = re.findall(r"\b[\w\-]+\b", query)
    flag = False
    for w in words:
        lw = w.lower()
        for doc in docs:
            if lw not in COMMON_WORDS and lw in doc.page_content.lower():
                flag = True
    return flag
                
            
def quote_fts_token(token: str) -> str:
    token = token.replace('"', '""')
    # only quote if needed (spaces, operators, etc.)
    if re.search(r'[^\w]', token):
        return f'"{token}"'
    return token

    
class HybridRetriever(BaseRetriever):
    semantic_retriever: BaseRetriever = Field(...)
    k_bm25: int = Field(default=3)
    k_semantic: int = Field(default=5)
    k_final: int = Field(default=6)
    use_bm25: bool = Field(default=True)
    use_scores: bool = Field(default=False)
    db_path: str = Field(default="fts_index.db")


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
            if self.use_scores:    
                # Prefer normalized similarity (relevance) if available
                if hasattr(self.semantic_retriever, "vectorstore"):
                    raw = self.semantic_retriever.vectorstore.similarity_search_with_relevance_scores(
                        query, k=self.k_semantic
                    )
                else:
                    raw = self.semantic_retriever.vectorstore.similarity_search_with_score(
                        query, k=self.k_semantic
                    )
            else:
                docs = self.semantic_retriever.invoke(query)[: self.k_semantic]
                raw = [(doc, None) for doc in docs]
            results = raw
        except Exception as e:
            logging.warning(f"Falling back to raw semantic retrieval: {e}")
            docs = self.semantic_retriever.invoke(query)
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
        special_words = get_uncommon_or_identifier_words(query)

        final_docs = list(semantic_docs)
        
        add_bm25 = any(
            not self.semantic_has_exact_match(semantic_docs, word)
            for word in special_words
        ) or results_contain_relevance_words(query, final_docs)

        if self.use_bm25 and (add_bm25 or special_words):
            bm25_query = " AND ".join(quote_fts_token(w) for w in special_words)
            bm25_docs = search_fts(bm25_query, self.k_bm25, db_path=self.db_path)

            if bm25_docs:
                for doc in bm25_docs:
                    doc.metadata = dict(doc.metadata or {})
                    doc.metadata.setdefault("score", 0.0)

                    logging.info(
                        f"[Retriever BM25] Source={doc.metadata.get('source','?')} | Score={doc.metadata.get('score')}"
                    )

                seen = set()
                merged = []
                for doc in bm25_docs + semantic_docs:
                    if doc.page_content not in seen:
                        merged.append(doc)
                        seen.add(doc.page_content)
                final_docs = merged

        # Enforce document continuity
        final_docs = group_and_order_by_document(final_docs)

        # hard cap after ordering
        if self.k_final:
            final_docs = final_docs[: self.k_final]

        return final_docs


    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)
