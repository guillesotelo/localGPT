import logging
from langchain.schema import Document


def web_search_fallback(query: str, max_results: int = 4) -> list[Document]:
    """Search the web via DuckDuckGo and return results as Documents.

    Returns an empty list on any failure so callers can treat it as a soft fallback.
    """
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        if not results:
            logging.info("[Web Search] No results returned for query: %r", query)
            return []

        docs = []
        for r in results:
            content = f"{r.get('title', '')}\n{r.get('body', '')}"
            docs.append(Document(
                page_content=content,
                metadata={"source": r.get("href", ""), "origin": "web"},
            ))

        logging.info("[Web Search] Query %r → %d results", query, len(docs))
        return docs

    except Exception as e:
        logging.warning("[Web Search] Failed: %s", e)
        return []
