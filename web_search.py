import logging
from langchain.schema import Document


def context_is_sufficient(llm, question: str, docs: list[Document]) -> bool:
    """Ask the LLM whether the retrieved chunks contain enough information to answer.

    Returns True (sufficient) on any error so the caller never blocks on a check failure.
    """
    context_preview = "\n\n".join(doc.page_content[:400] for doc in docs[:6])
    check_prompt = (
        "Given the context excerpts and the user question below, decide whether the context "
        "contains enough specific information to give a complete, concrete answer.\n\n"
        f"Context:\n{context_preview}\n\n"
        f"Question: {question}\n\n"
        "Reply with exactly one word: SUFFICIENT or INSUFFICIENT"
    )
    try:
        response = llm.invoke(check_prompt, config={"callbacks": []})
        text = response.content if hasattr(response, "content") else str(response)
        result = text.strip().upper()
        logging.info("[Web Search] Sufficiency check → %s", result)
        return "INSUFFICIENT" not in result
    except Exception as e:
        logging.warning("[Web Search] Sufficiency check failed: %s", e)
        return True  # fail open — don't trigger web search on error


def web_search_fallback(query: str, max_results: int = 4) -> list[Document]:
    """Search the web via DuckDuckGo and return results as Documents.

    Returns an empty list on any failure so callers can treat it as a soft fallback.
    """
    try:
        try:
            from ddgs import DDGS  # current package name (renamed from duckduckgo_search)
        except ImportError:
            from duckduckgo_search import DDGS  # legacy fallback

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
