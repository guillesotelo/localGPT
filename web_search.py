import logging
from langchain.schema import Document


def context_is_sufficient(llm, question: str, docs: list[Document]) -> bool:
    """Ask the LLM whether the retrieved chunks contain enough information to answer.

    Returns True (sufficient) on any error so the caller never blocks on a check failure.
    """
    # Use all docs (up to 10) and more chars per doc so the check isn't fooled by
    # short previews that miss the key content in the second half of a chunk.
    context_preview = "\n\n---\n\n".join(doc.page_content[:800] for doc in docs[:10])
    check_prompt = (
        "You are a relevance judge for a technical documentation chatbot.\n"
        "Decide whether the documentation excerpts below contain enough specific information "
        "to answer the user's question with concrete details (steps, parameters, code, etc.).\n"
        "A partial match is SUFFICIENT if the excerpts cover the main topic even if not every detail.\n"
        "Reply with exactly one word: SUFFICIENT or INSUFFICIENT\n\n"
        f"Documentation excerpts:\n{context_preview}\n\n"
        f"Question: {question}"
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
            title = r.get("title", "").replace('HP Developer Portal', 'External')
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": r.get("href", ""), 
                    "origin": "web",
                    "title": title
                },
            ))

        logging.info("[Web Search] Query %r → %d results", query, len(docs))
        return docs

    except Exception as e:
        logging.warning("[Web Search] Failed: %s", e)
        return []
