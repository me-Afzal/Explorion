import os
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

# Collects every query made during a pipeline run.
# app.py clears this before each invocation and reads it for progress display.
search_log: list[str] = []

MAX_SEARCHES = 15

# Optional callback invoked (in the same thread) each time a search fires.
# app.py registers this to drive real-time UI updates.
_search_callback = None


def register_search_callback(cb) -> None:
    global _search_callback
    _search_callback = cb


def clear_search_callback() -> None:
    global _search_callback
    _search_callback = None


@tool
def web_search(query: str) -> str:
    """Search the web for information about the given query."""
    normalized = query.strip().lower()

    # Reject duplicates so the agent can't loop on the same query.
    if any(normalized == q.strip().lower() for q in search_log):
        return (
            "[DUPLICATE QUERY — you already searched for this. "
            "Choose a completely different angle or proceed to synthesis.]"
        )

    # Hard cap on total searches per run.
    if len(search_log) >= MAX_SEARCHES:
        return (
            "[SEARCH LIMIT REACHED — no more searches allowed. "
            "Immediately write your final synthesis using the information gathered so far.]"
        )

    search_log.append(query)
    print(f"[Web Search] {query}")

    if _search_callback is not None:
        try:
            _search_callback(query)
        except Exception:
            pass  # never let a UI callback crash the tool

    try:
        tavily = TavilySearch(max_results=4)
        results = tavily.invoke(query)
        content = "\n".join([r["content"] for r in results["results"]])
        return content
    except Exception as exc:
        print(f"[TOOL FAILED] {exc}")
        return f"Search failed: {exc}"
