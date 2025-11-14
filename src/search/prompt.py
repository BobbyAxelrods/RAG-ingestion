from typing import List, Dict


def format_citations(citations: List[Dict]) -> str:
    """Create a compact citations block from search results."""
    lines: List[str] = []
    for i, c in enumerate(citations, start=1):
        fn = (c.get("file_name") or "").strip()
        pg = c.get("chunk_page_number")
        summary = (c.get("chunk_function_summary") or "").strip()
        snippet = (c.get("chunk_content") or "").strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "…"
        lines.append(f"[{i}] file: {fn} | page: {pg} | summary: {summary}\n{snippet}")
    return "\n\n".join(lines)


def build_rag_messages(query: str, citations: List[Dict]) -> List[Dict[str, str]]:
    """Build chat messages for RAG answer generation using provided citations only."""
    citation_text = format_citations(citations)
    system = (
        "You are a helpful insurance assistant. Answer strictly based on the provided citations. "
        "If the citations do not contain enough information, say you cannot answer. "
        "Return a clear, step-by-step instruction when applicable. Keep the answer concise."
    )
    user = (
        f"Question: {query}\n\n"
        f"Citations:\n{citation_text}\n\n"
        "Answer using only the content above and include practical steps if relevant."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]