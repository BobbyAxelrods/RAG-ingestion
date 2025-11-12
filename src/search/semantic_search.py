import re
from typing import Any, Dict, List, Optional

from azure.search.documents.models import QueryCaptionType, QueryAnswerType


class SemanticRAGStrategy:
    """Semantic search strategy returning an LLM-generated answer with citations.

    This class is aligned with the unified strategy interface used in strategies.py.
    """

    def __init__(self, search_client, openai_client):
        self.search_client = search_client
        self.openai_client = openai_client

    def _is_traditional_chinese(self, text: str) -> bool:
        return bool(re.search(r"[\u4E00-\u9FFF]", text))

    def _build_llm_answer(self, query: str, context: str, model: str, temperature: float = 0.2, max_tokens: int = 700) -> Dict[str, Any]:
        system_message = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Always cite your sources using [Source N]. "
            "If the context lacks sufficient information, state that clearly."
        )

        user_message = f"""Context:
{context}

Question: {query}

Provide a comprehensive answer based on the context, and cite sources."""

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
        }

    def generate_answer(
        self,
        *,
        query: str,
        top_k: int = 5,
        model: str,
        semantic_config_tc: str = "insurance-semantic-config-tc",
        semantic_config_en: str = "insurance-semantic-config-en",
    ) -> Dict[str, Any]:
        semantic_config = semantic_config_tc if self._is_traditional_chinese(query) else semantic_config_en

        search_results = list(
            self.search_client.search(
                search_text=query,
                query_type="semantic",
                semantic_configuration_name=semantic_config,
                query_caption=QueryCaptionType.EXTRACTIVE,
                query_answer=QueryAnswerType.EXTRACTIVE,
                select=[
                    "doc_id",
                    "file_name",
                    "chunk_content",
                    "file_url",
                    "chunk_page_number",
                ],
                top=top_k,
            )
        )

        # Prefer high-confidence semantic answer if provided
        best_answer: Optional[Dict[str, Any]] = None
        for r in search_results:
            answers = r.get("@search.answers") or []
            if answers:
                ba = max(answers, key=lambda a: getattr(a, "score", 0))
                if getattr(ba, "score", 0) > 0.85:
                    best_answer = {
                        "content": getattr(ba, "text", ""),
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    }
                    break

        # Build context from captions or content
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        for rank, r in enumerate(search_results, 1):
            captions = r.get("@search.captions") or []
            if captions:
                first_cap = captions[0]
                cap_text = getattr(first_cap, "text", str(first_cap))
            else:
                cap_text = r.get("chunk_content")
            fn = r.get("file_name")
            pg = r.get("chunk_page_number")
            context_parts.append(f"[Source {rank}] {fn} (page {pg})\n{cap_text}\n")
            sources.append(
                {
                    "rank": rank,
                    "doc_id": r.get("doc_id"),
                    "file_name": fn,
                    "page": pg,
                    "url": r.get("file_url"),
                    "search_score": r.get("@search.score"),
                    "rerank_score": r.get("@search.rerankScore") or getattr(r, "@search.rerankScore", None),
                }
            )
        context = "\n".join(context_parts)

        llm = best_answer or self._build_llm_answer(query, context, model)
        return {
            "query": query,
            "answer": llm["content"],
            "sources": sources,
            "retrieval_method": "semantic",
            "model": model,
            "token_usage": llm["usage"],
        }