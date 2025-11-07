import re
from typing import Any, Dict, Optional, Tuple

from .state import SearchStrategy, QueryIntent


class StrategySelector:
    """Select a search strategy using the analyzed intent and optional user context.

    Relies on intent produced by the analyze_query step. Provides a reason string
    to aid debugging and traceability.
    """

    DEFAULT = SearchStrategy.HYBRID_SEARCH

    def select(
        self,
        query: str,
        intent: Optional[QueryIntent] = None,
        user_context: Optional[Dict[str, Any]] = None,
        override: Optional[SearchStrategy] = None,
    ) -> Tuple[SearchStrategy, str]:
        # Respect explicit override
        if override:
            return override, "Using override strategy from initial state"

        user_context = user_context or {}

        # If we don't have intent (LLM off or analysis failed), default safely
        if not intent:
            return self.DEFAULT, "No analyzed intent available; defaulting to hybrid search"

        # Document-focused queries
        if intent.get("mentions_document"):
            # Strengthen decision if filename hint present in context or query
            if user_context.get("filename") or re.search(r"\b\w+\.pdf\b", query or ""):
                return SearchStrategy.DOCUMENT_SEARCH, "Document mention/filename hint detected; using document search"
            return SearchStrategy.DOCUMENT_SEARCH, "Document mention detected; using document search"

        # Questions
        if (intent.get("query_type") or "").lower() == "question":
            # Entities in a question → entity search first (structured filters)
            if intent.get("has_entities"):
                return SearchStrategy.ENTITY_SEARCH, "Question with entities; using entity search"

            # Explicit overview aspect → summary search
            aspects = intent.get("aspects") or []
            if any(re.search(r"\b(overview|summary|brief|quick)\b", a or "") for a in aspects):
                return SearchStrategy.SUMMARY_SEARCH, "Overview aspect; using summary search"

            # Complex or multi-hop → hybrid search
            if intent.get("requires_multi_hop") or (intent.get("complexity") or "").lower() == "high":
                return SearchStrategy.HYBRID_SEARCH, "Complex/multi-hop question; using hybrid search"

            # Default for direct questions → curated QA pairs
            return SearchStrategy.QA_PAIRS, "Direct question; preferring curated QA pairs"

        # Statements
        if intent.get("has_entities"):
            return SearchStrategy.ENTITY_SEARCH, "Entity-like statement; using entity search"

        aspects = intent.get("aspects") or []
        if any(re.search(r"\b(overview|summary|brief|quick)\b", a or "") for a in aspects):
            return SearchStrategy.SUMMARY_SEARCH, "Overview aspect; using summary search"

        if intent.get("requires_multi_hop") or (intent.get("complexity") or "").lower() == "high":
            return SearchStrategy.HYBRID_SEARCH, "Complex statement; using hybrid search"

        # Default
        return self.DEFAULT, "Defaulting to hybrid based on analyzed intent"