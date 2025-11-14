from enum import Enum
from typing import List, Optional, TypedDict


class SearchStrategy(str, Enum):
    HYBRID_SEARCH = "hybrid_search"
    QA_PAIRS = "qa_pairs"
    ENTITY_SEARCH = "entity_search"
    SUMMARY_SEARCH = "summary_search"
    DOCUMENT_SEARCH = "document_search"
    HYBRID_SEMANTIC = "hybrid_semantic"
    FACETED_SEARCH = "faceted_search"
    EXPANDED_QUERY = "expanded_query"
    BRANCH_FILTER = "branch_filter"
    CATEGORY_FILTER = "category_filter"
    LIBRARY_FILTER = "library_filter"
    DATE_RANGE_FILTER = "date_range_filter"
    PAGE_RANGE_FILTER = "page_range_filter"


class QueryIntent(TypedDict):
    query_type: str
    complexity: str
    mentions_document: bool
    has_entities: bool
    requires_multi_hop: bool
    aspects: List[str]
    language: str


class AgentState(TypedDict):
    # Input
    query: str
    session_id: str
    user_context: dict
    conversation_history: List[dict]

    # Analysis
    query_intent: Optional[QueryIntent]

    # Strategy
    current_strategy: Optional[SearchStrategy]
    strategies_tried: List[str]
    strategy_reasons: dict
    attempt_count: int

    # Search results
    retrieved_documents: List[dict]
    raw_search_results: dict
    search_metadata: dict

    # Evaluation
    retrieval_evaluation: Optional[dict]
    is_satisfied: bool
    failure_reason: str
    evaluation_scores: dict

    # Notifications
    user_notifications: List[str]

    # Answer generation
    generated_answer: Optional[str]
    answer_evaluation: Optional[dict]
    answer_confidence: float

    # Final
    final_response: Optional[dict]
    execution_metadata: dict
    processing_time_ms: int
    # Settings
    confidence_threshold: float