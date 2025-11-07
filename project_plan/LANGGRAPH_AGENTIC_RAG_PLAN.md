# LangGraph-Powered Agentic RAG System - Comprehensive Implementation Plan
## Self-Evaluating, Strategy-Switching RAG with User Feedback Loop

---

## 🎯 Executive Summary

Build an **intelligent, self-evaluating RAG system** using LangGraph state machines and LangChain agents that:

- **Analyzes** user queries to understand intent and complexity
- **Selects** optimal search strategy from 13 available strategies
- **Executes** search against Azure AI Search
- **Evaluates** results quality using LLM-based assessment
- **Iterates** with strategy switching until satisfied OR max attempts reached
- **Notifies** user during strategy changes: *"We didn't find the result yet, changing search strategy, please hold on"*
- **Generates** final answer only when evaluation passes
- **Returns** transparent metadata about the process

---

## 🧠 Core Architecture

### System Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   USER QUERY RECEIVED VIA API                            │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        LANGGRAPH STATE MACHINE                           │
│                                                                          │
│   [START]                                                                │
│      │                                                                   │
│      ▼                                                                   │
│   ┌──────────────────┐                                                  │
│   │ analyze_query    │ ◄── LangChain: Extract intent, complexity        │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │ select_strategy  │ ◄── LangChain: Choose optimal search strategy    │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │ execute_search   │ ◄── Execute strategy against Azure AI Search     │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │ evaluate_results │ ◄── LangChain: Assess quality of retrieval       │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│      ┌─────────┐                                                        │
│      │ Satisfied?│     Agent: # Embedding-based Evaluator(embedding_distance)
                                                  │
│      └──┬───┬──┘                                                        │
│     YES │   │ NO                                                         │
│         │   │                                                            │
│         │   ├──► ┌──────────────────┐                                   │
│         │   │    │ notify_user      │ ◄── Send: "Changing strategy..."  │
│         │   │    └────────┬─────────┘                                   │
│         │   │             │                                              │
│         │   │             ▼                                              │
│         │   │    ┌──────────────────┐                                   │
│         │   │    │ replan_strategy  │ ◄── Select new strategy based     │
│         │   │    └────────┬─────────┘     on failure analysis           │
│         │   │             │                                              │
│         │   │             ├──► Check: Max attempts (3)?                  │
│         │   │             │    NO: Loop back to execute_search          │
│         │   │             │   YES: Go to return_partial                 │
│         │   │             │                                              │
│         │   └─────────────┘                                              │
│         │                                                                │
│         ▼                                                                │
│   ┌──────────────────┐                                                  │
│   │ generate_answer  │ ◄── LLM generates answer from retrieved docs     │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │ evaluate_answer  │ ◄── Self-assess answer quality                   │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│   ┌──────────────────┐                                                  │
│   │ return_response  │ ◄── Send final answer + metadata                 │
│   └────────┬─────────┘                                                  │
│            │                                                             │
│            ▼                                                             │
│         [END]                                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 LangGraph State Machine Implementation

### AgentState Definition

```python
from typing import TypedDict, List, Annotated, Optional
from operator import add
from enum import Enum

class SearchStrategy(str, Enum):
    """Available search strategies from SEARCH_STRATEGY.md"""
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
    """Query analysis results"""
    query_type: str  # question, statement, command, summary
    complexity: str  # low, medium, high
    mentions_document: bool
    has_entities: bool
    requires_multi_hop: bool
    aspects: List[str]
    language: str  # en, tc


class AgentState(TypedDict):
    """
    Central state shared across all LangGraph nodes
    This is the "brain" of the agent
    """
    # INPUT
    query: str
    session_id: str
    user_context: dict  # {branch, category, library, language}
    conversation_history: List[dict]

    # ANALYSIS (from analyze_query node)
    query_intent: Optional[QueryIntent]

    # STRATEGY PLANNING (from select_strategy node)
    current_strategy: Optional[SearchStrategy]
    strategies_tried: Annotated[List[str], add]  # Append-only
    strategy_reasons: dict  # Why each strategy was chosen
    attempt_count: int

    # SEARCH RESULTS (from execute_search node)
    retrieved_documents: List[dict]
    raw_search_results: dict
    search_metadata: dict

    # EVALUATION (from evaluate_results node)
    retrieval_evaluation: Optional[dict]
    is_satisfied: bool
    failure_reason: str
    evaluation_scores: dict  # relevance, coverage, confidence

    # USER NOTIFICATIONS (from notify_user node)
    user_notifications: Annotated[List[str], add]  # Messages sent to user

    # ANSWER GENERATION (from generate_answer node)
    generated_answer: Optional[str]
    answer_evaluation: Optional[dict]
    answer_confidence: float

    # FINAL OUTPUT (from return_response node)
    final_response: Optional[dict]
    execution_metadata: dict
    processing_time_ms: int
```

---

## 📐 LangGraph Workflow Construction

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
import asyncio
import time


def create_agentic_rag_workflow():
    """
    Build the LangGraph StateGraph workflow
    """
    # Initialize state graph
    workflow = StateGraph(AgentState)

    # Add all nodes
    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("select_strategy", select_strategy_node)
    workflow.add_node("execute_search", execute_search_node)
    workflow.add_node("evaluate_results", evaluate_results_node)
    workflow.add_node("notify_user", notify_user_node)
    workflow.add_node("replan_strategy", replan_strategy_node)
    workflow.add_node("generate_answer", generate_answer_node)
    workflow.add_node("evaluate_answer", evaluate_answer_node)
    workflow.add_node("return_partial", return_partial_node)
    workflow.add_node("return_response", return_response_node)

    # Set entry point
    workflow.set_entry_point("analyze_query")

    # Linear flow for initial steps
    workflow.add_edge("analyze_query", "select_strategy")
    workflow.add_edge("select_strategy", "execute_search")
    workflow.add_edge("execute_search", "evaluate_results")

    # CRITICAL: Conditional routing after evaluation
    workflow.add_conditional_edges(
        "evaluate_results",
        should_continue_or_replan,  # Decision function
        {
            "satisfied": "generate_answer",      # Results are good
            "retry": "notify_user",              # Need to retry
            "max_attempts": "return_partial"     # Exhausted attempts
        }
    )

    # Notification → Replanning → Search (RETRY LOOP)
    workflow.add_edge("notify_user", "replan_strategy")
    workflow.add_edge("replan_strategy", "execute_search")

    # Answer generation flow
    workflow.add_edge("generate_answer", "evaluate_answer")
    workflow.add_edge("evaluate_answer", "return_response")

    # Terminal nodes
    workflow.add_edge("return_response", END)
    workflow.add_edge("return_partial", END)

    # Add checkpointing for persistence
    memory = SqliteSaver.from_conn_string(":memory:")

    # Compile workflow
    app = workflow.compile(checkpointer=memory)

    return app


def should_continue_or_replan(state: AgentState) -> str:
    """
    CRITICAL DECISION POINT: Determine next action based on evaluation

    Returns:
        "satisfied" - Results are good, generate answer
        "retry" - Results are poor, try different strategy
        "max_attempts" - Exhausted all retries, return best available
    """
    MAX_ATTEMPTS = 3

    # Check evaluation result
    if state["is_satisfied"]:
        return "satisfied"

    # Check if max attempts reached
    if state["attempt_count"] >= MAX_ATTEMPTS:
        return "max_attempts"

    # Otherwise, retry with new strategy
    return "retry"
```

---

## 🎯 Node Implementations

### Node 1: analyze_query

```python
async def analyze_query_node(state: AgentState) -> AgentState:
    """
    Analyze user query to understand intent and complexity
    Uses LangChain LLM for intelligent analysis
    """
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query analyzer for an insurance RAG system.

Analyze the user query and extract:
1. Query type: question, statement, command, summary, exploration
2. Complexity: low (simple lookup), medium (multi-aspect), high (multi-hop reasoning)
3. Mentions specific document: yes/no
4. Contains entities: yes/no (insurance products, amounts, terms)
5. Requires multi-hop: yes/no (needs multiple sources)
6. Query aspects: List all sub-questions/aspects
7. Language: en (English) or tc (Traditional Chinese)

Return JSON format."""),
        ("human", "Query: {query}\n\nAnalyze this query.")
    ])

    messages = prompt.format_messages(query=state["query"])
    response = await llm.ainvoke(messages)

    # Parse response (simplified - add proper JSON parsing)
    intent = parse_intent_response(response.content)

    state["query_intent"] = intent

    return state


def parse_intent_response(response: str) -> QueryIntent:
    """Parse LLM response into QueryIntent structure"""
    # Implementation: Parse JSON from LLM response
    # For production, use PydanticOutputParser
    pass
```

### Node 2: select_strategy

```python
async def select_strategy_node(state: AgentState) -> AgentState:
    """
    Select optimal search strategy based on query analysis
    Uses LangChain LLM to reason about strategy selection
    """
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        temperature=0.1
    )

    intent = state["query_intent"]
    user_ctx = state["user_context"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a search strategy planner for an insurance RAG system.

Available Search Strategies:
1. **qa_pairs**: Search pre-generated Q&A pairs (BEST for direct questions)
2. **hybrid_search**: Keyword + vector search (DEFAULT, balanced approach)
3. **hybrid_semantic**: Hybrid + AI semantic ranking (complex queries)
4. **entity_search**: Search by entities/products (specific terms/amounts)
5. **summary_search**: Search summaries only (quick overview)
6. **document_search**: Search specific document (when doc mentioned)
7. **branch_filter**: Filter by HK/MACAU
8. **category_filter**: Filter by insurance category
9. **library_filter**: Filter by document library
10. **faceted_search**: Browse with filters
11. **date_range_filter**: Recent documents only
12. **page_range_filter**: Specific page range in document
13. **expanded_query**: Query expansion with synonyms

SELECTION RULES:
- Question queries → Try qa_pairs first
- Mentions document name → document_search
- Mentions specific amount/term → entity_search
- "Overview" or "summary" → summary_search
- Location-specific → branch_filter
- Product-specific → category_filter
- Default → hybrid_search

Return JSON with:
- recommended_strategy: strategy name
- reasoning: why this strategy
- fallback_strategies: alternatives if this fails"""),
        ("human", """Query: {query}

Query Analysis:
- Type: {query_type}
- Complexity: {complexity}
- Mentions document: {mentions_doc}
- Has entities: {has_entities}
- Aspects: {aspects}

User Context:
- Branch: {branch}
- Category: {category}
- Language: {language}

Select the BEST search strategy and explain why.""")
    ])

    messages = prompt.format_messages(
        query=state["query"],
        query_type=intent["query_type"],
        complexity=intent["complexity"],
        mentions_doc=intent["mentions_document"],
        has_entities=intent["has_entities"],
        aspects=", ".join(intent["aspects"]),
        branch=user_ctx.get("branch", "N/A"),
        category=user_ctx.get("category", "N/A"),
        language=user_ctx.get("language", "en")
    )

    response = await llm.ainvoke(messages)
    plan = parse_strategy_plan(response.content)

    # Update state
    state["current_strategy"] = plan["recommended_strategy"]
    state["strategies_tried"].append(plan["recommended_strategy"])
    state["strategy_reasons"][plan["recommended_strategy"]] = plan["reasoning"]

    return state
```

### Node 3: execute_search

```python
async def execute_search_node(state: AgentState) -> AgentState:
    """
    Execute the selected search strategy against Azure AI Search
    """
    from azure.search.documents import SearchClient

    strategy = state["current_strategy"]
    query = state["query"]
    user_ctx = state["user_context"]

    # Route to appropriate search implementation
    search_engine = SearchStrategyEngine(search_client, embeddings_service)

    results = await search_engine.execute_strategy(
        strategy=strategy,
        query=query,
        user_context=user_ctx,
        top_k=10
    )

    # Update state
    state["retrieved_documents"] = results["documents"]
    state["raw_search_results"] = results["raw_results"]
    state["search_metadata"] = results["metadata"]
    state["attempt_count"] += 1

    return state


class SearchStrategyEngine:
    """
    Implements all 13 search strategies from SEARCH_STRATEGY.md
    """

    async def execute_strategy(
        self,
        strategy: SearchStrategy,
        query: str,
        user_context: dict,
        top_k: int = 10
    ) -> dict:
        """Route to appropriate strategy implementation"""

        strategy_map = {
            SearchStrategy.HYBRID_SEARCH: self._hybrid_search,
            SearchStrategy.QA_PAIRS: self._qa_search,
            SearchStrategy.ENTITY_SEARCH: self._entity_search,
            SearchStrategy.SUMMARY_SEARCH: self._summary_search,
            SearchStrategy.DOCUMENT_SEARCH: self._document_search,
            SearchStrategy.HYBRID_SEMANTIC: self._semantic_search,
            # ... map all 13 strategies
        }

        search_fn = strategy_map.get(strategy, self._hybrid_search)
        return await search_fn(query, user_context, top_k)

    async def _hybrid_search(self, query: str, context: dict, top_k: int):
        """Strategy 1: Hybrid (text + vector)"""
        embedding = await self.embeddings.embed(query)

        filters = self._build_filters(context)

        results = self.client.search(
            search_text=query,
            vector_queries=[{
                "vector": embedding,
                "k_nearest_neighbors": 50,
                "fields": "chunk_content_vector"
            }],
            filter=filters,
            select="doc_id,file_name,chunk_content,chunk_page_number,chunk_function_summary,file_url",
            top=top_k
        )

        return {
            "documents": list(results),
            "raw_results": results,
            "metadata": {"strategy": "hybrid_search", "filter": filters}
        }

    async def _qa_search(self, query: str, context: dict, top_k: int):
        """Strategy 2: QA Pairs"""
        min_confidence = 0.7
        filters = [f"qa_confidence ge {min_confidence}"]

        if context.get("branch"):
            filters.append(f"branch_name eq '{context['branch']}'")

        filter_str = " and ".join(filters)

        results = self.client.search(
            search_text=query,
            search_fields="qa_questions",
            filter=filter_str,
            select="qa_questions,qa_answers,file_name,chunk_page_number,qa_confidence",
            top=top_k
        )

        return {
            "documents": list(results),
            "raw_results": results,
            "metadata": {"strategy": "qa_pairs", "min_confidence": min_confidence}
        }

    # ... implement all other strategies from SEARCH_STRATEGY.md
```

### Node 4: evaluate_results

```python
async def evaluate_results_node(state: AgentState) -> AgentState:
    """
    Evaluate quality of retrieved documents using LLM

    Evaluation dimensions:
    1. Relevance: Do documents address the query?
    2. Coverage: Are all query aspects covered?
    3. Confidence: How confident are we in these results?
    4. Completeness: Is there enough information?
    """
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        temperature=0.0
    )

    query = state["query"]
    documents = state["retrieved_documents"]
    intent = state["query_intent"]

    # Check if we have documents
    if not documents or len(documents) == 0:
        state["is_satisfied"] = False
        state["failure_reason"] = "no_results_found"
        state["retrieval_evaluation"] = {
            "score": 0.0,
            "relevance": 0.0,
            "coverage": 0.0,
            "confidence": 0.0
        }
        return state

    # Prepare documents for evaluation
    docs_text = "\n\n".join([
        f"Document {i+1}:\nFile: {doc.get('file_name')}\nPage: {doc.get('chunk_page_number')}\nContent: {doc.get('chunk_content', '')[:500]}..."
        for i, doc in enumerate(documents[:5])  # Top 5 docs
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a retrieval quality evaluator for an insurance RAG system.

Evaluate the retrieved documents against the user query on these dimensions:

1. **Relevance** (0-1): Do documents address the query topic?
2. **Coverage** (0-1): Are all aspects of the query covered?
3. **Confidence** (0-1): Is information specific and trustworthy?
4. **Completeness** (0-1): Is there enough info to answer the query?

Scoring guidelines:
- 0.9-1.0: Excellent, comprehensive answer possible
- 0.7-0.9: Good, satisfactory answer possible
- 0.5-0.7: Acceptable, partial answer possible
- 0.3-0.5: Poor, limited answer possible
- 0.0-0.3: Very poor, cannot answer query

THRESHOLD: Score >= 0.7 is SATISFACTORY, < 0.7 is NOT SATISFACTORY

Return JSON:
{{
  "relevance": float,
  "coverage": float,
  "confidence": float,
  "completeness": float,
  "overall_score": float,
  "is_satisfactory": bool,
  "reasoning": "explanation"
}}"""),
        ("human", """Query: {query}

Query Aspects: {aspects}

Retrieved Documents:
{documents}

Evaluate these documents against the query.""")
    ])

    messages = prompt.format_messages(
        query=query,
        aspects=", ".join(intent["aspects"]),
        documents=docs_text
    )

    response = await llm.ainvoke(messages)
    evaluation = parse_evaluation_response(response.content)

    # Update state
    state["retrieval_evaluation"] = evaluation
    state["is_satisfied"] = evaluation["is_satisfactory"]
    state["failure_reason"] = evaluation.get("reasoning", "")
    state["evaluation_scores"] = {
        "relevance": evaluation["relevance"],
        "coverage": evaluation["coverage"],
        "confidence": evaluation["confidence"],
        "completeness": evaluation["completeness"]
    }

    return state
```

### Node 5: notify_user (CRITICAL FOR USER FEEDBACK)

```python
async def notify_user_node(state: AgentState) -> AgentState:
    """
    Notify user that we're changing strategy

    This node sends a message back to the user:
    "We didn't find the result yet, changing search strategy, please hold on"
    """
    attempt = state["attempt_count"]
    current_strategy = state["current_strategy"]
    failure_reason = state["failure_reason"]

    # Build user notification message
    notification = f"We didn't find the result yet (attempt {attempt}/{3}). " \
                   f"Previous strategy '{current_strategy}' {failure_reason}. " \
                   f"Changing search strategy, please hold on..."

    # Add to notifications list (visible to API)
    state["user_notifications"].append({
        "timestamp": time.time(),
        "attempt": attempt,
        "message": notification,
        "failed_strategy": current_strategy,
        "reason": failure_reason
    })

    # In production, this would:
    # 1. Send SSE event to frontend
    # 2. Update WebSocket connection
    # 3. Store in session for streaming response

    return state
```

### Node 6: replan_strategy

```python
async def replan_strategy_node(state: AgentState) -> AgentState:
    """
    Adaptive replanning: Select new strategy based on failure analysis
    Uses LLM to reason about why previous strategy failed
    """
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        temperature=0.2  # Slightly higher for creative alternatives
    )

    query = state["query"]
    failed_strategy = state["current_strategy"]
    strategies_tried = state["strategies_tried"]
    failure_reason = state["failure_reason"]
    evaluation = state["retrieval_evaluation"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an adaptive search planner. The previous strategy FAILED.

Analyze WHY it failed and select a DIFFERENT strategy.

Available strategies: qa_pairs, hybrid_search, hybrid_semantic, entity_search,
summary_search, document_search, branch_filter, category_filter, library_filter,
faceted_search, date_range_filter, page_range_filter, expanded_query

RULES:
1. DO NOT repeat any strategy in: {strategies_tried}
2. Address the specific failure reason
3. If no results → try broader search (remove filters, use expanded_query)
4. If low relevance → try semantic search or entity search
5. If incomplete coverage → try hybrid or multiple strategies

Return JSON:
{{
  "recommended_strategy": "strategy_name",
  "reasoning": "why this will work better",
  "expected_improvement": "what this addresses"
}}"""),
        ("human", """Query: {query}

Failed Strategy: {failed_strategy}
Failure Reason: {failure_reason}
Strategies Already Tried: {strategies_tried}

Evaluation Scores:
- Relevance: {relevance}
- Coverage: {coverage}
- Confidence: {confidence}
- Result Count: {result_count}

Replan: Select a NEW strategy that addresses this failure.""")
    ])

    messages = prompt.format_messages(
        query=query,
        failed_strategy=failed_strategy,
        failure_reason=failure_reason,
        strategies_tried=", ".join(strategies_tried),
        relevance=evaluation.get("relevance", 0.0),
        coverage=evaluation.get("coverage", 0.0),
        confidence=evaluation.get("confidence", 0.0),
        result_count=len(state["retrieved_documents"])
    )

    response = await llm.ainvoke(messages)
    new_plan = parse_strategy_plan(response.content)

    # Update state with new strategy
    state["current_strategy"] = new_plan["recommended_strategy"]
    state["strategies_tried"].append(new_plan["recommended_strategy"])
    state["strategy_reasons"][new_plan["recommended_strategy"]] = new_plan["reasoning"]

    return state
```

### Node 7: generate_answer

```python
async def generate_answer_node(state: AgentState) -> AgentState:
    """
    Generate final answer using LLM with retrieved documents
    Only called when evaluation is satisfied
    """
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        temperature=0.3
    )

    query = state["query"]
    documents = state["retrieved_documents"]
    user_ctx = state["user_context"]

    # Format documents as context
    context = "\n\n".join([
        f"[Source {i+1}]\nFile: {doc.get('file_name')}\nPage: {doc.get('chunk_page_number')}\n{doc.get('chunk_content', '')}"
        for i, doc in enumerate(documents[:5])
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful insurance assistant for Prudential.

Answer the user's question based ONLY on the provided sources.

RULES:
1. Be accurate and grounded in sources
2. Cite sources with [Source N] notation
3. If information is incomplete, acknowledge it
4. Use clear, professional language
5. For branch-specific queries, emphasize relevant branch
6. If answer spans multiple sources, synthesize them

Language: {language}"""),
        ("human", """User Question: {query}

Context from Retrieved Documents:
{context}

Answer the question comprehensively using the provided context.""")
    ])

    messages = prompt.format_messages(
        query=query,
        context=context,
        language=user_ctx.get("language", "en")
    )

    response = await llm.ainvoke(messages)

    # Update state
    state["generated_answer"] = response.content

    return state
```

### Node 8: evaluate_answer

```python
async def evaluate_answer_node(state: AgentState) -> AgentState:
    """
    Self-evaluate the generated answer
    Check for groundedness, completeness, accuracy
    """
    llm = AzureChatOpenAI(
        deployment_name="gpt-4o",
        temperature=0.0
    )

    query = state["query"]
    answer = state["generated_answer"]
    documents = state["retrieved_documents"]

    docs_summary = "\n".join([
        f"- {doc.get('file_name')}, Page {doc.get('chunk_page_number')}"
        for doc in documents[:5]
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Evaluate the generated answer on:

1. **Groundedness** (0-1): Is answer based on sources? No hallucinations?
2. **Completeness** (0-1): Does it fully answer the question?
3. **Accuracy** (0-1): Is information correct and precise?
4. **Clarity** (0-1): Is it well-structured and clear?

Return JSON:
{{
  "groundedness": float,
  "completeness": float,
  "accuracy": float,
  "clarity": float,
  "overall_confidence": float,
  "assessment": "brief assessment"
}}"""),
        ("human", """Query: {query}

Generated Answer:
{answer}

Available Sources:
{sources}

Evaluate the answer quality.""")
    ])

    messages = prompt.format_messages(
        query=query,
        answer=answer,
        sources=docs_summary
    )

    response = await llm.ainvoke(messages)
    evaluation = parse_evaluation_response(response.content)

    state["answer_evaluation"] = evaluation
    state["answer_confidence"] = evaluation["overall_confidence"]

    return state
```

### Node 9: return_response

```python
async def return_response_node(state: AgentState) -> AgentState:
    """
    Build final response to return to user
    """
    final_response = {
        "answer": state["generated_answer"],
        "confidence": state["answer_confidence"],
        "sources": [
            {
                "doc_id": doc.get("doc_id"),
                "file_name": doc.get("file_name"),
                "page_number": doc.get("chunk_page_number"),
                "content": doc.get("chunk_content", "")[:300],
                "summary": doc.get("chunk_function_summary"),
                "file_url": doc.get("file_url")
            }
            for doc in state["retrieved_documents"][:5]
        ],
        "metadata": {
            "session_id": state["session_id"],
            "strategy_used": state["current_strategy"],
            "attempts": state["attempt_count"],
            "strategies_tried": state["strategies_tried"],
            "strategy_reasons": state["strategy_reasons"],
            "retrieval_quality": state["evaluation_scores"],
            "answer_quality": state["answer_evaluation"],
            "user_notifications": state["user_notifications"],
            "processing_time_ms": state.get("processing_time_ms", 0)
        }
    }

    state["final_response"] = final_response

    return state
```

### Node 10: return_partial

```python
async def return_partial_node(state: AgentState) -> AgentState:
    """
    Return partial results when max attempts exhausted
    """
    final_response = {
        "answer": "I apologize, but I couldn't find a fully satisfactory answer to your question after trying multiple search strategies. Here are the best results I found:",
        "confidence": 0.5,
        "sources": [
            {
                "doc_id": doc.get("doc_id"),
                "file_name": doc.get("file_name"),
                "page_number": doc.get("chunk_page_number"),
                "content": doc.get("chunk_content", "")[:300],
                "summary": doc.get("chunk_function_summary"),
                "file_url": doc.get("file_url")
            }
            for doc in state["retrieved_documents"][:5]
        ],
        "metadata": {
            "session_id": state["session_id"],
            "status": "partial_results",
            "attempts": state["attempt_count"],
            "strategies_tried": state["strategies_tried"],
            "strategy_reasons": state["strategy_reasons"],
            "last_failure_reason": state["failure_reason"],
            "user_notifications": state["user_notifications"]
        }
    }

    state["final_response"] = final_response

    return state
```

---

## 🌐 FastAPI Integration

### Main Execution Function

```python
async def process_query_with_langgraph(
    query: str,
    session_id: str,
    user_context: dict,
    stream: bool = False
) -> dict:
    """
    Main entry point: Execute LangGraph workflow
    """
    # Create workflow
    app = create_agentic_rag_workflow()

    # Initialize state
    start_time = time.time()

    initial_state = {
        "query": query,
        "session_id": session_id,
        "user_context": user_context,
        "conversation_history": await get_conversation_history(session_id),
        "strategies_tried": [],
        "strategy_reasons": {},
        "attempt_count": 0,
        "is_satisfied": False,
        "user_notifications": [],
        "execution_metadata": {}
    }

    # Execute workflow
    if stream:
        # Streaming mode: yield notifications and final response
        return stream_workflow_execution(app, initial_state)
    else:
        # Non-streaming mode: wait for completion
        final_state = None
        async for state in app.astream(initial_state):
            # Get current node
            node_name = list(state.keys())[0]
            node_output = state[node_name]

            # Log progress
            logger.info(f"✓ Executed: {node_name}")

            # If notify_user node, we could send SSE here
            if node_name == "notify_user":
                logger.warning(f"Strategy change: {node_output['user_notifications'][-1]['message']}")

            final_state = node_output

        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        final_state["processing_time_ms"] = processing_time

        return final_state["final_response"]


async def stream_workflow_execution(app, initial_state):
    """
    Stream workflow execution with user notifications
    Yields events for SSE/WebSocket
    """
    async for state in app.astream(initial_state):
        node_name = list(state.keys())[0]
        node_output = state[node_name]

        # Yield different events based on node
        if node_name == "notify_user":
            notification = node_output["user_notifications"][-1]
            yield {
                "event": "strategy_change",
                "data": notification
            }

        elif node_name == "execute_search":
            yield {
                "event": "search_started",
                "data": {
                    "strategy": node_output["current_strategy"],
                    "attempt": node_output["attempt_count"]
                }
            }

        elif node_name == "evaluate_results":
            yield {
                "event": "evaluation_complete",
                "data": {
                    "is_satisfied": node_output["is_satisfied"],
                    "scores": node_output["evaluation_scores"]
                }
            }

        elif node_name == "generate_answer":
            # Stream answer token by token (if LLM supports streaming)
            answer = node_output["generated_answer"]
            for token in answer.split():
                yield {
                    "event": "token",
                    "data": {"token": token}
                }

        elif node_name in ["return_response", "return_partial"]:
            yield {
                "event": "done",
                "data": node_output["final_response"]
            }
```

### API Endpoints

```python
from fastapi import FastAPI, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Agentic RAG API")


class QueryRequest(BaseModel):
    query: str
    context: Optional[dict] = {}
    session_id: Optional[str] = None
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: list
    metadata: dict


@app.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Main query endpoint

    Example:
    POST /api/v1/query
    {
      "query": "What is the coverage limit for HK medical insurance?",
      "context": {"branch": "HK", "language": "en"},
      "stream": false
    }
    """
    session_id = request.session_id or generate_session_id()

    response = await process_query_with_langgraph(
        query=request.query,
        session_id=session_id,
        user_context=request.context,
        stream=False
    )

    return response


@app.post("/api/v1/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """
    Streaming query endpoint
    Returns Server-Sent Events (SSE)

    Events:
    - strategy_change: When strategy switches (with notification message)
    - search_started: When search executes
    - evaluation_complete: When results evaluated
    - token: Answer tokens (during generation)
    - done: Final response
    """
    session_id = request.session_id or generate_session_id()

    async def event_generator():
        async for event in process_query_with_langgraph(
            query=request.query,
            session_id=session_id,
            user_context=request.context,
            stream=True
        ):
            yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

---

## 📊 Key Features Summary

### ✅ What This System Does

1. **Intelligent Query Analysis**: Uses LLM to understand query intent, complexity, and requirements

2. **Strategy Selection**: LLM-based reasoning to select optimal search strategy from 13 options

3. **Self-Evaluation**: After each search, LLM evaluates result quality on multiple dimensions

4. **Iterative Refinement**: If results unsatisfactory, automatically switches strategy and retries

5. **User Notifications**: Sends real-time updates: *"We didn't find the result yet, changing search strategy, please hold on"*

6. **Max Attempts Limit**: Stops after 3 attempts to prevent infinite loops

7. **Transparent Metadata**: Returns full execution trace (strategies tried, reasons, evaluations)

8. **Answer Quality Check**: Self-evaluates generated answer for groundedness and completeness

9. **Streaming Support**: Can stream notifications and answers in real-time via SSE

10. **State Persistence**: LangGraph checkpointing allows resuming from any point

---

## 🎯 User Experience Flow

### Example: Successful Query After Strategy Switch

```
USER: "What is the coverage limit for HK medical insurance?"

SYSTEM INTERNAL:
1. analyze_query → Type: question, Complexity: medium
2. select_strategy → Chose: qa_pairs (for direct question)
3. execute_search → Searching QA pairs...
4. evaluate_results → Score: 0.3 (NOT SATISFIED - no matching QA)
5. notify_user → 🔔 "We didn't find the result yet (attempt 1/3). Previous strategy 'qa_pairs' found no matching QA pairs. Changing search strategy, please hold on..."
6. replan_strategy → Chose: hybrid_search (keyword + vector)
7. execute_search → Searching with hybrid...
8. evaluate_results → Score: 0.85 (SATISFIED!)
9. generate_answer → Generating answer from docs...
10. evaluate_answer → Confidence: 0.87
11. return_response → Done!

USER SEES:
[Notification after attempt 1]
"We didn't find the result yet, changing search strategy, please hold on..."

[Final Response]
"The coverage limit for HK medical insurance is HK$50,000 per year for outpatient services and HK$500,000 for inpatient services. [Source 1] [Source 2]"

Confidence: 0.87
Sources: [Medical Insurance Brochure 2024.pdf, Page 5], [...]
Metadata: {
  "attempts": 2,
  "strategies_tried": ["qa_pairs", "hybrid_search"],
  "strategy_reasons": {...}
}
```

---

## 🚀 Implementation Checklist

### Phase 1: Core LangGraph Setup (Week 1)
- [ ] Define `AgentState` TypedDict
- [ ] Implement `create_agentic_rag_workflow()` function
- [ ] Build `should_continue_or_replan()` decision function
- [ ] Set up LangGraph checkpointing

### Phase 2: Node Implementations (Week 2-3)
- [ ] Node 1: `analyze_query_node`
- [ ] Node 2: `select_strategy_node`
- [ ] Node 3: `execute_search_node`
- [ ] Node 4: `evaluate_results_node`
- [ ] Node 5: `notify_user_node`
- [ ] Node 6: `replan_strategy_node`
- [ ] Node 7: `generate_answer_node`
- [ ] Node 8: `evaluate_answer_node`
- [ ] Node 9: `return_response_node`
- [ ] Node 10: `return_partial_node`

### Phase 3: Search Strategy Engine (Week 3-4)
- [ ] Implement all 13 search strategies from SEARCH_STRATEGY.md
- [ ] Build `SearchStrategyEngine` class
- [ ] Add Azure AI Search integration
- [ ] Add embedding service integration

### Phase 4: API Layer (Week 4-5)
- [ ] FastAPI app setup
- [ ] `/api/v1/query` endpoint (non-streaming)
- [ ] `/api/v1/query/stream` endpoint (SSE streaming)
- [ ] Session management
- [ ] Authentication & authorization

### Phase 5: Testing (Week 5-6)
- [ ] Unit tests for each node
- [ ] Integration tests for workflow
- [ ] End-to-end API tests
- [ ] Test strategy switching logic
- [ ] Test max attempts limit
- [ ] Test notification system

### Phase 6: Optimization & Deployment (Week 6-7)
- [ ] Performance tuning
- [ ] Caching layer (Redis)
- [ ] Monitoring & logging
- [ ] Docker containerization
- [ ] Azure deployment

---

## 📝 Sample Configuration

```python
# config.py

# LangGraph Configuration
MAX_RETRY_ATTEMPTS = 3
EVALUATION_THRESHOLD = 0.7  # Minimum score to be satisfied

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
AZURE_OPENAI_API_KEY = "your-api-key"
AZURE_OPENAI_DEPLOYMENT = "gpt-4o"
AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

# Azure AI Search Configuration
AZURE_SEARCH_ENDPOINT = "https://your-search.search.windows.net"
AZURE_SEARCH_API_KEY = "your-search-key"
AZURE_SEARCH_INDEX_NAME = "insurance-documents-index"

# Embedding Configuration
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 1536

# Search Configuration
DEFAULT_TOP_K = 10
DEFAULT_STRATEGY = "hybrid_search"

# Session Configuration
SESSION_TIMEOUT_MINUTES = 30
MAX_CONVERSATION_HISTORY = 10
```

---

## 📚 Dependencies

```txt
# requirements.txt

# LangChain & LangGraph
langchain==0.1.0
langgraph==0.0.20
langchain-openai==0.0.5

# Azure
azure-search-documents==11.4.0
azure-identity==1.15.0
openai==1.10.0

# FastAPI
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-multipart==0.0.6
sse-starlette==1.8.2

# Database
sqlalchemy==2.0.25
psycopg2-binary==2.9.9
redis==5.0.1

# Utilities
pydantic==2.5.3
python-dotenv==1.0.0
httpx==0.26.0
numpy==1.26.3

# Monitoring
applicationinsights==0.11.10
python-json-logger==2.0.7
```

---

**Document Version**: 2.0
**Last Updated**: 2025-11-06
**Status**: Implementation Ready
**Author**: Agentic RAG Team
