import re
import time
from typing import Any, Dict, List, Optional
import json
import math

from langchain_openai import AzureChatOpenAI
from src.etl.services.openai_service import OpenAIService
from src.config import get_config

from .state import AgentState, QueryIntent, SearchStrategy
from .strategy_engine import SearchStrategyEngine
from .strategy_selector import StrategySelector


def _safe_get_llm() -> Optional[AzureChatOpenAI]:
    try:
        return AzureChatOpenAI(deployment_name="gpt-4o", temperature=0.0)
    except Exception:
        return None


def _heuristic_intent(query: str) -> QueryIntent:
    q = query.strip().lower()
    query_type = "question" if q.endswith("?") or re.search(r"\b(what|how|when|where|who|why|which)\b", q) else "statement"
    mentions_document = bool(re.search(r"\b(form|application|policy|guide|manual|brochure|document)\b", q))
    has_entities = bool(re.search(r"\b(hk\$|usd|premium|levy|credit card|visa|mastercard|macau|hong kong)\b", q))
    requires_multi_hop = bool(re.search(r"\b(compare|difference|versus|vs|pros|cons|calculate)\b", q))
    aspects = []
    language = "tc" if re.search(r"[\u4e00-\u9fff]", q) else "en"
    complexity = "high" if requires_multi_hop else ("medium" if has_entities else "low")
    return QueryIntent(
        query_type=query_type,
        complexity=complexity,
        mentions_document=mentions_document,
        has_entities=has_entities,
        requires_multi_hop=requires_multi_hop,
        aspects=aspects,
        language=language,
    )


async def analyze_query_node(state: AgentState) -> AgentState:
    llm = _safe_get_llm()
    intent: QueryIntent
    if llm:
        msg = [
            ("system", "You are an intent classifier. Return ONLY JSON with keys: query_type (question|statement), complexity (low|medium|high), mentions_document (bool), has_entities (bool), requires_multi_hop (bool), aspects (list of strings), language (en|tc)."),
            ("human", f"Query: {state['query']}")
        ]
        try:
            resp = await llm.ainvoke(msg)
            raw = resp.content if resp else ""
            parsed = {}
            try:
                parsed = json.loads(raw) if isinstance(raw, str) else {}
            except Exception:
                parsed = {}
            # Map and sanitize
            intent = {
                "query_type": str(parsed.get("query_type") or "").lower() or _heuristic_intent(state["query"])['query_type'],
                "complexity": str(parsed.get("complexity") or "").lower() or _heuristic_intent(state["query"])['complexity'],
                "mentions_document": bool(parsed.get("mentions_document")) if parsed.get("mentions_document") is not None else _heuristic_intent(state["query"])['mentions_document'],
                "has_entities": bool(parsed.get("has_entities")) if parsed.get("has_entities") is not None else _heuristic_intent(state["query"])['has_entities'],
                "requires_multi_hop": bool(parsed.get("requires_multi_hop")) if parsed.get("requires_multi_hop") is not None else _heuristic_intent(state["query"])['requires_multi_hop'],
                "aspects": list(parsed.get("aspects") or []),
                "language": str(parsed.get("language") or "").lower() or _heuristic_intent(state["query"])['language'],
            }
        except Exception:
            intent = _heuristic_intent(state["query"])
    else:
        intent = _heuristic_intent(state["query"])

    state["query_intent"] = intent
    return state


def _select_strategy(intent: QueryIntent, user_ctx: Dict[str, Any]) -> SearchStrategy:
    if intent["query_type"] == "question":
        return SearchStrategy.QA_PAIRS
    if intent["mentions_document"]:
        return SearchStrategy.DOCUMENT_SEARCH
    if intent["has_entities"]:
        return SearchStrategy.ENTITY_SEARCH
    if re.search(r"\b(overview|summary|brief|quick)\b", " ".join(intent.get("aspects", []))):
        return SearchStrategy.SUMMARY_SEARCH
    return SearchStrategy.HYBRID_SEARCH


async def select_strategy_node(state: AgentState) -> AgentState:
    selector = StrategySelector()
    override = state.get("current_strategy")
    strat, reason = selector.select(
        query=state.get("query") or "",
        intent=state.get("query_intent"),
        user_context=state.get("user_context", {}),
        override=override,
    )
    tried = state.get("strategies_tried") or []
    tried.append(strat.value)
    state["strategies_tried"] = tried
    state["current_strategy"] = strat
    state.setdefault("strategy_reasons", {})[strat.value] = reason
    return state


ALL_TRY_STRATEGIES: List[SearchStrategy] = [
    SearchStrategy.QA_PAIRS,
    SearchStrategy.HYBRID_SEARCH,
    SearchStrategy.SUMMARY_SEARCH,
    SearchStrategy.DOCUMENT_SEARCH,
    SearchStrategy.ENTITY_SEARCH,
]

def _fallback_strategy(current: SearchStrategy) -> SearchStrategy:
    order = ALL_TRY_STRATEGIES
    try:
        i = order.index(current)
        return order[(i + 1) % len(order)]
    except ValueError:
        return SearchStrategy.HYBRID_SEARCH


async def execute_search_node(state: AgentState) -> AgentState:
    engine = SearchStrategyEngine()
    strategy = state["current_strategy"] or SearchStrategy.HYBRID_SEARCH
    query = state["query"]
    ctx = state.get("user_context", {})
    top_k = int(state.get("search_metadata", {}).get("top_k", 10) or 10)

    if strategy == SearchStrategy.QA_PAIRS:
        results = engine.qa_search(query, ctx, top_k=5)
    elif strategy == SearchStrategy.ENTITY_SEARCH:
        # naive entity extraction
        m = re.search(r"(HK\$[0-9,]+|credit card|levy|premium)", query, re.IGNORECASE)
        entity = m.group(1) if m else None
        results = engine.entity_search(query, entity, ctx, top_k=10)
    elif strategy == SearchStrategy.SUMMARY_SEARCH:
        results = engine.summary_search(query, ctx, top_k=10)
    elif strategy == SearchStrategy.DOCUMENT_SEARCH:
        # naive filename hint
        m = re.search(r"([\w\s]+\.pdf)", query)
        fname = m.group(1) if m else ctx.get("filename")
        results = engine.document_search(query, filename=fname or "", top_k=100)
    else:
        results = engine.hybrid_search(query, ctx, top_k=10)

    state["retrieved_documents"] = results
    state["raw_search_results"] = {"count": len(results)}
    meta = state.get("search_metadata", {})
    meta["last_strategy"] = strategy.value
    state["search_metadata"] = meta
    state["attempt_count"] = int(state.get("attempt_count", 0)) + 1
    return state


async def evaluate_results_node(state: AgentState) -> AgentState:
    results = state.get("retrieved_documents", [])
    is_good = len(results) > 0
    state["is_satisfied"] = is_good
    state["retrieval_evaluation"] = {
        "result_count": len(results),
        "strategy": (state.get("current_strategy") or SearchStrategy.HYBRID_SEARCH).value,
    }
    return state


async def notify_user_node(state: AgentState) -> AgentState:
    msgs = state.get("user_notifications", [])
    tried = state.get("strategies_tried", [])
    cur = (state.get("current_strategy") or SearchStrategy.HYBRID_SEARCH).value
    msgs.append(f"Retrying with a new strategy. Tried={len(tried)}; current={cur}.")
    state["user_notifications"] = msgs
    return state


async def replan_strategy_node(state: AgentState) -> AgentState:
    cur = state.get("current_strategy") or SearchStrategy.HYBRID_SEARCH
    tried = set(state.get("strategies_tried", []))
    # Pick next unused strategy; if all used, keep current
    nxt: Optional[SearchStrategy] = None
    for s in ALL_TRY_STRATEGIES:
        if s.value not in tried:
            nxt = s
            break
    if nxt is None:
        nxt = _fallback_strategy(cur)
    state["current_strategy"] = nxt
    tried_list = list(state.get("strategies_tried", []))
    tried_list.append(nxt.value)
    state["strategies_tried"] = tried_list
    state.setdefault("strategy_reasons", {})[nxt.value] = f"Fallback from {cur.value}"
    return state


def _best_answer(results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    tokens = [t for t in query.strip().lower().replace("?", " ").split() if len(t) > 3]
    boost = {"credit", "card", "account", "authorize", "authorization", "levy", "premium", "payment", "payments", "recurring", "renewal", "renewals"}
    best = {"score": -1.0, "answer": "", "question": "", "file": None, "page": None}
    for doc in results:
        qa_ans = doc.get("qa_answers") or []
        qa_qs = doc.get("qa_questions") or []
        if isinstance(qa_ans, list) and isinstance(qa_qs, list):
            for i, a in enumerate(qa_ans):
                a_i = str(a).strip().lower()
                token_hits = sum(1 for t in tokens if t in a_i)
                token_hits += sum(3 for t in boost if t in a_i)
                if token_hits > best["score"]:
                    best = {
                        "score": float(token_hits),
                        "answer": str(a),
                        "question": str(qa_qs[i]) if i < len(qa_qs) else "",
                        "file": doc.get("file_name") or doc.get("sys_file_name"),
                        "page": doc.get("chunk_page_number"),
                    }
    return best


async def generate_answer_node(state: AgentState) -> AgentState:
    results = state.get("retrieved_documents", [])
    if not results:
        state["generated_answer"] = ""
        return state
    best = _best_answer(results, state["query"]) if results else {"answer": ""}
    if best.get("answer"):
        state["generated_answer"] = best["answer"]
    else:
        # Fallback to first chunk summary/content
        top = results[0]
        state["generated_answer"] = top.get("chunk_function_summary") or (top.get("chunk_content") or "").strip()[:600]
    return state


async def evaluate_answer_node(state: AgentState) -> AgentState:
    """Embedding-based evaluator: compute confidence via cosine similarities.

    Signals:
        - sim_qa: cosine(E(query), E(answer))
        - sim_ac_max: max cosine(E(answer), E(citation_i)) for top citations
    Confidence:
        - conf = 0.4 * sim_qa + 0.6 * sim_ac_max
    """
    query = state.get("query") or ""
    ans = state.get("generated_answer") or ""
    citations = state.get("retrieved_documents", [])

    # Early exit if no answer
    if not ans:
        state["answer_confidence"] = 0.0
        state["answer_evaluation"] = {"length": 0, "citations_used": 0, "sim_qa": 0.0, "sim_ac_max": 0.0, "method": "embeddings"}
        return state

    # Prepare context snippets for embedding
    def _citation_text(d: Dict[str, Any]) -> str:
        return (d.get("chunk_function_summary") or d.get("chunk_content") or "")[:600]

    texts = [_citation_text(d) for d in citations[:3] if _citation_text(d)]

    # Initialize embeddings client
    sim_qa = 0.0
    sim_ac_max = 0.0
    try:
        cfg = get_config()
        openai = OpenAIService(cfg.azure_openai)

        # Embed query and answer
        e_q = openai.generate_embedding(query)
        e_a = openai.generate_embedding(ans)

        # Helper: cosine similarity
        def _cos(u: List[float], v: List[float]) -> float:
            dot = sum((ui * vi) for ui, vi in zip(u, v))
            nu = math.sqrt(sum((ui * ui) for ui in u))
            nv = math.sqrt(sum((vi * vi) for vi in v))
            if nu == 0.0 or nv == 0.0:
                return 0.0
            return max(-1.0, min(1.0, dot / (nu * nv)))

        sim_qa = _cos(e_q, e_a)

        # Embed citations and compute max similarity to answer
        sim_ac_max = 0.0
        if texts:
            e_ctx_list = openai.generate_embeddings_batch(texts)
            for e_c in e_ctx_list:
                sim = _cos(e_a, e_c)
                if sim > sim_ac_max:
                    sim_ac_max = sim
    except Exception:
        # If embeddings unavailable, keep zeros (will trigger replanning if gated)
        sim_qa = 0.0
        sim_ac_max = 0.0

    # Combine signals into confidence [0,1]
    conf = 0.4 * sim_qa + 0.6 * sim_ac_max
    conf = max(0.0, min(1.0, conf))

    state["answer_confidence"] = conf
    state["answer_evaluation"] = {
        "length": len(ans),
        "citations_used": len(texts),
        "sim_qa": round(sim_qa, 4),
        "sim_ac_max": round(sim_ac_max, 4),
        "method": "embeddings",
        "formula": "0.4*sim_qa + 0.6*sim_ac_max",
    }
    # Track candidate for best-of return
    top = state.get("retrieved_documents", [{}])
    top0 = top[0] if top else {}
    cands = list(state.get("candidates", []))
    cands.append({
        "strategy_used": (state.get("current_strategy") or SearchStrategy.HYBRID_SEARCH).value,
        "confidence": float(conf),
        "result_count": len(top),
        "answer": ans,
        "top_file": top0.get("file_name") or top0.get("sys_file_name"),
        "top_page": top0.get("chunk_page_number"),
        "top_snippet": (top0.get("chunk_function_summary") or top0.get("chunk_content") or "")[:600],
        "answer_length": len(ans),
    })
    state["candidates"] = cands
    return state


async def return_partial_node(state: AgentState) -> AgentState:
    top = state.get("retrieved_documents", [{}])
    top0 = top[0] if top else {}
    state["final_response"] = {
        "status": "partial",
        "message": "Max attempts reached; returning best available results.",
        "answer": state.get("generated_answer") or "",
        "top_file": top0.get("file_name") or top0.get("sys_file_name"),
        "top_page": top0.get("chunk_page_number"),
        "top_snippet": (top0.get("chunk_function_summary") or top0.get("chunk_content") or "")[:600],
        "result_count": len(top),
        "strategy_used": (state.get("current_strategy") or SearchStrategy.HYBRID_SEARCH).value,
        "attempts": state.get("attempt_count", 1),
    }
    state["processing_time_ms"] = int((state.get("processing_time_ms") or 0) + 0)
    return state


async def return_response_node(state: AgentState) -> AgentState:
    top = state.get("retrieved_documents", [{}])
    top0 = top[0] if top else {}
    state["final_response"] = {
        "status": "ok",
        "answer": state.get("generated_answer"),
        "top_file": top0.get("file_name") or top0.get("sys_file_name"),
        "top_page": top0.get("chunk_page_number"),
        "top_snippet": (top0.get("chunk_function_summary") or top0.get("chunk_content") or "")[:600],
        "result_count": len(top),
        "strategy_used": (state.get("current_strategy") or SearchStrategy.HYBRID_SEARCH).value,
        "attempts": state.get("attempt_count", 1),
    }
    state["processing_time_ms"] = int((state.get("processing_time_ms") or 0) + 0)
    return state


def should_continue_or_replan(state: AgentState) -> str:
    if state.get("is_satisfied"):
        return "satisfied"
    tried = set(state.get("strategies_tried", []))
    all_count = len(ALL_TRY_STRATEGIES)
    if len(tried) >= all_count:
        return "all_done"
    # Safety: also gate by attempts in case strategies_tried isn't updating
    if int(state.get("attempt_count", 0) or 0) >= all_count:
        return "all_done"
    return "retry"


def should_accept_answer(state: AgentState) -> str:
    """Gate answer delivery by confidence threshold; otherwise trigger replanning.

    Returns:
        - "accept" when confidence >= threshold
        - "retry" when confidence < threshold and attempts remain
        - "max_attempts" when attempts exhausted
    """
    thresh = float(state.get("confidence_threshold", 0.8) or 0.8)
    conf = float(state.get("answer_confidence", 0.0) or 0.0)
    if conf >= thresh:
        return "accept"
    tried = set(state.get("strategies_tried", []))
    all_count = len(ALL_TRY_STRATEGIES)
    if len(tried) >= all_count:
        return "all_done"
    # Safety: also gate by attempts in case strategies_tried isn't updating
    if int(state.get("attempt_count", 0) or 0) >= all_count:
        return "all_done"
    return "retry"


async def return_best_node(state: AgentState) -> AgentState:
    """Select the best candidate across all tried strategies and return it."""
    cands: List[Dict[str, Any]] = list(state.get("candidates", []))
    if not cands:
        # Fallback to partial
        return await return_partial_node(state)
    # Sort by confidence desc, then result_count desc, then answer_length desc
    cands.sort(key=lambda c: (c.get("confidence", 0.0), c.get("result_count", 0), c.get("answer_length", 0)), reverse=True)
    best = cands[0]
    state["final_response"] = {
        "status": "ok",
        "answer": best.get("answer") or "",
        "top_file": best.get("top_file"),
        "top_page": best.get("top_page"),
        "top_snippet": best.get("top_snippet"),
        "result_count": best.get("result_count"),
        "strategy_used": best.get("strategy_used"),
        "attempts": len(state.get("strategies_tried", [])),
    }
    state["answer_confidence"] = float(best.get("confidence", 0.0) or 0.0)
    return state