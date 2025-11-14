import argparse
import csv
import os
import asyncio
from typing import Any, Dict, Optional

from .orchestrator import create_agentic_rag_workflow
from .state import AgentState, SearchStrategy


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v else default


def _parse_strategy(s: Optional[str]) -> Optional[SearchStrategy]:
    if not s:
        return None
    try:
        return SearchStrategy(s.lower())
    except Exception:
        return None


def run_single(query: str, index_name: Optional[str], endpoint: Optional[str], api_key: Optional[str], strategy: Optional[str], top_k: int) -> Dict[str, Any]:
    app = create_agentic_rag_workflow()
    override = _parse_strategy(strategy)
    confidence_threshold = float(_env("CONFIDENCE_THRESHOLD", "0.8") or "0.8")
    initial_state: AgentState = {
        "query": query,
        "session_id": "cli",
        "user_context": {"branch": None, "category": None, "library": None, "filename": None, "language": "en"},
        "conversation_history": [],
        "query_intent": None,
        "current_strategy": override,
        "strategies_tried": [],
        "strategy_reasons": {},
        "attempt_count": 0,
        "retrieved_documents": [],
        "raw_search_results": {},
        "search_metadata": {"top_k": top_k},
        "retrieval_evaluation": None,
        "is_satisfied": False,
        "failure_reason": "",
        "evaluation_scores": {},
        "user_notifications": [],
        "generated_answer": None,
        "answer_evaluation": None,
        "answer_confidence": 0.0,
        "final_response": None,
        "execution_metadata": {},
        "processing_time_ms": 0,
        "confidence_threshold": confidence_threshold,
    }

    result_state: AgentState = asyncio.run(app.ainvoke(initial_state, config={"recursion_limit": 60}))
    resp = result_state.get("final_response") or {}
    return {
        "query": query,
        "index_name": index_name,
        "strategy_used": resp.get("strategy_used"),
        "attempts": resp.get("attempts"),
        "confidence": round(result_state.get("answer_confidence", 0.0), 4),
        "result_count": resp.get("result_count"),
        "top_file": resp.get("top_file"),
        "top_page": resp.get("top_page"),
        "top_snippet": resp.get("top_snippet"),
        "top_question": "",
        "top_answer": resp.get("answer"),
    }


def run_csv(csv_input: str, csv_output: str, index_name: Optional[str], endpoint: Optional[str], api_key: Optional[str], default_strategy: Optional[str], top_k: int) -> None:
    app = create_agentic_rag_workflow()

    with open(csv_input, newline='', encoding='utf-8') as f_in, open(csv_output, 'w', newline='', encoding='utf-8') as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = [
            "query",
            "strategy_requested",
            "strategy_used",
            "attempts",
            "confidence",
            "result_count",
            "top_file",
            "top_page",
            "top_snippet",
            "top_question",
            "top_answer",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            q = (row.get("query") or "").strip()
            s_req = (row.get("strategy") or default_strategy)
            if not q:
                continue
            override = _parse_strategy(s_req)
            confidence_threshold = float(_env("CONFIDENCE_THRESHOLD", "0.8") or "0.8")
            initial_state: AgentState = {
                "query": q,
                "session_id": "csv",
                "user_context": {"branch": None, "category": None, "library": None, "filename": None, "language": "en"},
                "conversation_history": [],
                "query_intent": None,
                "current_strategy": override,
                "strategies_tried": [],
                "strategy_reasons": {},
                "attempt_count": 0,
                "retrieved_documents": [],
                "raw_search_results": {},
                "search_metadata": {"top_k": top_k},
                "retrieval_evaluation": None,
                "is_satisfied": False,
                "failure_reason": "",
                "evaluation_scores": {},
                "user_notifications": [],
                "generated_answer": None,
                "answer_evaluation": None,
                "answer_confidence": 0.0,
                "final_response": None,
                "execution_metadata": {},
                "processing_time_ms": 0,
                "confidence_threshold": confidence_threshold,
            }
            result_state: AgentState = asyncio.run(app.ainvoke(initial_state, config={"recursion_limit": 60}))
            resp = result_state.get("final_response") or {}
            writer.writerow({
                "query": q,
                "strategy_requested": s_req or "",
                "strategy_used": resp.get("strategy_used"),
                "attempts": resp.get("attempts"),
                "confidence": round(result_state.get("answer_confidence", 0.0), 4),
                "result_count": resp.get("result_count"),
                "top_file": resp.get("top_file"),
                "top_page": resp.get("top_page"),
                "top_snippet": resp.get("top_snippet"),
                "top_question": "",
                "top_answer": resp.get("answer"),
            })


def main():
    parser = argparse.ArgumentParser(description="Agentic RAG CLI: single or CSV batch queries")
    parser.add_argument("--query", type=str, help="Single query to run", default=None)
    parser.add_argument("--strategy", type=str, help="Strategy override (qa_pairs, hybrid_search, entity_search, document_search, summary_search)", default=None)
    parser.add_argument("--top-k", type=int, help="Top K results", default=5)
    parser.add_argument("--index-name", type=str, help="Azure Search index name (defaults to env)", default=None)
    parser.add_argument("--endpoint", type=str, help="Azure Search endpoint (defaults to env)", default=None)
    parser.add_argument("--api-key", type=str, help="Azure Search api key (defaults to env)", default=None)
    parser.add_argument("--csv-input", type=str, help="Path to CSV with column 'query' and optional 'strategy'", default=None)
    parser.add_argument("--csv-output", type=str, help="Path to output CSV results", default=None)

    args = parser.parse_args()

    # Force ONLINE mode unless explicitly overridden by env
    os.environ["OFFLINE_MODE"] = os.getenv("OFFLINE_MODE", "false") or "false"

    endpoint = args.endpoint or _env("AZURE_SEARCH_ENDPOINT")
    api_key = args.api_key or _env("AZURE_SEARCH_API_KEY")
    index_name = args.index_name or _env("AZURE_SEARCH_INDEX_NAME")

    if args.csv_input:
        if not args.csv_output:
            raise SystemExit("--csv-output is required when using --csv-input")
        run_csv(args.csv_input, args.csv_output, index_name, endpoint, api_key, args.strategy, args.top_k)
        return

    if not args.query:
        raise SystemExit("Provide --query for single run or --csv-input for batch")

    res = run_single(args.query, index_name, endpoint, api_key, args.strategy, args.top_k)
    # Lightweight stdout printing for CLI usage
    print(f"Query: {res['query']}")
    print(f"Index: {res['index_name']}")
    print(f"Strategy: {res['strategy_used']} (attempts={res['attempts']})")
    print(f"Confidence: {res['confidence']}  Results: {res['result_count']}")
    print(f"Top: file={res['top_file']} page={res['top_page']}")
    if res.get("top_answer"):
        if res.get("top_question"):
            print(f"Question: {res['top_question']}")
        print(f"Answer: {res['top_answer']}")
    else:
        print(f"Snippet: {res['top_snippet']}")


if __name__ == "__main__":
    main()