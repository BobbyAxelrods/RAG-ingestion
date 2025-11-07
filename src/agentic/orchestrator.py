from typing import Any, Dict

from langgraph.graph import StateGraph, END
# Checkpointing disabled for compatibility with current langgraph version

from .state import AgentState
from .nodes import (
    analyze_query_node,
    select_strategy_node,
    execute_search_node,
    evaluate_results_node,
    notify_user_node,
    replan_strategy_node,
    generate_answer_node,
    evaluate_answer_node,
    return_partial_node,
    return_response_node,
    should_continue_or_replan,
    should_accept_answer,
)


def create_agentic_rag_workflow() -> Any:
    """Build and compile the LangGraph StateGraph workflow."""
    workflow = StateGraph(AgentState)

    # Nodes
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
    from .nodes import return_best_node
    workflow.add_node("return_best", return_best_node)

    # Entry
    workflow.set_entry_point("analyze_query")

    # Initial path
    workflow.add_edge("analyze_query", "select_strategy")
    workflow.add_edge("select_strategy", "execute_search")
    workflow.add_edge("execute_search", "evaluate_results")

    # Conditional routing
    workflow.add_conditional_edges(
        "evaluate_results",
        should_continue_or_replan,
        {
            "satisfied": "generate_answer",
            "retry": "notify_user",
            "max_attempts": "return_partial",
            "all_done": "return_best",
        },
    )

    # Retry loop
    workflow.add_edge("notify_user", "replan_strategy")
    workflow.add_edge("replan_strategy", "execute_search")

    # Answer path with confidence gating
    workflow.add_edge("generate_answer", "evaluate_answer")
    workflow.add_conditional_edges(
        "evaluate_answer",
        should_accept_answer,
        {
            "accept": "return_response",
            "retry": "notify_user",
            "max_attempts": "return_partial",
            "all_done": "return_best",
        },
    )

    # End
    workflow.add_edge("return_response", END)
    workflow.add_edge("return_partial", END)
    workflow.add_edge("return_best", END)

    # Compile without checkpointing
    app = workflow.compile()
    return app