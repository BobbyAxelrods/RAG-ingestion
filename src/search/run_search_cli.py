import os
import argparse

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from src.search.query_vector import VectorSearcher
from src.search.strategies import HybridRAGStrategy, HierarchicalRAGStrategy
from src.search.semantic_search import SemanticRAGStrategy
from src.search.hirarchical_hybrid_search import HierarchicalHybridSearcher


def build_clients():
    endpoint = os.getenv("SEARCH_SERVICE_ENDPOINT") or os.getenv("AZURE_SEARCH_ENDPOINT")
    api_key = os.getenv("SEARCH_SERVICE_KEY") or os.getenv("AZURE_SEARCH_API_KEY")
    index_name = os.getenv("SEARCH_INDEX_NAME") or os.getenv("AZURE_SEARCH_INDEX_NAME")
    if not endpoint or not api_key or not index_name:
        raise RuntimeError("Missing SEARCH_SERVICE_ENDPOINT/KEY/INDEX_NAME in environment")

    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    aoai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    aoai_key = os.getenv("AZURE_OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
    aoai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    aoai_api_version = os.getenv("AZURE_OPENAI_API_VERSION") or "2024-02-01"
    if not aoai_endpoint or not aoai_key or not aoai_deployment:
        raise RuntimeError("Missing Azure OpenAI env (endpoint/key/deployment)")
    openai_client = AzureOpenAI(api_key=aoai_key, api_version=aoai_api_version, azure_endpoint=aoai_endpoint)

    # Embedding function using your deployment
    vec = VectorSearcher(search_client, openai_client, deployment_name=aoai_deployment)
    embedding_function = vec.generate_embedding

    # Prefer CHAT_* envs; fallback to AZURE_OPENAI_* for chat deployment name
    chat_model = os.getenv("CHAT_MODEL") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    return search_client, openai_client, embedding_function, chat_model


def main():
    parser = argparse.ArgumentParser(description="Run RAG query with selectable strategy")
    parser.add_argument("--mode", type=str, default="hybrid_rag", choices=["hybrid_rag", "hierarchical_rag", "semantic"], help="Search strategy to use")
    parser.add_argument("--query", type=str, required=True, help="Query to run")
    parser.add_argument("--top-k", type=int, default=6, help="Top K chunks to retrieve")
    parser.add_argument("--k-neighbors", type=int, default=100, help="Vector K neighbors (hybrid/hierarchical)")
    parser.add_argument("--document-id", type=str, default=None, help="Optional document_id filter (e.g., GI000001)")
    parser.add_argument("--branch", type=str, default=None, help="Optional branch filter (HK/MACAU)")
    args = parser.parse_args()

    search_client, openai_client, embedding_function, chat_model = build_clients()

    if args.mode == "hybrid_rag":
        strategy = HybridRAGStrategy(search_client, openai_client, embedding_function)
        result = strategy.generate_answer(
            query=args.query,
            top_k=args.top_k,
            k_neighbors=args.k_neighbors,
            model=chat_model or "",
            document_id_filter=args.document_id,
            branch_filter=args.branch,
        )
    elif args.mode == "hierarchical_rag":
        strategy = HierarchicalRAGStrategy(search_client, openai_client, embedding_function, HierarchicalHybridSearcher)
        result = strategy.generate_answer(
            query=args.query,
            top_k=args.top_k,
            k_neighbors=args.k_neighbors,
            model=chat_model or "",
            document_id_filter=args.document_id,
            branch_filter=args.branch,
        )
    else:  # semantic
        strategy = SemanticRAGStrategy(search_client, openai_client)
        result = strategy.generate_answer(query=args.query, top_k=args.top_k, model=chat_model or "")

    print("Question:", result["query"])  # noqa: T201
    print("\nAnswer:", result["answer"])  # noqa: T201
    print("\nSources:")  # noqa: T201
    for src in result["sources"]:
        fn = src.get("file_name")
        pg = src.get("page")
        score = src.get("search_score")
        rscore = src.get("rerank_score")
        print(f"  [Source {src['rank']}] {fn} (page {pg}) score={score} rerank={rscore}")  # noqa: T201
    print(f"\nTokens used: {result['token_usage']['total_tokens']}")  # noqa: T201


if __name__ == "__main__":
    main()