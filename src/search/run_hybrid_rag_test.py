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

from src.search.hybrid_rag import HybridRAGSystem
from src.search.query_vector import VectorSearcher


def main():
    parser = argparse.ArgumentParser(description="Run a single HybridRAG query and print answer + citations")
    parser.add_argument("--query", type=str, default="理財顧問海外會議 2023 的出席資格是什麼？", help="Query to run")
    parser.add_argument("--top-k", type=int, default=6, help="Top K chunks to retrieve")
    parser.add_argument("--k-neighbors", type=int, default=100, help="Vector K neighbors")
    parser.add_argument("--document-id", type=str, default="GI000001", help="Optional document_id filter (e.g., GI000001)")
    parser.add_argument("--branch", type=str, default=None, help="Optional branch filter (HK/MACAU)")
    args = parser.parse_args()

    # Env config
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

    rag = HybridRAGSystem(search_client, embedding_function, openai_client)
    # Prefer CHAT_* envs; fallback to AZURE_OPENAI_* for chat deployment name
    chat_model = os.getenv("CHAT_MODEL") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    result = rag.generate_answer(
        query=args.query,
        top_k=args.top_k,
        k_neighbors=args.k_neighbors,
        model=chat_model if chat_model else "",
        document_id_filter=args.document_id,
        branch_filter=args.branch,
    )

    # Fallback: if no citations returned with filters, retry without filters
    if not result.get("sources"):
        print("[Info] No citations found with filters; retrying without filters...")  # noqa: T201
        result = rag.generate_answer(
            query=args.query,
            top_k=args.top_k,
            k_neighbors=args.k_neighbors,
            model=chat_model if chat_model else "",
            document_id_filter=None,
            branch_filter=None,
        )

    print("Question:", result["query"])  # noqa: T201
    print("\nAnswer:", result["answer"])  # noqa: T201
    print("\nSources:")  # noqa: T201
    for src in result["sources"]:
        fn = src.get("file_name")
        pg = src.get("page")
        score = src.get("search_score")
        print(f"  [Source {src['rank']}] {fn} (page {pg}) score={score}")  # noqa: T201
    print(f"\nTokens used: {result['token_usage']['total_tokens']}")  # noqa: T201


if __name__ == "__main__":
    main()