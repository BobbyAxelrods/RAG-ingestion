import os
import json
import argparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from openai import AzureOpenAI

try:
    from src.search_prod.hybrid_search import AzureHybridSearch
except Exception:
    from hybrid_search import AzureHybridSearch  # type: ignore


def _get_env(primary: str, *alternates: str, default=None, required=False):
    for key in (primary, *alternates):
        val = os.getenv(key)
        if val:
            return val
    if required and default is None:
        alt = ", ".join(alternates) if alternates else ""
        raise RuntimeError(f"Missing environment variable: {primary}{f' (alternatives: {alt})' if alt else ''}")
    return default


def _build_clients(index_name: str):
    endpoint = _get_env("SEARCH_SERVICE_ENDPOINT", "AZURE_SEARCH_ENDPOINT", required=True)
    api_key = _get_env("SEARCH_SERVICE_KEY", "AZURE_SEARCH_API_KEY", required=True)
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))

    aoai_endpoint = _get_env("AZURE_OPENAI_ENDPOINT", required=True)
    aoai_key = _get_env("AZURE_OPENAI_KEY", "AZURE_OPENAI_API_KEY", required=True)
    aoai_api_version = _get_env("AZURE_OPENAI_API_VERSION", default="2024-02-01")
    openai_client = AzureOpenAI(api_key=aoai_key, api_version=aoai_api_version, azure_endpoint=aoai_endpoint)
    return search_client, openai_client


def main():
    parser = argparse.ArgumentParser(description="Run hybrid search (BM25 + vector with RRF fusion) against Azure AI Search.")
    parser.add_argument("--query", "-q", required=True, help="Search query text.")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of results to return.")
    parser.add_argument(
        "--index", "-i",
        default=os.getenv("AZURE_SEARCH_INDEX_NAME") or os.getenv("SEARCH_INDEX_NAME") or "experiment_full_2",
        help="Index name to query.",
    )
    parser.add_argument("--lang", "-l", choices=["auto", "en", "tc"], default="auto", help="Query language routing.")
    parser.add_argument("--filter", default=None, help="OData filter expression (applied post-fusion).")
    parser.add_argument(
        "--embed-deployment",
        default=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME") or "text-embedding-3-small",
        help="Embedding deployment name in Azure OpenAI.",
    )
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    parser.add_argument("--scores", action="store_true", help="Show component scores (bm25/vector/hybrid).")
    parser.add_argument("--out", "-o", default=None, help="Path to save results JSON.")
    parser.add_argument("--answer", action="store_true", help="Generate RAG answer from retrieved context using Azure OpenAI.")
    parser.add_argument(
        "--chat-deployment",
        default=os.getenv("CHAT_MODEL") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        help="Chat deployment name for Azure OpenAI when generating an answer.",
    )
    args = parser.parse_args()

    search_client, openai_client = _build_clients(args.index)
    hs = AzureHybridSearch(search_client, openai_client)
    hs.embedding_deployment = args.embed_deployment

    query_lang = None if args.lang == "auto" else ("en" if args.lang == "en" else "tc")

    if args.scores:
        results = hs.hybrid_search_with_score_breakdown(args.query, top_k=args.top_k, query_lang=query_lang)
    else:
        results = hs.hybrid_search(args.query, top_k=args.top_k, filter_expression=args.filter, query_lang=query_lang)

    # Enrich results with desired fields for JSON saving
    def _enrich(results_list):
        enriched = []
        for r in results_list:
            # Prefer content in requested language, else fallback
            content_chunk = None
            if query_lang == "tc":
                content_chunk = r.get("content_tc") or r.get("content_en")
            elif query_lang == "en":
                content_chunk = r.get("content_en") or r.get("content_tc")
            else:
                content_chunk = r.get("content_en") or r.get("content_tc")
            enriched.append({
                **r,
                "content_chunk": content_chunk,
                "filename": r.get("filename"),
                "page_number": r.get("page_number"),
            })
        return enriched

    # Optional: Generate an answer using retrieved context
    if args.answer:
        enriched = _enrich(results)
        # Build RAG prompt from context
        context_blocks = []
        for i, r in enumerate(enriched, start=1):
            fn = (r.get("filename") or "").strip()
            pg = r.get("page_number")
            content = (r.get("content_chunk") or "").strip()
            if not content:
                continue
            context_blocks.append(f"[Source {i}] {fn} (page {pg})\n{content}")
        context_text = "\n\n".join(context_blocks)

        system_message = (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Always cite your sources using [Source N]. If the context lacks sufficient information, say so clearly."
        )
        user_message = f"Context:\n{context_text}\n\nQuestion: {args.query}\n\nProvide a concise, accurate answer and cite sources."

        if not args.chat_deployment:
            raise RuntimeError("Missing chat deployment name in --chat-deployment or environment.")

        resp = openai_client.chat.completions.create(
            model=args.chat_deployment,
            messages=[{"role": "system", "content": system_message}, {"role": "user", "content": user_message}],
            temperature=0.2,
            max_tokens=700,
        )

        answer_text = resp.choices[0].message.content
        citations = [
            {
                "rank": i,
                "filename": (r.get("filename") or "").strip(),
                "page_number": r.get("page_number"),
                "score": r.get("hybrid_score") or r.get("score") or r.get("@search.score"),
            }
            for i, r in enumerate(enriched, start=1)
        ]

        payload_obj = {
            "query": args.query,
            "answer": answer_text,
            "citations": citations,
            "results": enriched,
            "token_usage": {
                "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                "total_tokens": getattr(resp.usage, "total_tokens", None),
            },
        }

        payload = json.dumps(payload_obj, ensure_ascii=False, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(payload)
            print(f"Saved results to {args.out}")
        else:
            print(payload)
        return

    # Default: print results or save JSON list
    if args.json or args.out:
        enriched = _enrich(results)
        payload = json.dumps(enriched, ensure_ascii=False, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(payload)
            print(f"Saved results to {args.out}")
        else:
            print(payload)
    else:
        for i, r in enumerate(results, 1):
            title = r.get("title_en") or r.get("title_tc") or "(no title)"
            score = r.get("hybrid_score") or r.get("score") or r.get("@search.score")
            lang = "en" if r.get("title_en") else ("tc" if r.get("title_tc") else "")
            print(f"[{i}] {title} | score={score} {f'| lang={lang}' if lang else ''}")
            if not args.scores:
                snippet = r.get("content_en") or r.get("content_tc") or ""
                if snippet:
                    print(f"    {snippet[:180].replace('\n', ' ')}{'...' if len(snippet) > 180 else ''}")


if __name__ == "__main__":
    main()