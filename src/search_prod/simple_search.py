import re
from typing import List, Optional, Dict, Any

try:
    from azure.search.documents import SearchClient
    from azure.search.documents.models import QueryType
except Exception:
    SearchClient = None  # type: ignore
    QueryType = None  # type: ignore


class AzureSimpleSearch:
    def __init__(self, search_client: "SearchClient"):
        import os
        self.search_client = search_client
        def _csv(name, default):
            v = os.getenv(name)
            return [x.strip() for x in v.split(",")] if v else default
        self.default_select_fields: List[str] = _csv(
            "SELECT_FIELDS",
            [
                "document_id",
                "filename",
                "page_number",
                "branch_name",
                "title_name_en",
                "title_name_tc",
                "content_en",
                "content_tc",
                "lang_tags",
                "entities",
                "word_count",
                "char_count",
            ],
        )
        self.search_fields_en: List[str] = _csv("SEARCH_FIELDS_EN", ["content_en", "title_name_en"])
        self.search_fields_tc: List[str] = _csv("SEARCH_FIELDS_TC", ["content_tc", "title_name_tc"])

    def _detect_lang_fields(self, query: str, query_lang: Optional[str]) -> List[str]:
        if query_lang is None:
            if re.search(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]", query):
                return self.search_fields_tc
            elif re.search(r"[A-Za-z]", query):
                return self.search_fields_en
            else:
                return self.search_fields_en + self.search_fields_tc
        return self.search_fields_en if query_lang == "en" else self.search_fields_tc

    def simple_search(
        self,
        query: str,
        top_k: int = 10,
        filter_expression: Optional[str] = None,
        query_lang: Optional[str] = None,
        search_fields: Optional[List[str]] = None,
        select_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if QueryType is None:
            raise RuntimeError("QueryType model not available in this environment")

        sf = search_fields or self._detect_lang_fields(query, query_lang)
        sel = select_fields or self.default_select_fields

        results = self.search_client.search(
            search_text=query,
            search_fields=sf or None,
            filter=filter_expression or None,
            top=top_k,
            query_type=QueryType.SIMPLE,
            select=sel or None,
        )

        out: List[Dict[str, Any]] = []
        try:
            _ = next(iter(results))
            results = self.search_client.search(
                search_text=query,
                search_fields=sf or None,
                filter=filter_expression or None,
                top=top_k,
                query_type=QueryType.SIMPLE,
                select=sel or None,
            )
        except Exception:
            results = self.search_client.search(
                search_text=query,
                search_fields=sf or None,
                filter=filter_expression or None,
                top=top_k,
                query_type=QueryType.SIMPLE,
                select=sel or None,
            )

        for r in results:
            entry: Dict[str, Any] = {}
            for f in sel:
                entry[f] = r.get(f)
            rid = r.get("id")
            base_id = entry.get("document_id") or entry.get("doc_id") or (rid.split("_")[0] if isinstance(rid, str) and "_" in rid else rid)
            entry["id"] = base_id
            entry["document_id"] = base_id
            entry["content_en"] = entry.get("content_en")
            entry["content_tc"] = entry.get("content_tc")
            entry["filename"] = entry.get("filename") or entry.get("file_name")
            entry["page_number"] = entry.get("page_number") or entry.get("chunk_page_number")
            entry["score"] = r.get("@search.score")
            chunk = entry.get("content_en") if (query_lang == "en") else (entry.get("content_tc") if (query_lang == "tc") else (entry.get("content_en") or entry.get("content_tc")))
            entry["content_chunk"] = chunk or ""
            out.append(entry)
        return out

    def simple_search_raw(
        self,
        query: str,
        top_k: int = 10,
        filter_expression: Optional[str] = None,
        query_lang: Optional[str] = None,
        search_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        if QueryType is None:
            raise RuntimeError("QueryType model not available in this environment")

        sf = search_fields or self._detect_lang_fields(query, query_lang)

        results = self.search_client.search(
            search_text=query,
            search_fields=sf or None,
            filter=filter_expression or None,
            top=top_k,
            query_type=QueryType.SIMPLE,
        )

        return [dict(r) for r in results]