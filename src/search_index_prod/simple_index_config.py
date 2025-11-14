import os
from typing import Dict, Any


def build_simple_index_schema(index_name: str) -> Dict[str, Any]:
    """
    Define a simple hybrid index for bilingual chunks:
    - Lexical: content_en (en.microsoft), content_tc (zh-Hant.lucene)
    - Semantic: content_chunk_vector (1536 dims, cosine, HNSW m=8)
    - Metadata: filename, branch_name, entities, page_number, word_count, char_count,
                title_name_en, title_name_tc, document_id, lang_tags

    This schema keeps one document per chunk, 
    with bilingual content fields (content_en, content_tc) and vector for semantic search.
    """

    return {
        "name": index_name,
        "fields": [
            {
                "name": "id", # filename should be the key 
                "type": "Edm.String",
                "key": True,
                "filterable": True,
                "sortable": False,
                "facetable": False,
                "searchable": False,
            },
            # Display field (optional). Keep entire bilingual text retrievable.
         
            # Dual BM25 fields for proper tokenization per language
            {
                "name": "content_en",
                "type": "Edm.String",
                "searchable": True,
                "analyzer": "en.microsoft",
            },
            {
                "name": "content_tc",
                "type": "Edm.String",
                "searchable": True,
                "analyzer": "zh-Hant.lucene",
            },
            # Shared vector for semantic search
            {
                "name": "content_chunk_vector",
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "dimensions": 1536,
                "vectorSearchProfile": "simple-vector-profile",
            },
            # Titles for stronger lexical matches
            {
                "name": "title_name_en",
                "type": "Edm.String",
                "searchable": True,
                "analyzer": "en.microsoft",
            },
            {
                "name": "title_name_tc",
                "type": "Edm.String",
                "searchable": True,
                "analyzer": "zh-Hant.lucene",
            },
            # Core metadata
            {
                "name": "filename",
                "type": "Edm.String",
                "filterable": True,
                "facetable": False,
                "searchable": False,
            },
            {
                "name": "branch_name",
                "type": "Edm.String",
                "filterable": True,
                "facetable": True,
                "searchable": False,
            },
            {
                "name": "document_id",
                "type": "Edm.String",
                "filterable": True,
                "facetable": True,
                "searchable": False,
            },
            {
                "name": "entities",
                "type": "Collection(Edm.String)",
                "filterable": True,
                "facetable": True,
                "searchable": False,
            },
            {
                "name": "page_number",
                "type": "Edm.Int32",
                "filterable": True,
                "sortable": True,
                "facetable": False,
            },
            {
                "name": "word_count",
                "type": "Edm.Int32",
                "filterable": True,
                "sortable": True,
                "facetable": False,
            },
            {
                "name": "char_count",
                "type": "Edm.Int32",
                "filterable": True,
                "sortable": True,
                "facetable": False,
            },
            {
                "name": "lang_tags",
                "type": "Collection(Edm.String)",
                "filterable": True,
                "facetable": True,
                "searchable": False,
            },
        ],
        "vectorSearch": {
            "algorithms": [
                {
                    "name": "hybrid-hnsw",
                    "kind": "hnsw",
                    "hnswParameters": {
                        "metric": "cosine",
                        "m": 8,
                        "efConstruction": 400,
                        "efSearch": 500,
                    },
                }
            ],
            "profiles": [
                {
                    "name": "simple-vector-profile",
                    "algorithm": "hybrid-hnsw",
                }
            ],
        },
    }