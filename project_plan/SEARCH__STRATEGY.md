# Insurance Document Search Strategy Guide
## Based on Flat AI Search Index Schema

---

## 📊  Flat AI Search Schema 

```json
{
  "fields": [
    // PRIMARY KEY
    {"name": "doc_id", "type": "String", "key": true, "filterable": true},
    
    // SYSTEM METADATA
    {"name": "sys_file_name", "type": "String", "searchable": true, "filterable": true, "facetable": true},
    {"name": "sys_file_path", "type": "String", "filterable": true},
    {"name": "sys_file_size_bytes", "type": "Int64", "filterable": true, "sortable": true},
    {"name": "sys_file_type", "type": "String", "filterable": true, "facetable": true},
    {"name": "sys_last_updated", "type": "DateTimeOffset", "filterable": true, "sortable": true},
    {"name": "sys_page_count", "type": "Int32", "filterable": true, "sortable": true},
    {"name": "sys_extracted_at", "type": "DateTimeOffset", "filterable": true, "sortable": true},
    {"name": "sys_processing_version", "type": "String", "filterable": true, "facetable": true},
    
    // FILE INDEX METADATA
    {"name": "file_name", "type": "String", "searchable": true, "filterable": true, "sortable": true},
    {"name": "library_name_en", "type": "String", "searchable": true, "filterable": true, "facetable": true},
    {"name": "library_name_tc", "type": "String", "searchable": true, "filterable": true, "facetable": true},
    {"name": "category_name_en", "type": "String", "searchable": true, "filterable": true, "facetable": true},
    {"name": "category_name_tc", "type": "String", "searchable": true, "filterable": true, "facetable": true},
    {"name": "title_name_en", "type": "String", "searchable": true, "sortable": true},
    {"name": "title_name_tc", "type": "String", "searchable": true, "sortable": true},
    {"name": "file_url", "type": "String", "retrievable": true},
    {"name": "branch_name", "type": "String", "filterable": true, "facetable": true},
    {"name": "item_type", "type": "String", "filterable": true, "facetable": true},
    {"name": "item_url", "type": "String", "retrievable": true},
    
    // CONTENT FIELDS
    {"name": "chunk_content", "type": "String", "searchable": true},
    {"name": "chunk_content_vector", "type": "Collection(Single)", "dimensions": 1536, "searchable": true},
    
    // CHUNK METADATA
    {"name": "chunk_page_number", "type": "Int32", "filterable": true, "sortable": true},
    {"name": "chunk_function_summary", "type": "String", "searchable": true},
    {"name": "chunk_char_count", "type": "Int32", "filterable": true},
    {"name": "chunk_word_count", "type": "Int32", "filterable": true},
    {"name": "chunk_sentence_count", "type": "Int32", "filterable": true},
    {"name": "chunk_entities", "type": "Collection(String)", "searchable": true, "filterable": true, "facetable": true},
    
    // QA PAIRS
    {"name": "qa_questions", "type": "Collection(String)", "searchable": true},
    {"name": "qa_answers", "type": "Collection(String)", "searchable": true},
    {"name": "qa_confidence", "type": "Double", "filterable": true, "sortable": true}
  ]
}
```

---

## 🎯 10 SEARCH STRATEGIES 

---

### **STRATEGY 1: Hybrid Search (Text + Vector)**
**When to use:** Default for 80% of queries
**Best for:** General searches, exploratory queries

```python
def hybrid_search(query: str, branch: str = None, category: str = None, top: int = 10):
    """
    Combines keyword search (BM25) + vector similarity (cosine)
    This should be your DEFAULT search strategy
    """
    # Generate embedding for query
    embedding = get_embedding(query)
    
    # Build filters
    filters = []
    if branch:
        filters.append(f"branch_name eq '{branch}'")
    if category:
        filters.append(f"category_name_en eq '{category}'")
    
    # Avoid sparse pages
    filters.append("chunk_word_count ge 50")
    
    filter_str = " and ".join(filters) if filters else None
    
    # Execute hybrid search
    results = search_client.search(
        search_text=query,  # Keyword search on chunk_content
        vector_queries=[{
            "vector": embedding,
            "k_nearest_neighbors": 50,
            "fields": "chunk_content_vector"
        }],
        filter=filter_str,
        select="doc_id,file_name,chunk_content,chunk_page_number,chunk_function_summary,file_url",
        top=top
    )
    
    return list(results)
```

**Example queries:**
- "medical insurance coverage"
- "outpatient services Hong Kong"
- "claim process"

---

### **STRATEGY 2: QA Pair Search (Direct Answers)**
**When to use:** Question-type queries
**Best for:** Instant answers without LLM generation

```python
def qa_search(query: str, branch: str = None, min_confidence: float = 0.7):
    """
    Search pre-generated QA pairs for instant answers
    Try this FIRST for question queries, fallback to hybrid if no match
    """
    filters = [f"qa_confidence ge {min_confidence}"]
    if branch:
        filters.append(f"branch_name eq '{branch}'")
    
    filter_str = " and ".join(filters)
    
    # Search ONLY in qa_questions field
    results = search_client.search(
        search_text=query,
        search_fields="qa_questions",
        filter=filter_str,
        select="qa_questions,qa_answers,file_name,chunk_page_number,qa_confidence",
        top=3
    )
    
    results_list = list(results)
    
    # Check if we got a good match
    if results_list and results_list[0]['@search.score'] > 0.6:
        best = results_list[0]
        return {
            "type": "direct_answer",
            "question": best['qa_questions'][0],
            "answer": best['qa_answers'][0],
            "source": best['file_name'],
            "page": best['chunk_page_number'],
            "confidence": best['qa_confidence']
        }
    
    return None  # No good match, fallback to hybrid


def qa_first_search(query: str, branch: str = None):
    """
    Smart wrapper: Try QA first, fallback to hybrid
    """
    # Try QA
    qa_result = qa_search(query, branch)
    if qa_result:
        return qa_result
    
    # Fallback to hybrid
    return {
        "type": "hybrid_search",
        "results": hybrid_search(query, branch)
    }
```

**Example queries:**
- "What is the coverage limit?"
- "How do I file a claim?"
- "When does the policy expire?"

---

### **STRATEGY 3: Filter by Branch**
**When to use:** Location-specific queries
**Best for:** HK vs MACAU separation

```python
def search_by_branch(query: str, branch: str):
    """
    Filter results to specific branch only
    branch: "HK" or "MACAU"
    """
    results = search_client.search(
        search_text=query,
        filter=f"branch_name eq '{branch}'",
        select="doc_id,file_name,chunk_content,chunk_page_number",
        top=20
    )
    
    return list(results)


# Example usage
hk_results = search_by_branch("medical insurance", branch="HK")
macau_results = search_by_branch("medical insurance", branch="MACAU")
```

**Example queries:**
- "Hong Kong medical insurance" → Filter: branch_name eq 'HK'
- "MACAU life insurance" → Filter: branch_name eq 'MACAU'

---

### **STRATEGY 4: Filter by Category**
**When to use:** Product-specific searches
**Best for:** Narrowing by insurance type

```python
def search_by_category(query: str, category_en: str = None, category_tc: str = None):
    """
    Filter by insurance category (English or Traditional Chinese)
    Examples: "Medical Insurance", "Life Insurance", "醫療保險"
    """
    if category_en:
        filter_str = f"category_name_en eq '{category_en}'"
    elif category_tc:
        filter_str = f"category_name_tc eq '{category_tc}'"
    else:
        filter_str = None
    
    results = search_client.search(
        search_text=query,
        filter=filter_str,
        select="file_name,category_name_en,chunk_page_number,chunk_function_summary",
        top=20
    )
    
    return list(results)


# Example usage
medical = search_by_category("coverage", category_en="Medical Insurance")
life = search_by_category("coverage", category_en="Life Insurance")
```

**Example queries:**
- "Medical insurance coverage" → Filter: category_name_en eq 'Medical Insurance'
- "醫療保險" → Filter: category_name_tc eq '醫療保險'

---

### **STRATEGY 5: Filter by Library**
**When to use:** Document collection filtering
**Best for:** Searching within specific libraries

```python
def search_by_library(query: str, library_en: str = None, library_tc: str = None):
    """
    Filter by document library
    Examples: "Product Brochures", "Policy Documents", "產品手冊"
    """
    if library_en:
        filter_str = f"library_name_en eq '{library_en}'"
    elif library_tc:
        filter_str = f"library_name_tc eq '{library_tc}'"
    else:
        filter_str = None
    
    results = search_client.search(
        search_text=query,
        filter=filter_str,
        select="file_name,library_name_en,chunk_page_number,chunk_function_summary",
        top=20
    )
    
    return list(results)


# Example usage
brochures = search_by_library("coverage", library_en="Product Brochures")
policies = search_by_library("terms", library_en="Policy Documents")
```

---

### **STRATEGY 6: Combined Filters**
**When to use:** Precise targeting
**Best for:** Multi-dimensional filtering

```python
def advanced_filter_search(
    query: str,
    branch: str = None,
    category: str = None,
    library: str = None,
    min_word_count: int = 50
):
    """
    Combine multiple filters for precise results
    """
    filters = []
    
    if branch:
        filters.append(f"branch_name eq '{branch}'")
    
    if category:
        filters.append(f"category_name_en eq '{category}'")
    
    if library:
        filters.append(f"library_name_en eq '{library}'")
    
    if min_word_count:
        filters.append(f"chunk_word_count ge {min_word_count}")
    
    filter_str = " and ".join(filters) if filters else None
    
    results = search_client.search(
        search_text=query,
        filter=filter_str,
        select="file_name,chunk_page_number,chunk_function_summary",
        top=20
    )
    
    return list(results)


# Example: HK Medical Insurance in Product Brochures only
results = advanced_filter_search(
    query="outpatient coverage",
    branch="HK",
    category="Medical Insurance",
    library="Product Brochures"
)
```

---

### **STRATEGY 7: Search Specific Document**
**When to use:** User mentions document name
**Best for:** "Find X in Y document"

```python
def search_in_document(query: str, filename: str):
    """
    Search within specific document by filename
    Results sorted by page number
    """
    results = search_client.search(
        search_text=query,
        filter=f"file_name eq '{filename}'",
        orderby="chunk_page_number asc",
        select="chunk_page_number,chunk_content,chunk_function_summary,file_url",
        top=100  # Get all pages from document
    )
    
    return list(results)


# Example usage
results = search_in_document(
    query="deductible",
    filename="Medical Insurance Brochure 2024.pdf"
)

# Show results by page
for r in results:
    print(f"Page {r['chunk_page_number']}: {r['chunk_function_summary']}")
```

**Example queries:**
- "Find claim process in 2024 brochure"
- "Search deductible in policy document"

---

### **STRATEGY 8: Entity-Based Search**
**When to use:** Finding specific terms/amounts
**Best for:** Domain-specific entities

```python
def entity_search(entity: str, query: str = "", branch: str = None):
    """
    Search by extracted entities (insurance terms, amounts, products)
    Uses chunk_entities field
    """
    filters = [f"chunk_entities/any(e: e eq '{entity}')"]
    
    if branch:
        filters.append(f"branch_name eq '{branch}'")
    
    filter_str = " and ".join(filters)
    
    results = search_client.search(
        search_text=query,
        filter=filter_str,
        select="file_name,chunk_page_number,chunk_entities,chunk_function_summary",
        facets=["file_name,count:10"],  # Show which docs have this entity
        top=20
    )
    
    return list(results)


# Example usage
# Find all pages mentioning specific amount
results = entity_search("HK$50,000")

# Find pages with specific insurance product
results = entity_search("medical insurance", query="coverage")

# Find specific term in specific branch
results = entity_search("outpatient services", branch="HK")
```

**Example queries:**
- "Show all pages mentioning HK$50,000"
- "Find critical illness coverage"
- "Where is deductible mentioned?"

---

### **STRATEGY 9: Faceted Search (Browse + Filter)**
**When to use:** Building filter UI
**Best for:** Category counts, sidebar filters

```python
def faceted_search(query: str = ""):
    """
    Get search results + facet counts for UI filters
    Returns both results and category counts
    """
    results = search_client.search(
        search_text=query or "*",
        facets=[
            "category_name_en,count:20",      # Top 20 categories
            "library_name_en,count:10",       # Top 10 libraries
            "branch_name",                     # All branches (HK, MACAU)
            "item_type,count:10",             # Document types
            "chunk_entities,count:50",        # Top 50 mentioned entities
            "sys_file_type"                   # File types (pdf, docx)
        ],
        select="file_name,chunk_page_number,chunk_function_summary",
        top=20
    )
    
    return {
        "results": list(results),
        "facets": results.get_facets()
    }


# Example usage
data = faceted_search("insurance")

# Display facets for UI
print("Filter by Category:")
for cat in data["facets"]["category_name_en"]:
    print(f"  ☐ {cat['value']} ({cat['count']} pages)")

print("\nFilter by Branch:")
for branch in data["facets"]["branch_name"]:
    print(f"  ☐ {branch['value']} ({branch['count']} pages)")

print("\nFilter by Library:")
for lib in data["facets"]["library_name_en"]:
    print(f"  ☐ {lib['value']} ({lib['count']} pages)")
```

**UI Example:**
```
Filter by Category:
☐ Medical Insurance (245 pages)
☐ Life Insurance (156 pages)
☐ Travel Insurance (89 pages)

Filter by Branch:
☐ HK (320 pages)
☐ MACAU (180 pages)

Filter by Library:
☐ Product Brochures (200 pages)
☐ Policy Documents (150 pages)
☐ Application Forms (150 pages)
```

---

### **STRATEGY 10: Summary Search (Quick Overview)**
**When to use:** High-level information
**Best for:** Fast scanning without full content

```python
def summary_search(query: str, branch: str = None):
    """
    Search ONLY in summaries for quick overview
    Searches chunk_function_summary field only
    """
    filters = []
    if branch:
        filters.append(f"branch_name eq '{branch}'")
    
    filter_str = " and ".join(filters) if filters else None
    
    results = search_client.search(
        search_text=query,
        search_fields="chunk_function_summary",  # Only search summaries
        filter=filter_str,
        select="file_name,chunk_page_number,chunk_function_summary",
        top=20
    )
    
    return list(results)


# Example usage
summaries = summary_search("coverage limits", branch="HK")

for s in summaries:
    print(f"{s['file_name']} - Page {s['chunk_page_number']}")
    print(f"  {s['chunk_function_summary']}\n")
```

**Example queries:**
- "Give me overview of coverage"
- "Quick summary of benefits"
- "What's the main point about claims?"

---

## 🎯 BONUS STRATEGIES

### **STRATEGY 11: Date Range Filter**
```python
def search_recent_documents(query: str, days_ago: int = 30):
    """Find recently updated documents"""
    from datetime import datetime, timedelta
    
    cutoff = (datetime.utcnow() - timedelta(days=days_ago)).isoformat() + 'Z'
    
    results = search_client.search(
        search_text=query,
        filter=f"sys_last_updated ge {cutoff}",
        orderby="sys_last_updated desc",
        select="file_name,sys_last_updated,chunk_function_summary",
        top=20
    )
    
    return list(results)
```

### **STRATEGY 12: Page Range Filter**
```python
def search_page_range(query: str, filename: str, start_page: int, end_page: int):
    """Search specific page range in document"""
    results = search_client.search(
        search_text=query,
        filter=(
            f"file_name eq '{filename}' and "
            f"chunk_page_number ge {start_page} and "
            f"chunk_page_number le {end_page}"
        ),
        orderby="chunk_page_number asc",
        top=100
    )
    
    return list(results)
```

### **STRATEGY 13: Semantic Search (AI-Powered)**
```python
def semantic_search(query: str, branch: str = None):
    """
    Use AI semantic ranking for better relevance
    Requires semantic configuration in index
    """
    filter_str = f"branch_name eq '{branch}'" if branch else None
    
    results = search_client.search(
        search_text=query,
        filter=filter_str,
        query_type="semantic",
        semantic_configuration_name="insurance-semantic-config",
        top=10
    )
    
    return list(results)
```

---

## 🤖 SMART QUERY ROUTER

```python
def route_query(query: str, user_context: dict = None):
    """
    Intelligent router that picks the best search strategy
    
    Args:
        query: User's search query
        user_context: Optional context (branch, category, etc.)
    
    Returns:
        Search results from optimal strategy
    """
    # Step 1: Analyze query intent (simple rule-based or LLM)
    query_lower = query.lower()
    
    # Route based on query patterns
    
    # Priority 1: Question queries → Try QA first
    if any(q in query_lower for q in ['what', 'how', 'when', 'where', 'why', '?']):
        result = qa_first_search(query, user_context.get('branch'))
        if result.get('type') == 'direct_answer':
            return result
    
    # Priority 2: Document-specific queries
    if 'in' in query_lower and ('brochure' in query_lower or 'document' in query_lower or 'policy' in query_lower):
        # Extract document name (simplified)
        # In production, use NER or LLM to extract filename
        pass
    
    # Priority 3: Entity queries
    if 'all pages' in query_lower or 'mentions' in query_lower or 'hk$' in query_lower:
        # Extract entity and do entity search
        pass
    
    # Priority 4: Summary queries
    if any(word in query_lower for word in ['overview', 'summary', 'briefly', 'quick']):
        return {
            'type': 'summary_search',
            'results': summary_search(query, user_context.get('branch'))
        }
    
    # Default: Hybrid search with context filters
    return {
        'type': 'hybrid_search',
        'results': hybrid_search(
            query,
            branch=user_context.get('branch'),
            category=user_context.get('category')
        )
    }


# Example usage
result = route_query(
    query="What is the coverage limit?",
    user_context={"branch": "HK"}
)

if result['type'] == 'direct_answer':
    print(f"✅ {result['answer']}")
else:
    print(f"📊 Found {len(result['results'])} results")
```

---

## 📊 STRATEGY COMPARISON TABLE

| Strategy | Use Case | Complexity | Speed | Accuracy |
|----------|----------|------------|-------|----------|
| **1. Hybrid** | General queries | Medium | Fast | High |
| **2. QA Pairs** | Direct questions | Medium | Very Fast | Very High |
| **3. Branch Filter** | Location-specific | Low | Very Fast | High |
| **4. Category Filter** | Product-specific | Low | Very Fast | High |
| **5. Library Filter** | Collection-specific | Low | Very Fast | High |
| **6. Combined Filters** | Precise targeting | Medium | Fast | Very High |
| **7. Document Search** | Single doc queries | Low | Fast | High |
| **8. Entity Search** | Term/amount search | Medium | Fast | Very High |
| **9. Faceted Search** | Browse/explore | Medium | Fast | N/A |
| **10. Summary Search** | Quick overview | Low | Very Fast | Medium |

---

