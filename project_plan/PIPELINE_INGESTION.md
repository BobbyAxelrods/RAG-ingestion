# Final Workflow Summary - Document Ingestion Pipeline

## 🎯 What We Are Building

A **simplified document ingestion pipeline** for Prudential Hong Kong that processes documents from Azure Blob Storage and indexes them into Azure AI Search with a **specific custom schema** that includes **CSV-based metadata enrichment**.

---

## 🏗️ Complete System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRUDENTIAL HK RAG ECOSYSTEM                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  PART 1: INGESTION PIPELINE (What We're Building Now)                    │
└──────────────────────────────────────────────────────────────────────────┘

   [Azure Blob Storage]
         │
         │ (Manual/Scheduled Trigger)
         ▼
   ┌────────────────┐
   │  Main Pipeline │
   │   Orchestrator │
   └────────┬───────┘
            │
            ├──> Load metadata.csv into memory
            │
            ▼
   ┌─────────────────────────────┐
   │  File Type Detection        │
   │  (.pdf, .docx, .txt, etc.)  │
   └──────────┬──────────────────┘
              │
              ├──────────────┬───────────────┐
              ▼              ▼               ▼
      ┌──────────────┐ ┌──────────┐ ┌──────────────┐
      │ .pdf, .docx  │ │Standalone│ │ .txt, .md    │
      │   .pptx      │ │  Images  │ │ .csv, .json  │
      └──────┬───────┘ │.png, jpg │ └──────┬───────┘
             │         └────┬─────┘        │
             ▼              │              │
      ┌──────────────────────────────┐    │
      │ Azure Document Intelligence  │    │
      │ - prebuilt-layout model      │    │
      │ - Extract tables, sections   │    │
      │ - Extract embedded images    │◄───┘
      │ - Convert to Markdown        │
      └──────────┬───────────────────┘
                 │
                 ▼
      ┌──────────────────────────────┐
      │ Image Extraction & Processing│
      │ - Identify embedded images   │
      │ - Extract image files        │
      │ - Apply OCR per image        │
      │ - Generate image summaries   │
      │ - Enrich markdown with       │
      │   image descriptions         │
      └──────────┬───────────────────┘
                 │
                 ▼
      ┌──────────────────────────────┐
      │ Enhanced Content Assembly    │
      │ - Merge text + image context │
      │ - Preserve document structure│
      │ - Link images to sections    │
      └──────────┬───────────────────┘
                 │
                 ▼
      ┌──────────────────────────────┐
      │ CSV Metadata Lookup (EARLY) │
      │ - Match filename             │
      │ - Enrich file_metadata       │
      │ - library, category          │
      │ - branch (HK/MACAU)          │
      │ - language (en/zh-HK)        │
      └──────────┬───────────────────┘
                 │
                 ▼
      ┌─────────────────────────┐
      │ File Summary Gen        │
      │ (Extractive/LLM)        │
      │ file_function_summary   │
      └──────────┬──────────────┘
                 │
                 ▼
      ┌─────────────────────────────────┐
      │ Page-Level Enrichment (CRITICAL)│
      │ Process EACH page with FULL     │
      │ context BEFORE chunking:        │
      │                                 │
      │ For EACH page:                  │
      │ 1. Page Summary Gen             │
      │    - page_function_summary      │
      │                                 │
      │ 2. Keyword Extraction           │
      │    - Entities, products         │
      │    - Topics, key phrases        │
      │    - File type                  │
      │                                 │
      │ 3. Synthetic Q&A Gen            │
      │    - Generate questions         │
      │    - Generate answers           │
      │    - Confidence scoring         │
      └──────────┬──────────────────────┘
                 │
                 ▼
      ┌─────────────────────┐
      │ Chunking Service    │
      │ - DocAnalysisChunker│
      │ - LangChainChunker  │
      │ - 800-1200 tokens   │
      │ - 100-200 overlap   │
      │ - Image context inc.│
      │ - Page boundaries   │
      └──────────┬──────────┘
                            │
                            ▼
                 ┌─────────────────────────┐
                 │ Chunk Quality (NEW)     │
                 │ - Semantic density      │
                 │ - Quality scoring       │
                 └──────────┬──────────────┘
                            │
                            ▼
                 ┌─────────────────────────┐
                 │ Azure OpenAI Embeddings │
                 │ text-embedding-3-small  │
                 │ (1536 dims)             │
                 │ - file_summary → vector │
                 │ - page_summary → vector │
                 │   (for EACH page)       │
                 │ - chunk_text → vector   │
                 │   (for EACH chunk)      │
                 └──────────┬──────────────┘
                            │
                            ▼
                 ┌─────────────────────────┐
                 │ Azure AI Search         │
                 │ Index Creation/Update   │
                 │ (Hierarchical Schema)   │
                 │ doc → pages → chunks    │
                 └──────────┬──────────────┘
                            │
                            ▼
                         [DONE]

┌──────────────────────────────────────────────────────────────────────────┐
│  PART 2: AGENTIC RAG RETRIEVAL (Already Exists)                          │
└──────────────────────────────────────────────────────────────────────────┘

   [User Query]
         │
         ▼
   ┌─────────────────┐
   │ LangGraph       │
   │ Workflow        │
   │ (multiagent)    │
   └────────┬────────┘
            │
            ├──> node_detect_state (LLM: analyze query)
            ├──> node_web_clarify (Web search if needed)
            ├──> node_assess (Extract keywords)
            ├──> node_pre_bm25 + node_pre_vector (Pre-search on CSV index)
            ├──> node_main_original (Search THIS INDEX we're building)
            ├──> node_generate_response (LLM: RAG response)
            └──> node_evaluate (Quality check)
                 │
                 ▼
            [Response to User]
```

---

## 🔄 Codebase Workflow (Aligned to Diagram)

- Orchestrator: `src/etl/main.py` with command `process`
- Early CSV metadata load: `src/etl/processors/services/metadata_enrichment_service.py`
  - Loads CSV/Excel into a pandas `DataFrame`
  - Precomputes normalized columns for fast matching (vectorized masks)
  - Matches filename/title/url tail; fuzzy fallback if needed
- File type detection: processor logic using `DocumentMetadata`
- Document extraction: `src/services/doc_intel_service.py`
  - Prebuilt layout → markdown conversion, table/image detection
- Image processing: OCR via Document Intelligence `prebuilt-read`
- Enhanced content assembly: processors merge text + image context
- CSV Metadata Lookup (EARLY): enriches `file_metadata` before summaries/chunking
- File summary generation: `src/services/summary_generation_service.py`
  - Produces `file_function_summary` and embedding vector [1536]
- Page-level enrichment (critical):
  - Page summary gen → per-page vectors [1536]
  - Keyword extraction → entities, products, topics, key phrases
  - Synthetic Q&A → per-page or file-level training pairs
- Chunking service:
  - `DocAnalysisChunker` for structured docs (.pdf/.docx/.pptx)
  - `LangChainChunker` for plain text (.txt/.md/.json/.csv/.html)
  - Size 800–1200 tokens, overlap 100–200, preserves tables/lists
- Chunk quality scoring (planned): semantic density scoring
- Embeddings: `src/services/openai_service.py` (text-embedding-3-small, 1536 dims)
- Indexing: search services (upload disabled with `--no-index`)

### Test Workflow Command (Local, No Index)

- Run ETL for a single file and write JSON output:
  - `python -m src.etl.main process --file "artifact\matched_downloads\Cashier\PHKLMB Payment_Methods (Customer verion) Nov 2024.pdf" --output-json "artifact\etl_output\PHKLMB Payment_Methods (Customer verion) Nov 2024.json" --local --no-index`
- Expected outputs:
  - JSON at `artifact/etl_output/...` with hierarchical structure (doc → pages → chunks)
  - `keyword_extraction`, `synthetic_qa_pairs`, and page-level fields present
  - Logs in `logs/pipeline.log` indicating metadata load/match and processing stats

### Notes on CSV/Excel Metadata

- The metadata service is pandas-first with vectorized matching for speed and robustness.
- Matching strategy order:
  - Exact filename match → normalized equality → compact equality
  - Title equality/contains → file_name/file_description compact contains
  - URL tail contains → fuzzy fallback (SequenceMatcher + token Jaccard)
- If the metadata file is unreadable, the pipeline falls back to defaults and continues.

## 📋 Critical Requirements Summary

### 1. Custom Azure AI Search Schema (MANDATORY) - UPDATED TO MATCH etl_schema.json

We MUST implement this **hierarchical schema** with doc → pages → chunks structure:

#### Document Level Fields
| Field Name | Type | Dimensions | Purpose |
|------------|------|------------|---------|
| **doc_id** | String (Key) | - | Unique document ID (UUID or hash) |
| **filename** | String | - | Source document filename |
| **file_metadata** | JSON Object | - | Complete file metadata (see detailed structure below) |
| **file_summary** | JSON Object | - | `{"file_function_summary": "string", "file_summary_vector": [1536]}` |
| **keyword_extraction** | JSON Object | - | `{"entities": [], "product_names": [], "topics": [], "file_type": "", "key_phrases": []}` |
| **pages** | JSON Array | - | Array of page objects (see page structure below) |
| **synthetic_qa_pairs** | JSON Array | - | Generated Q&A pairs for training/evaluation |
| **tables** | JSON Array | - | Extracted tables with HTML/MD/CSV representations |
| **images** | JSON Array | - | Extracted images with descriptions and metadata |
| **processing_metadata** | JSON Object | - | Processing statistics, errors, timing |

#### File Metadata Structure
### Desired Extraction Result Schema 

{
  "file_metadata": 
}

```json
{
  "file_name": "string",
  "file_path": "string (original path)",
  "file_size_bytes": "integer",
  "file_type": "string (pdf/jpg/docx/xlsx)",
  "library_name": "string",
  "category_name": "string",
  "title_name": "string",
  "file_url": "string (blob storage URL)",
  "branch_name": "string (HK/MACAU)",
  "document_language": "string (en/zh-HK/zh-CN)",
  "last_updated": "datetime (ISO 8601)",
  "page_count": "integer",
  "extracted_at": "datetime (ISO 8601)",
  "processing_version": "string (e.g., v1.2.3)"
}
```

#### Page Structure (within pages array)
```json
{
  "page_number": "integer",
  "page_summary": {
    "page_function_summary": "string (LLM-generated page summary)",
    "page_summary_vector": "array<float>[1536]"
  },
  "page_metadata": {
    "page_width": "float",
    "page_height": "float",
    "page_orientation": "string (portrait/landscape)",
    "has_images": "boolean",
    "has_tables": "boolean",
    "image_count": "integer",
    "table_count": "integer"
  },
  "chunks": [/* array of chunk objects */]
}
```

#### Chunk Structure (within page.chunks array)
```json
{
  "chunk_id": "string (doc_id + page + position)",
  "chunk_text": "string (actual content)",
  "chunk_position": "integer (order within page)",
  "chunk_type": "string (text/table/image_caption)",
  "chunk_semantic_density": "float (0.0-1.0, content quality score)",
  "chunk_metadata": {
    "char_count": "integer",
    "word_count": "integer",
    "sentence_count": "integer",
    "bbox": {"x": "float", "y": "float", "width": "float", "height": "float"}
  },
  "content_chunk_vector": "array<float>[1536]"
}
```

### Implementation Status (Nov 2025)

- Document Intelligence extraction: Implemented (`src/services/doc_intel_service.py`)
- Image processing orchestration: Implemented; image byte extraction is placeholder
  - OCR: Implemented via Document Intelligence `prebuilt-read`
  - GPT-4 Vision descriptions: Implemented via Azure OpenAI
  - Markdown enrichment: Implemented (appends image blocks; insertion position TBD)
- Summary generation (extractive): Implemented (`src/services/summary_generation_service.py`)
- Chunking: Implemented (DocAnalysis + LangChain)
- CSV metadata enrichment: Implemented (`src/services/metadata_enrichment_service.py`)
- Embeddings: Implemented (`src/services/openai_service.py`)
- Azure AI Search indexing: Implemented (`src/services/search_service.py`)

### Known Limitations / Next Steps

- Implement actual embedded image byte extraction for PDFs/Office (`image_processing_service._extract_image_bytes`)
- Insert image markdown blocks at precise locations per page/section (currently appended)
- Add tests for image handling and chunk-image context alignment
- Optional: Support LLM-based summary method as configuration alternative

**Key Points**:
- Vector dimensions: **1536** (NOT 1564) - using text-embedding-3-small
- **Triple-level vectors**:
  - **File-level**: `file_summary.file_summary_vector` [1536]
  - **Page-level**: `page_summary.page_summary_vector` [1536] for each page
  - **Chunk-level**: `content_chunk_vector` [1536] for each chunk
- **Hierarchical structure**: Document → Pages → Chunks (NOT flat chunks)
- **New required fields**:
  - `keyword_extraction`: Entities, product names, topics, key phrases
  - `synthetic_qa_pairs`: Generated Q&A for evaluation/training
  - `chunk_semantic_density`: Quality score for each chunk (0.0-1.0)
  - `processing_metadata`: OCR confidence, timing, errors/warnings
- CSV enrichment: Populate `file_metadata` fields from CSV lookup

### 2. CSV Metadata Enrichment (MANDATORY) - UPDATED

**CSV Structure** (now includes branch, language, and more metadata):
```csv
filename,doc_id,library_name,category_name,title_name,branch_name,document_language,file_url,last_updated
document1.pdf,DOC001,TechnicalLibrary,Architecture,System Design Guide,HK,en,https://example.com/doc1.pdf,2025-01-15T10:30:00Z
保單指南.pdf,DOC002,保險產品,保單條款,保單指南,MACAU,zh-HK,https://example.com/doc2.pdf,2025-01-16T14:20:00Z
```

**Process** (UPDATED for hierarchical structure):
1. Load CSV at startup (pandas DataFrame)
2. For each document, look up filename in CSV
3. Populate `file_metadata` object with:
   - `file_name`, `file_path`, `file_size_bytes`, `file_type`
   - `library_name`, `category_name`, `title_name`
   - `file_url`, `branch_name` (HK/MACAU)
   - `document_language` (en/zh-HK/zh-CN)
   - `last_updated`, `page_count`, `extracted_at`, `processing_version`
4. **NEW**: Generate keyword extraction from content
5. **NEW**: Generate synthetic Q&A pairs for training
6. Handle missing entries gracefully (use fallback values)

### 3. Chunking Strategy (File-Type Based)

#### DocAnalysisChunker
**File types**: `.pdf`, `.docx`, `.pptx`, `.bmp`, `.png`, `.jpeg`, `.jpg`, `.tiff`

**Process**:
1. Send to Azure Document Intelligence API (prebuilt-layout model)
2. Extract structured elements (tables, sections, paragraphs)
3. Convert to Markdown format
4. Apply LangChain RecursiveCharacterTextSplitter
5. For images: OCR first, then chunk

#### LangChainChunker
**File types**: `.txt`, `.md`, `.json`, `.csv`, `.html`

**Process**:
1. Read file content directly (no API call)
2. Apply LangChain RecursiveCharacterTextSplitter
3. Split on natural boundaries (paragraphs, sections)

**Parameters (Both Chunkers)**:
- Chunk size: 800-1200 tokens
- Chunk overlap: 100-200 tokens
- Preserve: Tables, code blocks, lists (no mid-split)

### 4. File Summary Generation (Implemented)

**Purpose**: Create a concise summary for the entire document.

**Approach**: Extractive (implemented)
- Document purpose extraction via title/first paragraph/patterns
- Page-by-page key sentence extraction using importance scoring
- Structured combination with max length enforcement

**Output**: Stored in `file_summary` and embedded in `file_summary_chunk` (1536 dims)

See `src/services/summary_generation_service.py` and `project_definition/FILE_SUMMARY_STRATEGY.md` for details.

---

## 🔧 Technical Stack

### Programming Language
- **Python 3.11+** (already set: 3.13 in pyproject.toml)

### Core Dependencies
```
azure-storage-blob
azure-ai-formrecognizer (Document Intelligence)
azure-search-documents
openai (Azure OpenAI)
langchain
langchain-text-splitters
python-dotenv
pandas
```

### Project Structure (Target)
```
indexing_pipeline/
├── src/
│   ├── chunkers/
│   │   ├── __init__.py
│   │   ├── base_chunker.py
│   │   ├── doc_analysis_chunker.py
│   │   └── langchain_chunker.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── document_processor.py
│   │   └── enrichment_processor.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── blob_service.py
│   │   ├── doc_intel_service.py
│   │   ├── search_service.py
│   │   ├── openai_service.py
│   │   ├── image_processing_service.py  # Image extraction & enrichment
│   │   ├── summary_generation_service.py  # File & page summary generation
│   │   ├── metadata_enrichment_service.py
│   │   ├── keyword_extraction_service.py  # NEW: Entity/product/topic extraction
│   │   ├── qa_generation_service.py  # NEW: Synthetic Q&A generation
│   │   ├── chunk_quality_service.py  # NEW: Semantic density scoring
│   │   └── table_extraction_service.py  # NEW: Table HTML/MD/CSV generation
│   ├── models/
│   │   ├── __init__.py
│   │   └── document_models.py
│   ├── config.py
│   └── main.py
├── metadata/
│   └── document_metadata.csv
├── tests/
├── .env
├── .env.example
├── pyproject.toml
└── README.md
```

---

## 🎯 Development Phases (Recommended Order)

### Phase 1: Project Setup & Configuration
- ✅ Already done: Python 3.13 venv, pyproject.toml
- [ ] Add dependencies to pyproject.toml
- [ ] Create .env.example template
- [ ] Set up project structure (src/, metadata/, tests/)
- [ ] Create config.py for centralized configuration

### Phase 2: Core Services (Foundation)
- [ ] Implement `blob_service.py` (list/download from Blob Storage)
- [ ] Implement `doc_intel_service.py` (Document Intelligence API client)
- [ ] Implement `openai_service.py` (embedding generation + GPT-4 Vision)
- [ ] Implement `search_service.py` (Azure AI Search index creation/upload)

### Phase 3: CSV Metadata Service
- [ ] Implement `metadata_enrichment_service.py`
- [ ] Load CSV at startup (with bilingual fields)
- [ ] Lookup logic (filename → metadata)
- [ ] Fallback handling for missing entries
- [ ] Create sample CSV template (with EN/TC fields)

### Phase 3.5: Image Processing Service (NEW)
- [ ] Implement `image_processing_service.py`
- [ ] Extract embedded images from Document Intelligence results
- [ ] Apply OCR to extracted images
- [ ] Generate image summaries (GPT-4 Vision integration)
- [ ] Enrich markdown with image descriptions
- [ ] Create image metadata for chunks

### Phase 4: Summary Generation (File + Page) - BEFORE CHUNKING ✅ APPROACH DEFINED
- [ ] Implement `summary_generation_service.py`
- [ ] **File-level summary**:
  - [ ] Document purpose extraction (regex patterns + fallbacks)
  - [ ] Generate file_function_summary
  - [ ] Embed file_function_summary → file_summary_vector [1536]
- [ ] **Page-level summary** (CRITICAL - with full page context):
  - [ ] Extract page_function_summary for EACH page
  - [ ] Embed page_function_summary → page_summary_vector [1536]
  - [ ] Configure extractive vs LLM method (.env)
- [ ] Special case handling:
  - [ ] Very long documents (100+ pages) - adaptive sampling
  - [ ] Bilingual documents (EN/TC detection per page)
  - [ ] Documents with images (include image notes)
  - [ ] Documents with tables (summarize table content)
- [ ] Test with insurance policy documents

### Phase 5: Advanced Page Enrichment Services - BEFORE CHUNKING
- [ ] Implement `keyword_extraction_service.py`
  - [ ] Extract named entities per page (products, people, places)
  - [ ] Identify insurance product names per page
  - [ ] Extract thematic topics per page
  - [ ] Determine content type per page
  - [ ] Extract key phrases per page
- [ ] Implement `qa_generation_service.py`
  - [ ] Generate synthetic questions from full page content
  - [ ] Generate answers with source references to page sections
  - [ ] Calculate confidence scores
  - [ ] Link to doc_id, page_number
- [ ] Implement `table_extraction_service.py`
  - [ ] Extract tables from Document Intelligence results
  - [ ] Convert to HTML representation
  - [ ] Convert to Markdown representation
  - [ ] Convert to CSV representation
  - [ ] Generate table summaries
  - [ ] Extract column headers and metadata

### Phase 6: Chunking Implementation - AFTER ENRICHMENT
- [ ] Implement `base_chunker.py` (abstract base class)
- [ ] Implement `doc_analysis_chunker.py`
  - Document Intelligence API integration
  - **Image extraction integration** (call image service)
  - Markdown conversion logic (with image descriptions)
  - LangChain splitter integration (page-by-page)
  - Preserve page boundaries
- [ ] Implement `langchain_chunker.py`
  - Direct file reading
  - LangChain splitters for text files
- [ ] Implement `chunk_quality_service.py`
  - [ ] Calculate chunk_semantic_density (0.0-1.0)
  - [ ] Score content quality/relevance per chunk
  - [ ] Flag low-quality chunks

### Phase 7: Document Processing Pipeline (UPDATED ORDER)
- [ ] Implement `document_processor.py` with hierarchical structure
- [ ] File type detection and routing
- [ ] **NEW ORDER**: CSV enrichment → file summary → page enrichment → chunking → embeddings → indexing
- [ ] Maintain page-level boundaries during chunking
- [ ] Track processing metadata (OCR confidence, timing, errors)
- [ ] Batch processing logic

### Phase 8: Azure AI Search Integration (UPDATED)
- [ ] Create hierarchical index schema (doc → pages → chunks)
- [ ] Configure vector search for 3 vector types:
  - [ ] file_summary_vector [1536]
  - [ ] page_summary_vector [1536]
  - [ ] content_chunk_vector [1536]
- [ ] Configure HNSW algorithm for all vectors
- [ ] Configure Azure OpenAI vectorizer
- [ ] Implement document upload logic (single hierarchical object)
- [ ] Index creation/update logic

### Phase 9: Main Orchestrator
- [ ] Implement `main.py`
- [ ] End-to-end pipeline orchestration
- [ ] Parallel processing support
- [ ] Progress tracking and logging

### Phase 10: Error Handling & Logging
- [ ] Comprehensive error handling
- [ ] Retry logic (exponential backoff)
- [ ] Detailed logging (structured JSONL?)
- [ ] Failed document tracking
- [ ] Performance metrics

### Phase 11: Testing & Documentation
- [ ] Unit tests for each service
- [ ] Integration tests
- [ ] End-to-end test with sample documents
- [ ] README with setup instructions
- [ ] Example usage documentation

---

## 🔍 Key Differences from Azure GPT-RAG Reference

### What We're KEEPING:
✅ Chunking logic (DocAnalysisChunker, LangChainChunker)
✅ Document Intelligence API usage
✅ Markdown conversion approach
✅ LangChain text splitters
✅ Embedding generation

### What We're REMOVING:
❌ Event Grid (no event-driven triggers)
❌ Service Bus (no queuing)
❌ Function Apps (simple Python script instead)
❌ Cosmos DB (using CSV + index storage)
❌ App Configuration (using .env)

### What We're ADDING:
✅ CSV-based metadata enrichment
✅ Dual vector embeddings (file summary + chunks)
✅ Custom index schema
✅ Simplified orchestration

---

## 🚀 Execution Flow (Step-by-Step)

```
1. START Pipeline
   ├─ Load .env configuration
   ├─ Load metadata.csv into DataFrame
   └─ Initialize Azure clients (Blob, Doc Intel, OpenAI, Search)

2. Check/Create Azure AI Search Index
   ├─ Check if index exists
   ├─ If not, create with exact schema
   └─ Configure vector search profiles

3. List Documents from Blob Storage
   └─ Get list of files to process

4. FOR EACH Document:

   A. Download from Blob Storage

   B. File Type Detection
      └─ Determine extension (.pdf, .txt, etc.)

   C. Document Intelligence Processing (for PDFs/Office docs/standalone images)
      ├─ Call Document Intelligence API (prebuilt-layout)
      ├─ Extract structured content (tables, sections, paragraphs)
      ├─ Extract embedded images from PDF/DOCX
      └─ Convert initial output to Markdown

   D. Image Extraction & Enrichment (NEW STAGE)
      ├─ Identify all images (embedded + standalone)
      │  └─ For PDFs: Extract images at specific page locations
      │
      ├─ For EACH extracted/standalone image:
      │  ├─ Apply OCR using Document Intelligence
      │  ├─ Extract text from image
      │  ├─ Generate image summary/description (LLM or extractive)
      │  └─ Create image metadata (page_number, position, caption)
      │
      └─ Enrich Markdown output:
         ├─ Insert image descriptions inline
         ├─ Add image context to surrounding text
         └─ Link images to their document sections

   E. Enhanced Content Assembly
      ├─ Merge text content + image context
      ├─ Preserve document hierarchy
      └─ Create enriched markdown with image annotations

   F. CSV Metadata Lookup (EARLY - Before enrichment)
      ├─ Find row in metadata_df
      └─ Extract all metadata fields (library, category, title, branch, language, etc.)

   G. Generate File-Level Summary
      ├─ Extract/generate file_function_summary (including image content)
      ├─ Embed file_function_summary → file_summary_vector (1536 dims)
      └─ Create file_summary object

   H. Page-Level Enrichment (CRITICAL - Full page context available)

      FOR EACH Page:

      1. Collect Page Metadata First
         └─ {"page_width", "page_height", "page_orientation",
              "has_images", "has_tables", "image_count", "table_count"}

      2. Generate Page Summary (with FULL page text)
         ├─ Extract/generate page_function_summary
         ├─ Embed page_function_summary → page_summary_vector (1536 dims)
         └─ Create page_summary object

      3. Generate Synthetic Q&A Pairs (with FULL page context)
         ├─ Use LLM to generate questions from entire page content
         ├─ Generate answers referencing page sections
         ├─ Track confidence scores
         └─ Link to doc_id, page_number

      4. Extract Keywords/Entities (with FULL page context)
         ├─ Extract named entities (products, people, places)
         ├─ Identify product names (insurance products)
         ├─ Extract topics (thematic categories)
         ├─ Determine content type for this page
         └─ Extract key phrases

   I. Chunking Service (Page-by-Page - AFTER enrichment)
      ├─ IF originally .pdf/.docx/.pptx → Apply LangChain splitters to enriched markdown PER PAGE
      └─ IF originally .txt/.md/.csv/.json → Apply LangChain splitters directly
      └─ Track page boundaries to maintain page-level structure

      FOR EACH Chunk on each page:

         a. Create chunk metadata
            └─ {"char_count", "word_count", "sentence_count", "bbox"}

         b. Calculate chunk_semantic_density
            └─ Score chunk quality/relevance (0.0-1.0)

         c. Generate chunk embedding
            └─ Embed chunk_text → content_chunk_vector (1536 dims)

         d. Create chunk object (inherits page-level enrichment)
            └─ {chunk_id, chunk_text, chunk_position, chunk_type,
                 chunk_semantic_density, chunk_metadata, content_chunk_vector}

      Create page object with enrichment + chunks:
         └─ {page_number, page_summary, page_metadata, page_qa_pairs,
              page_keywords, chunks: []}

   J. Aggregate Document-Level Data
      ├─ Collect all page-level Q&A pairs into synthetic_qa_pairs[]
      ├─ Collect all page-level keywords into keyword_extraction
      ├─ Extract tables into tables[] array
      └─ Extract images into images[] array

   L. Create Complete Document Object
      └─ {doc_id, filename, file_metadata, file_summary, keyword_extraction,
           pages: [], synthetic_qa_pairs: [], tables: [], images: [], processing_metadata}

   M. Upload to Azure AI Search
      └─ Index complete hierarchical document object

5. Log Results
   ├─ Documents processed
   ├─ Chunks created
   ├─ Errors encountered
   └─ Performance metrics

6. END
```

---

## 🔐 Environment Variables Required

```bash
# Azure Blob Storage
AZURE_STORAGE_CONNECTION_STRING=
STORAGE_CONTAINER_NAME=

# Azure Document Intelligence
DOC_INTELLIGENCE_ENDPOINT=
DOC_INTELLIGENCE_KEY=
DOC_INTELLIGENCE_API_VERSION=2024-07-31-preview

# Azure AI Search (MAIN INDEX - what we're creating)
SEARCH_SERVICE_ENDPOINT=https://phkl-test-weu.search.windows.net
SEARCH_SERVICE_KEY=
SEARCH_INDEX_NAME=reindex-nov-2025

# Azure OpenAI (Embeddings)
AZURE_OPENAI_ENDPOINT=https://agenticpru1.cognitiveservices.azure.com/
AZURE_OPENAI_KEY=
AZURE_OPENAI_DEPLOYMENT_NAME=text-embedding-3-small
AZURE_OPENAI_API_VERSION=2024-12-01-preview
EMBEDDING_DIMENSIONS=1536

# Metadata CSV
METADATA_CSV_PATH=./metadata/document_metadata.csv

# Processing Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_CONCURRENT_PROCESSING=5

# File Summary Generation Settings
GENERATE_FILE_SUMMARY=true
FILE_SUMMARY_METHOD=extractive
SUMMARY_MAX_LENGTH=2000
SUMMARY_SENTENCES_PER_PAGE=2
SUMMARY_INCLUDE_PURPOSE=true
SUMMARY_INCLUDE_PAGE_NUMBERS=true
SUMMARY_SAMPLE_LARGE_DOCS=true
SUMMARY_LARGE_DOC_THRESHOLD=100

# Image Processing Settings (NEW)
EXTRACT_EMBEDDED_IMAGES=true
IMAGE_OCR_ENABLED=true
IMAGE_SUMMARY_METHOD=llm  # or 'extractive' or 'disabled'
IMAGE_SUMMARY_MAX_LENGTH=200
INCLUDE_IMAGE_CONTEXT_IN_CHUNKS=true

# NEW: Page-Level Summary Settings
GENERATE_PAGE_SUMMARIES=true
PAGE_SUMMARY_METHOD=llm  # or 'extractive'
PAGE_SUMMARY_MAX_LENGTH=500

# NEW: Keyword Extraction Settings
ENABLE_KEYWORD_EXTRACTION=true
EXTRACT_ENTITIES=true
EXTRACT_PRODUCT_NAMES=true
EXTRACT_TOPICS=true
EXTRACT_KEY_PHRASES=true
KEYWORD_EXTRACTION_MODEL=gpt-4o-mini

# NEW: Synthetic Q&A Generation Settings
GENERATE_SYNTHETIC_QA=true
QA_PAIRS_PER_DOCUMENT=5
QA_GENERATION_MODEL=gpt-4o-mini
QA_MIN_CONFIDENCE=0.7

# NEW: Chunk Quality Scoring
CALCULATE_CHUNK_SEMANTIC_DENSITY=true
SEMANTIC_DENSITY_MODEL=gpt-4o-mini

# NEW: Processing Metadata
TRACK_PROCESSING_METRICS=true
LOG_OCR_CONFIDENCE=true
LOG_PROCESSING_TIME=true
```

---

## 🖼️ Image Processing Strategy (CRITICAL ADDITION)

### Overview
Images are processed **AFTER** Document Intelligence extraction because PDFs and Office documents contain embedded images that need separate OCR and summarization before chunking.

### Three Types of Image Sources

#### 1. Embedded Images in PDFs
**Source**: Images within PDF pages (charts, diagrams, photos)
**Process**:
- Document Intelligence extracts image locations and bounding boxes
- Extract each image as separate file (base64 or blob)
- Process each image individually

#### 2. Embedded Images in Office Documents
**Source**: Images within .docx, .pptx files
**Process**:
- Document Intelligence API 4.0 can extract these
- Similar to PDF image extraction
- Preserve relationship to surrounding text

#### 3. Standalone Image Files
**Source**: Direct upload of .png, .jpg, .bmp, .tiff files
**Process**:
- Treat entire file as single image
- No parent document structure
- Create standalone document entry

### Image Processing Pipeline (Detailed)

```
FOR EACH Image (embedded or standalone):

1. Image Identification
   ├─ Extract image from parent document (if embedded)
   ├─ Get image metadata (page_number, position, size)
   └─ Store temporary image file

2. OCR Processing
   ├─ Send image to Document Intelligence (Read or Layout API)
   ├─ Extract all text within image
   ├─ Get text with bounding boxes and confidence scores
   └─ Structure extracted text (preserve layout)

3. Image Understanding (LLM-based - RECOMMENDED)
   ├─ Send image to GPT-4 Vision or equivalent
   ├─ Prompt: "Describe this image in detail, including any text, charts, diagrams, or visual elements"
   ├─ Get structured description:
   │  ├─ Image type (chart, diagram, photo, screenshot, etc.)
   │  ├─ Main content/subject
   │  ├─ Key visual elements
   │  └─ Any text/labels visible
   └─ Generate concise summary (max 200 chars)

4. Context Integration
   ├─ Link image to parent document section
   ├─ Identify surrounding text (before/after image)
   ├─ Create context window around image
   └─ Determine semantic relationship

5. Markdown Enrichment
   ├─ Insert image description at image location:
   │  ```markdown
   │  ![Image: Chart showing sales trends]
   │
   │  **Image Description**: A bar chart displaying quarterly sales data
   │  from Q1 2023 to Q4 2023. The chart shows an upward trend with Q4
   │  reaching $2.5M in revenue.
   │
   │  **Extracted Text**: Q1: $1.2M, Q2: $1.5M, Q3: $1.8M, Q4: $2.5M
   │  ```
   │
   ├─ Preserve image reference for retrieval
   └─ Add image metadata to chunk metadata

6. Update Metadata
   └─ Add to metadata_chunk:
      {
        "has_images": true,
        "image_count": 3,
        "image_types": ["chart", "diagram"],
        "image_page_numbers": [5, 7, 12]
      }
```

### Benefits of Post-Document-Intelligence Image Processing

1. **Unified Image Handling**: All images (embedded + standalone) processed the same way
2. **Better Context**: Know where image appears in document structure
3. **Richer Chunks**: Chunks include both text and image descriptions
4. **Improved Retrieval**: Search can match on image content
5. **Multimodal RAG**: Foundation for future vision-based RAG

### Image Processing Service Architecture

```python
# New service: src/services/image_processing_service.py

class ImageProcessingService:
    """
    Handles extraction, OCR, and enrichment of images
    from documents after Document Intelligence processing
    """

    def extract_images_from_doc_intel_result(self, doc_intel_result):
        """Extract embedded images from Document Intelligence output"""
        pass

    def apply_ocr_to_image(self, image_bytes):
        """Apply OCR using Document Intelligence Read API"""
        pass

    def generate_image_summary(self, image_bytes, ocr_text):
        """Generate description using GPT-4 Vision"""
        pass

    def enrich_markdown_with_images(self, markdown, image_data):
        """Insert image descriptions into markdown"""
        pass

    def create_image_metadata(self, images):
        """Create metadata for images in chunk"""
        pass
```

### Updated CSV Structure (with Image Metadata)

```csv
filename,document_id,library_name_en,library_name_tc,category_name_en,category_name_tc,title_name_en,title_name_tc,item_url,has_images,image_count
document1.pdf,DOC001,Technical Library,技術資料庫,Architecture,架構,System Design,系統設計,https://example.com,true,5
```

### Image Storage Considerations

**Option A**: Store image descriptions only (RECOMMENDED for MVP)
- Lightweight, fast indexing
- Images referenced by URL or blob path
- Descriptions embedded in chunks

**Option B**: Store image embeddings separately
- Create separate vector for each image
- Enable image similarity search
- More complex but powerful

**Option C**: Store images in blob + reference
- Keep images in Azure Blob Storage
- Index contains blob URLs
- Retrieve images on-demand for RAG

**Recommendation**: Start with Option A, expand to B/C later

---

## 📄 File Summary Generation Strategy (DETAILED)

### Overview
The `file_summary` field provides a concise document overview combining:
1. **Document purpose** (what is this document for?)
2. **Page-by-page key sentences** (what's on each page?)

This approach is **extractive** (no LLM cost) and preserves the document's natural flow.

### Implementation Approach

#### Step 1: Document Purpose Extraction

**Goal**: Identify WHY this document exists

**Methods**:
```python
def extract_document_purpose(markdown_content, title, first_page_content):
    """
    Extract document purpose using multiple strategies
    """
    purpose = ""

    # Strategy 1: Look for explicit purpose statements
    purpose_patterns = [
        r"Purpose:?\s*(.+?)(?:\n|$)",
        r"This document (?:describes|outlines|provides|explains)\s+(.+?)(?:\.|$)",
        r"Overview:?\s*(.+?)(?:\n|$)",
        r"Introduction:?\s*(.+?)(?:\n|$)",
    ]

    for pattern in purpose_patterns:
        match = re.search(pattern, first_page_content, re.IGNORECASE)
        if match:
            purpose = match.group(1).strip()
            break

    # Strategy 2: If no explicit purpose, use first substantial paragraph
    if not purpose:
        paragraphs = first_page_content.split('\n\n')
        for para in paragraphs:
            if len(para.strip()) > 50:  # Substantial paragraph
                purpose = para.strip()[:300]  # First 300 chars
                break

    # Strategy 3: Infer from title and document type
    if not purpose and title:
        doc_type = infer_document_type(title)
        purpose = f"This is a {doc_type} document titled '{title}'"

    return purpose
```

**Example Output**:
```
Purpose: This document provides guidelines for life insurance policy
applications in Hong Kong, including eligibility criteria, required
documentation, and the application process.
```

#### Step 2: Page-by-Page Key Sentence Extraction

**Goal**: Extract 1-2 most important sentences from each page

**Selection Algorithm**:
```python
def extract_key_sentences_from_page(page_content, page_number):
    """
    Extract key sentences using scoring algorithm
    """
    sentences = split_into_sentences(page_content)
    scored_sentences = []

    for sentence in sentences:
        score = calculate_sentence_importance(sentence, page_content)
        scored_sentences.append((sentence, score))

    # Sort by score and take top 1-2
    scored_sentences.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [s[0] for s in scored_sentences[:2]]

    return top_sentences

def calculate_sentence_importance(sentence, page_context):
    """
    Score sentence based on multiple criteria
    """
    score = 0

    # Criteria 1: Position (first sentences are often important)
    if is_first_sentence_of_section(sentence, page_context):
        score += 10

    # Criteria 2: Formatting (headers, bold text)
    if has_header_formatting(sentence):
        score += 15
    if has_bold_formatting(sentence):
        score += 5

    # Criteria 3: Key terms (domain-specific)
    key_terms = [
        'policy', 'premium', 'coverage', 'claim', 'benefit',
        'requirement', 'must', 'required', 'important', 'note',
        'effective', 'expires', 'valid', 'applicable'
    ]
    term_count = sum(1 for term in key_terms if term.lower() in sentence.lower())
    score += term_count * 3

    # Criteria 4: Contains numbers/dates (often important facts)
    if contains_numbers(sentence):
        score += 5
    if contains_dates(sentence):
        score += 5

    # Criteria 5: Action words (indicates instructions)
    action_words = ['must', 'shall', 'should', 'required', 'provides', 'includes']
    if any(word in sentence.lower() for word in action_words):
        score += 8

    # Criteria 6: Sentence length (not too short, not too long)
    word_count = len(sentence.split())
    if 10 <= word_count <= 30:
        score += 5
    elif word_count < 5:
        score -= 10  # Too short, probably not substantial

    # Criteria 7: Question sentences (often highlight key points)
    if sentence.strip().endswith('?'):
        score += 7

    return score
```

**Example Processing**:
```
Page 1:
  "Prudential Hong Kong offers comprehensive life insurance solutions."
  "This policy provides coverage up to HKD 10 million with flexible terms."
  → Score sentences → Select: "This policy provides coverage up to HKD 10 million..."

Page 2:
  "# Eligibility Requirements"
  "Applicants must be between 18-65 years old and residents of Hong Kong."
  → Header detected → Select: "Applicants must be between 18-65 years old..."

Page 3:
  "The premium calculation considers age, health status, and coverage amount."
  → Contains key terms + numbers → Select this sentence
```

#### Step 3: Combine into Final Summary

**Output Format**:
```python
def generate_file_summary(document_purpose, page_summaries):
    """
    Combine purpose + page summaries into final summary
    """
    summary_parts = [document_purpose]
    summary_parts.append("\n\nPage-by-Page Summary:")

    for page_num, sentences in page_summaries.items():
        summary_parts.append(f"\n- Page {page_num}: {' '.join(sentences)}")

    final_summary = ''.join(summary_parts)

    # Limit total length (optional)
    if len(final_summary) > 2000:
        final_summary = final_summary[:2000] + "..."

    return final_summary
```

**Example Output**:
```
Purpose: This document provides guidelines for life insurance policy
applications in Hong Kong, including eligibility criteria, required
documentation, and the application process.

Page-by-Page Summary:
- Page 1: This policy provides coverage up to HKD 10 million with flexible
  payment terms ranging from 5 to 30 years.
- Page 2: Applicants must be between 18-65 years old and permanent residents
  of Hong Kong or Macau.
- Page 3: The premium calculation considers age, health status, and desired
  coverage amount with annual reviews.
- Page 4: Required documents include HKID card, proof of income, and medical
  examination results from approved clinics.
- Page 5: The application process takes 7-14 business days with fast-track
  options available for qualified applicants.
```

### Benefits of This Approach

1. **Fast Processing**: No LLM API calls, purely rule-based
2. **No Cost**: No additional API charges
3. **Consistent**: Deterministic output, reproducible results
4. **Page Reference**: Maintains page structure for easy lookup
5. **Preserves Facts**: Extracts actual sentences, no paraphrasing errors
6. **Language Agnostic**: Works with English and Traditional Chinese
7. **Scalable**: Can process 100+ page documents efficiently

### Handling Special Cases

#### Case 1: Very Long Documents (100+ pages)
```python
# Adaptive sampling: Extract from every Nth page
if total_pages > 100:
    sample_interval = max(1, total_pages // 50)  # Sample ~50 pages
    pages_to_process = range(0, total_pages, sample_interval)
else:
    pages_to_process = range(total_pages)
```

#### Case 2: Documents with Images
```python
# Include image descriptions in key sentences
if page_has_images(page_number):
    key_sentences.append(f"[Contains {image_count} images: {image_types}]")
```

#### Case 3: Bilingual Documents (EN + TC)
```python
# Detect language per page and extract accordingly
if is_chinese(page_content):
    key_sentences = extract_key_sentences_chinese(page_content)
else:
    key_sentences = extract_key_sentences_english(page_content)
```

#### Case 4: Tables and Structured Data
```python
# Summarize tables with key metrics
if page_has_tables(page_content):
    table_summary = summarize_table_content(tables)
    key_sentences.append(f"[Table: {table_summary}]")
```

### Service Implementation

```python
# New service: src/services/summary_generation_service.py

class SummaryGenerationService:
    """
    Generates document summaries using extractive methods
    """

    def __init__(self):
        self.key_terms = self.load_domain_terms()
        self.action_words = ['must', 'shall', 'should', 'required', 'provides']

    def generate_file_summary(self, enriched_document):
        """
        Main method: Generate file_summary from enriched document

        Args:
            enriched_document: EnrichedDocument with markdown + images

        Returns:
            str: Complete file summary
        """
        # Step 1: Extract document purpose
        purpose = self.extract_document_purpose(
            title=enriched_document.title,
            first_page=enriched_document.pages[0]
        )

        # Step 2: Extract key sentences page by page
        page_summaries = {}
        for page_num, page_content in enumerate(enriched_document.pages, start=1):
            key_sentences = self.extract_key_sentences_from_page(
                page_content,
                page_num
            )
            if key_sentences:
                page_summaries[page_num] = key_sentences

        # Step 3: Combine into final summary
        file_summary = self.combine_summary(purpose, page_summaries)

        return file_summary

    def extract_document_purpose(self, title, first_page):
        """Extract document purpose from title and first page"""
        # Implementation as shown above
        pass

    def extract_key_sentences_from_page(self, page_content, page_number):
        """Extract 1-2 key sentences from page"""
        # Implementation as shown above
        pass

    def calculate_sentence_importance(self, sentence, context):
        """Score sentence based on importance criteria"""
        # Implementation as shown above
        pass

    def combine_summary(self, purpose, page_summaries):
        """Combine purpose + page summaries"""
        # Implementation as shown above
        pass
```

### Configuration Options

```bash
# Add to .env

# Summary Generation Settings
SUMMARY_METHOD=extractive
SUMMARY_MAX_LENGTH=2000
SUMMARY_SENTENCES_PER_PAGE=2
SUMMARY_INCLUDE_PURPOSE=true
SUMMARY_INCLUDE_PAGE_NUMBERS=true
SUMMARY_SAMPLE_LARGE_DOCS=true
SUMMARY_LARGE_DOC_THRESHOLD=100
```

### Testing Strategy

```python
# Test with sample documents
def test_summary_generation():
    """
    Test summary generation with various document types
    """
    test_cases = [
        {
            'type': 'insurance_policy',
            'pages': 10,
            'expected_purpose': 'policy terms',
            'expected_pages': 10
        },
        {
            'type': 'application_form',
            'pages': 5,
            'expected_purpose': 'application instructions',
            'expected_pages': 5
        },
        {
            'type': 'product_guide',
            'pages': 50,
            'expected_purpose': 'product features',
            'expected_pages': 50
        }
    ]

    for test in test_cases:
        summary = generate_summary(test['document'])
        assert_contains_purpose(summary)
        assert_has_page_summaries(summary, test['expected_pages'])
        assert_within_length_limit(summary, 2000)
```

---

## ❓ Open Questions for Clarification

### 1. File Summary Generation ✅ DECIDED
**Approach**: Extractive method focusing on document purpose and content summary

**Implementation**:
1. **Document Purpose Extraction**:
   - Extract first 1-2 paragraphs (typically contains document purpose)
   - Identify document type from title/headers (policy, guide, form, etc.)
   - Look for purpose keywords: "This document...", "Purpose:", "Overview:"

2. **Page-by-Page Key Sentence Extraction**:
   - For each page: Extract 1-2 most important sentences
   - Selection criteria:
     - Sentences with headers/bold text
     - First sentence of each major section
     - Sentences with key terms (dates, amounts, names)
     - Sentences with action words (must, required, provides)
   - Combine extracted sentences maintaining page order

3. **Final Summary Structure**:
   ```
   [Document Purpose]

   Page-by-Page Summary:
   - Page 1: [Key sentence(s)]
   - Page 2: [Key sentence(s)]
   - Page 3: [Key sentence(s)]
   ...
   ```

**Benefits**: Fast, no API cost, preserves key information, maintains document flow

### 2. Authentication Method
**Question**: Should we use:
- **Option A**: API keys (simpler, less secure)
- **Option B**: Managed identity (more secure, production-ready)

**Recommendation**: API keys for MVP, managed identity for production

### 3. Large Document Handling
**Question**: For documents with 100+ pages:
- **Option A**: Process entire document at once
- **Option B**: Break into batches
- **Option C**: Set page limit

**Recommendation**: Process in batches to avoid timeout

### 4. CSV Caching
**Question**: Should we:
- **Option A**: Load CSV once at startup (fast, memory-efficient)
- **Option B**: Reload for each document (slower, always fresh)

**Recommendation**: Load once at startup, add refresh option

### 5. Missing CSV Entries
**Question**: When filename not found in CSV:
- **Option A**: Skip document with warning
- **Option B**: Use fallback metadata (library="Unknown", category="Uncategorized")
- **Option C**: Halt pipeline with error

**Recommendation**: Use fallback metadata (already in project_brief.md)

### 6. Unique ID Generation
**Question**: How to generate chunk IDs?
- **Option A**: UUID
- **Option B**: Hash of content
- **Option C**: Pattern like `{doc_id}_{page}_{chunk_num}`

**Recommendation**: Option C for traceability

### 7. Incremental Updates
**Question**: Should we support re-processing updated documents?
- **Option A**: Delete old chunks + re-index (full refresh)
- **Option B**: Compare and update only changed chunks
- **Option C**: No support (manual index deletion required)

**Recommendation**: Option A for MVP

### 8. Batch Size for Indexing
**Question**: How many documents to upload per batch?
- **Option A**: 1000 chunks per batch (Azure recommended)
- **Option B**: Dynamic based on size
- **Option C**: All chunks for one document per batch

**Recommendation**: Option A (1000 chunks)

---

## 🔄 Key Changes from Original Design

### Major Architectural Changes

1. **Flat → Hierarchical Structure**
   - **OLD**: Flat list of chunks with metadata_file and metadata_chunk JSON strings
   - **NEW**: Document → Pages → Chunks hierarchy with nested objects

2. **Dual → Triple Vector Embeddings**
   - **OLD**: file_summary_chunk [1536] + content_chunk_dim [1536]
   - **NEW**: file_summary_vector [1536] + page_summary_vector [1536] + content_chunk_vector [1536]

3. **New Required Features**
   - **Keyword Extraction**: Entities, product names, topics, file types, key phrases
   - **Synthetic Q&A Generation**: Questions/answers for training and evaluation
   - **Chunk Semantic Density**: Quality scoring (0.0-1.0) for each chunk
   - **Page-Level Summaries**: Each page gets its own summary and embedding
   - **Processing Metadata**: OCR confidence, timing, errors/warnings tracking

4. **Enhanced Metadata**
   - **OLD**: Basic file_name, doc_id, library, category, title
   - **NEW**: + branch_name (HK/MACAU), document_language (en/zh-HK/zh-CN), processing_version, page_count, extracted_at

5. **Table/Image Handling**
   - **OLD**: Images embedded in markdown, tables in chunks
   - **NEW**: Separate `tables[]` and `images[]` arrays with dedicated metadata structures

### Impact on Implementation

- **Complexity**: Significantly increased due to hierarchical structure and new services
- **LLM Usage**: Higher cost due to page summaries, keyword extraction, Q&A generation, semantic density scoring
- **Storage**: Larger index size due to triple vectors and richer metadata
- **Processing Time**: Longer due to additional LLM calls per document
- **Benefits**: Much better retrieval quality, page-level search, Q&A training data, quality filtering

---

## ✅ Success Criteria (UPDATED)

### Core Functionality
- [ ] Successfully processes PDFs, DOCX, images, text files
- [ ] Maintains hierarchical structure (doc → pages → chunks)
- [ ] Chunks maintain semantic coherence and page boundaries
- [ ] Tables and structure preserved with HTML/MD/CSV representations
- [ ] CSV metadata lookup works for all files

### Schema Compliance
- [ ] `file_metadata` correctly populated with all required fields
- [ ] `file_summary` includes file_function_summary + file_summary_vector [1536]
- [ ] Each page has `page_summary` with page_function_summary + page_summary_vector [1536]
- [ ] Each chunk has `content_chunk_vector` [1536]
- [ ] All vector dimensions are exactly 1536

### New Features
- [ ] Keyword extraction generates entities, product names, topics, file types, key phrases
- [ ] Synthetic Q&A pairs generated with confidence scores
- [ ] Chunk semantic density calculated (0.0-1.0)
- [ ] Processing metadata tracked (OCR confidence, timing, errors)
- [ ] Tables extracted with all 3 formats (HTML/MD/CSV)
- [ ] Images extracted with descriptions and metadata

### Quality & Reliability
- [ ] Missing CSV entries handled gracefully with fallbacks
- [ ] Error handling prevents pipeline failures
- [ ] Low-quality chunks identified via semantic density
- [ ] Bilingual content (EN/TC) handled correctly
- [ ] Branch routing (HK/MACAU) works correctly

### Performance
- [ ] Process 100 documents in < 20 minutes (adjusted for new features)
- [ ] Embedding generation doesn't exceed rate limits
- [ ] Memory usage stays reasonable for large documents (100+ pages)

### Code Quality
- [ ] Code is modular, testable, maintainable
- [ ] All services properly abstracted
- [ ] Comprehensive error logging
- [ ] Unit tests for each service
- [ ] Integration tests for end-to-end flow

---

## 🔗 Integration with Existing Agentic RAG System

This ingestion pipeline creates the **Main Search Index** that is used by:

**Existing System**: `src/multiagent_workflow.py` (parent directory)
- `node_main_original`: Searches THIS index we're building
- `node_main_prefilter`: Searches THIS index with pre-filtered filenames

**Index Connection**:
```python
# From parent CLAUDE.md:
# Main Search Index (lines 16-21 in .env):
AZURE_SEARCH_SERVICE=phkl-test-weu
AZURE_SEARCH_INDEX=reindex-nov-2025  # <-- THIS IS THE INDEX WE'RE BUILDING
```

**Flow**:
1. We build ingestion pipeline → populate `reindex-nov-2025` index
2. Existing RAG workflow reads from `reindex-nov-2025` index
3. Users get responses based on our indexed documents

---

## 📊 Comparison: Current State vs Target State

### Current State (As of Now)
```
indexing_pipeline/
├── main.py (just prints "Hello")
├── pyproject.toml (basic setup)
├── chunking/ (empty)
└── project_definition/ (extensive docs ✅)
```

### Target State (After Implementation)
```
indexing_pipeline/
├── src/ (complete implementation)
├── metadata/ (CSV file + samples)
├── tests/ (comprehensive tests)
├── .env (configured)
├── main.py (full orchestrator)
└── README.md (setup guide)
```

---

## 🎬 Next Steps

### Immediate Actions:
1. **Clarify open questions** (especially file summary generation method)
2. **Set up project structure** (create src/ folders)
3. **Install dependencies** (add to pyproject.toml)
4. **Create sample CSV** (metadata/document_metadata.csv)
5. **Implement Phase 1** (project setup)

### Development Order (REVISED):
```
Phase 1: Setup ✅
  ↓
Phase 2: Core Services (Blob, Doc Intel, OpenAI, Search)
  ↓
Phase 3: CSV Metadata Enrichment
  ↓
Phase 3.5: Image Processing Service
  ↓
Phase 4: Summary Generation (File + Page) ⚠️ BEFORE CHUNKING
  ↓
Phase 5: Advanced Page Enrichment (Keywords, Q&A, Tables) ⚠️ BEFORE CHUNKING
  ↓
Phase 6: Chunking (AFTER all page enrichment)
  ↓
Phase 7: Document Processing Pipeline (orchestrate all steps)
  ↓
Phase 8: Azure AI Search Integration
  ↓
Phase 9: Main Orchestrator
  ↓
Phase 10: Error Handling & Logging
  ↓
Phase 11: Testing & Documentation
```

**Key Change**: Phases 4-5 (enrichment) now come BEFORE Phase 6 (chunking) to ensure full page context is available for all enrichment operations.

---

## 📝 Summary in One Paragraph (UPDATED)

We are building a **hierarchical document ingestion pipeline** that reads files from Azure Blob Storage, routes them to appropriate chunkers (DocAnalysisChunker for PDFs/images, LangChainChunker for text files), and creates a **multi-level structured schema** (Document → Pages → Chunks). The pipeline generates **three levels of embeddings** using Azure OpenAI (text-embedding-3-small with 1536 dimensions): file-level summaries, page-level summaries, and chunk-level vectors. It enriches documents with **keyword extraction** (entities, products, topics), generates **synthetic Q&A pairs** for training, calculates **chunk semantic density** for quality scoring, and extracts **tables in multiple formats** (HTML/MD/CSV) along with **image descriptions**. CSV metadata lookup populates file metadata including branch (HK/MACAU) and language (en/zh-HK/zh-CN). The resulting **hierarchical index** (`reindex-nov-2025`) will be queried by the existing Agentic RAG workflow to provide intelligent, page-aware responses to Prudential Hong Kong users.

---

## 🤝 Ready to Start?

**Confirm understanding and ask**:
1. File summary generation method preference?
2. Any modifications to the schema?
3. Any additional CSV fields needed?
4. Priority order for file types (.pdf first? all types together?)
5. Sample documents available for testing?

**Then proceed with**:
- Creating project structure
- Installing dependencies
- Implementing core services
- Building the pipeline step-by-step
### Updated Execution Order (Priority)
- Document Intelligence extraction (page text, tables, images)
- CSV metadata enrichment first (fuzzy match across `filename`, `file_name`, `file_description`, and `item_url`)
- Page insights before chunking:
  - Generate page-by-page function summaries
  - Generate synthetic Q&A pairs per page
  - Extract keywords/entities/topics per page
- Chunking page-by-page (preserve structure and image/table context)
- Embeddings for file/page/chunk
- ETL JSON writing with empty sections omitted (no duplicate placeholders):
  - Omit `synthetic_qa_pairs`, `tables`, `images` when arrays are empty
  - Omit `keyword_extraction.key_phrases` when not computed
  - Drop `processing_metadata` when effectively empty
