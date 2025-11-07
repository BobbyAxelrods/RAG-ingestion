# AI Search Indexing Pipeline - Implementation Plan

## Overview
Build a robust indexing pipeline that processes JSON extraction files, flattens nested structures, validates against Azure AI Search schema, and creates/updates search indexes with progress tracking.

---

## 1. Architecture Overview

### 1.1 Pipeline Components
```
┌─────────────────┐
│  Input Files    │
│  (JSON ETL)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  File Discovery │
│  & Validation   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JSON Flattener │
│  & Transformer  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Schema Mapper  │
│  & Validator    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Index Manager  │
│  (Create/Update)│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Batch Uploader │
│  (w/ Progress)  │
└─────────────────┘
```

### 1.2 Tech Stack
- **Language**: Python 3.11+
- **Azure SDK**: `azure-search-documents`, `azure-identity`
- **Progress Tracking**: `tqdm`, `rich`
- **Logging**: `loguru` or standard `logging`
- **Config Management**: `pydantic-settings` or `python-dotenv`
- **Testing**: `pytest`, `pytest-asyncio`

---

## 2. Core Modules

### 2.1 Configuration Manager (`config.py`)
**Purpose**: Centralized configuration management

**Features**:
- Load Azure AI Search credentials (endpoint, key, index name)
- Support environment variables and config files
- Schema file path configuration
- Batch size settings (default: 1000 documents)
- Retry configuration (max retries, backoff strategy)

**Schema**:
```python
class AzureSearchConfig:
    endpoint: str
    api_key: str
    default_index_name: str
    schema_path: str
    batch_size: int = 1000
    max_retries: int = 3
    timeout_seconds: int = 300
```

---

### 2.2 File Discovery Module (`file_discovery.py`)
**Purpose**: Discover and validate input JSON files

**Features**:
- Support single file or directory input
- Recursive directory scanning for `*_extraction.json` files
- File validation (exists, readable, valid JSON structure)
- Generate file processing queue with metadata

**Functions**:
```python
def discover_files(path: str | Path, pattern: str = "*_extraction.json") -> List[FileInfo]
def validate_json_file(file_path: Path) -> Tuple[bool, Optional[str]]
def get_file_metadata(file_path: Path) -> FileInfo
```

**Output**:
```python
@dataclass
class FileInfo:
    path: Path
    size_bytes: int
    modified_time: datetime
    is_valid: bool
    error_message: Optional[str]
```

---

### 2.3 JSON Flattener (`json_transformer.py`)
**Purpose**: Transform nested JSON to flat structure matching AI Search schema

**Key Transformations**:

#### Input Structure:
```json
{
  "doc_id": "doc_af4a91050dde",
  "system_file_metadata": {
    "sys_file_name": "...",
    "sys_file_path": "...",
    "sys_file_size_bytes": 791688,
    ...
  },
  "file_index_metadata": {
    "file_name": "...",
    "library_name_en": "...",
    ...
  },
  "chunk_data": [
    {
      "chunk_content": "...",
      "chunk_content_vector": [...],
      ...
    }
  ]
}
```

#### Output Structure (Flattened per chunk):
```json
{
  "doc_id": "doc_af4a91050dde_chunk_0",
  "sys_file_name": "...",
  "sys_file_path": "...",
  "sys_file_size_bytes": 791688,
  "file_name": "...",
  "library_name_en": "...",
  "chunk_content": "...",
  "chunk_content_vector": [...],
  ...
}
```

**Functions**:
```python
def flatten_document(doc: Dict[str, Any]) -> List[Dict[str, Any]]
def extract_metadata_fields(doc: Dict) -> Dict[str, Any]
def explode_chunks(doc: Dict, metadata: Dict) -> List[Dict[str, Any]]
def generate_chunk_doc_id(base_doc_id: str, chunk_index: int) -> str
```

**Logic**:
1. Extract `system_file_metadata` fields → flatten with `sys_*` prefix
2. Extract `file_index_metadata` fields → flatten directly
3. For each item in `chunk_data`:
   - Create new document with metadata + chunk fields
   - Generate unique `doc_id` = `{original_doc_id}_chunk_{index}`
   - Preserve all chunk fields (`chunk_content`, `chunk_content_vector`, etc.)

---

### 2.4 Schema Mapper & Validator (`schema_validator.py`)
**Purpose**: Validate documents against AI Search schema

**Features**:
- Load AI Search schema from JSON file
- Build field type mapping from schema
- Validate each document field:
  - Type compatibility (Edm.String, Edm.Int32, Edm.DateTimeOffset, etc.)
  - Required fields (especially `doc_id` as key)
  - Collection fields validation
  - Vector dimensions validation (1536 for embeddings)
- Generate validation report

**Functions**:
```python
def load_schema(schema_path: Path) -> SearchIndexSchema
def validate_document(doc: Dict, schema: SearchIndexSchema) -> ValidationResult
def validate_batch(docs: List[Dict], schema: SearchIndexSchema) -> BatchValidationResult
def convert_to_edm_types(value: Any, field_type: str) -> Any
```

**Type Mappings**:
| Schema Type | Python Type | Validation |
|-------------|-------------|------------|
| Edm.String | str | Length check |
| Edm.Int32 | int | Range: -2³¹ to 2³¹-1 |
| Edm.Int64 | int | Range: -2⁶³ to 2⁶³-1 |
| Edm.Double | float | Finite number |
| Edm.DateTimeOffset | str (ISO 8601) | Format validation |
| Collection(Edm.String) | List[str] | Each item is string |
| Collection(Edm.Single) | List[float] | Vector dimension check |

---

### 2.5 Index Manager (`index_manager.py`)
**Purpose**: Create or verify Azure AI Search index

**Features**:
- Check if index exists by name
- Create index if not exists using schema
- Validate existing index schema matches expected schema
- Handle index creation errors

**Functions**:
```python
async def index_exists(client: SearchIndexClient, index_name: str) -> bool
async def create_index(client: SearchIndexClient, schema_path: Path, index_name: str) -> bool
async def get_index_schema(client: SearchIndexClient, index_name: str) -> Dict
async def validate_index_schema(existing_schema: Dict, expected_schema: Dict) -> SchemaValidationResult
```

**Index Creation Logic**:
1. Check if index with name exists
2. If not exists → Create from schema file
3. If exists → Validate schema compatibility (log warnings for mismatches)
4. Return index client for document operations

---

### 2.6 Batch Uploader (`batch_uploader.py`)
**Purpose**: Upload documents to AI Search with batching and progress tracking

**Features**:
- Batch upload (default: 1000 docs per batch)
- Retry logic with exponential backoff
- Progress bar per file and overall progress
- Error handling and partial failure recovery
- Upload statistics (success/fail/skipped counts)

**Functions**:
```python
async def upload_documents(
    client: SearchClient,
    documents: List[Dict],
    batch_size: int,
    show_progress: bool = True
) -> UploadResult

async def upload_batch(client: SearchClient, batch: List[Dict]) -> BatchUploadResult
def create_progress_tracker(total_docs: int, file_name: str) -> ProgressTracker
```

**Progress Display**:
```
Processing Files: 3/10 (30%)
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫

Current File: document_001.json
Documents: 5,234/10,000 (52.34%) [████████████░░░░░░░░░░]
Batches: 6/10 | Success: 5,234 | Failed: 0 | Rate: 523 docs/sec

Estimated Time Remaining: 00:01:23
```

---

### 2.7 Pipeline Orchestrator (`pipeline.py`)
**Purpose**: Main orchestration logic

**Main Flow**:
```python
async def run_pipeline(
    input_path: str | Path,
    config: AzureSearchConfig,
    index_name: Optional[str] = None
) -> PipelineResult:
    """
    Main pipeline execution flow
    """
    # 1. Initialize
    logger.info("Starting AI Search Indexing Pipeline")

    # 2. Discover files
    files = discover_files(input_path)
    logger.info(f"Found {len(files)} files to process")

    # 3. Load schema
    schema = load_schema(config.schema_path)

    # 4. Initialize Azure clients
    index_client = SearchIndexClient(endpoint, credential)

    # 5. Ensure index exists
    target_index = index_name or config.default_index_name
    await ensure_index(index_client, schema, target_index)

    # 6. Get search client
    search_client = SearchClient(endpoint, target_index, credential)

    # 7. Process each file
    total_stats = Statistics()

    for file_info in track(files, description="Processing files"):
        # 7.1 Load JSON
        doc = load_json(file_info.path)

        # 7.2 Flatten
        flat_docs = flatten_document(doc)

        # 7.3 Validate
        validation = validate_batch(flat_docs, schema)
        if not validation.is_valid:
            logger.error(f"Validation failed: {validation.errors}")
            continue

        # 7.4 Upload
        result = await upload_documents(search_client, flat_docs)
        total_stats.add(result)

        # 7.5 Log progress
        logger.info(f"Uploaded {result.success_count} docs from {file_info.path.name}")

    # 8. Return summary
    return PipelineResult(
        total_files=len(files),
        total_documents=total_stats.total,
        successful_uploads=total_stats.success,
        failed_uploads=total_stats.failed,
        duration=elapsed_time
    )
```

---

## 3. CLI Interface

### 3.1 Command Structure
```bash
# Single file
python index_pipeline.py \
  --input "path/to/file.json" \
  --index "insurance-documents-index" \
  --config "config.env"

# Multiple files (directory)
python index_pipeline.py \
  --input "path/to/etl_output/" \
  --index "insurance-documents-index" \
  --recursive \
  --pattern "*_extraction.json"

# Dry run (validate only, no upload)
python index_pipeline.py \
  --input "path/to/etl_output/" \
  --dry-run

# Custom batch size
python index_pipeline.py \
  --input "path/to/file.json" \
  --batch-size 500
```

### 3.2 CLI Arguments
```python
--input, -i          : Input file or directory path (required)
--index, -idx        : Target index name (optional, uses default from config)
--config, -c         : Config file path (default: .env)
--schema, -s         : Schema file path (default: from config)
--batch-size, -b     : Upload batch size (default: 1000)
--recursive, -r      : Scan directories recursively (flag)
--pattern, -p        : File pattern for discovery (default: *_extraction.json)
--dry-run           : Validate only, skip upload (flag)
--create-index      : Force create index even if exists (flag)
--log-level, -l     : Logging level (DEBUG|INFO|WARNING|ERROR)
--progress          : Show progress bars (default: True)
--output-report, -o : Export summary report to JSON/CSV
```

---

## 4. Error Handling & Recovery

### 4.1 Error Categories
| Error Type | Handling Strategy |
|------------|-------------------|
| File not found | Skip file, log error, continue |
| Invalid JSON | Skip file, log error, save to error log |
| Schema validation fail | Skip document, log details, continue |
| Network timeout | Retry with backoff (max 3 attempts) |
| 429 Rate limit | Exponential backoff (2s → 4s → 8s) |
| 401 Auth error | Fail fast, terminate pipeline |
| Index not found | Create index if flag set, else fail |

### 4.2 Recovery Mechanisms
- **Checkpoint system**: Save progress after each file
- **Resume capability**: Skip already processed files
- **Failed documents log**: Export failed docs to `failed_uploads.json`
- **Validation report**: Export validation errors to `validation_errors.json`

---

## 5. Progress Tracking Requirements

### 5.1 Multi-Level Progress
1. **File Level Progress**
   - Total files to process
   - Current file being processed
   - Files completed/remaining

2. **Document Level Progress**
   - Total documents in current file
   - Documents processed
   - Upload rate (docs/sec)

3. **Batch Level Progress**
   - Current batch number
   - Total batches
   - Batch success rate

### 5.2 Real-Time Metrics
```python
@dataclass
class ProgressMetrics:
    # File metrics
    total_files: int
    completed_files: int
    current_file: str

    # Document metrics
    total_documents: int
    processed_documents: int
    successful_uploads: int
    failed_uploads: int

    # Performance metrics
    elapsed_time: timedelta
    docs_per_second: float
    estimated_time_remaining: timedelta

    # Batch metrics
    current_batch: int
    total_batches: int
    batch_success_rate: float
```

### 5.3 Progress Display Components
- **Console Progress Bar**: Using `rich.progress` or `tqdm`
- **Live Statistics Table**: Real-time metric updates
- **Log File**: Detailed timestamped events
- **Summary Report**: JSON/CSV export at completion

---

## 6. Testing Strategy

### 6.1 Unit Tests
- `test_json_transformer.py`: Test flattening logic
- `test_schema_validator.py`: Test type conversions and validations
- `test_file_discovery.py`: Test file scanning
- `test_batch_uploader.py`: Mock upload tests

### 6.2 Integration Tests
- `test_pipeline_e2e.py`: End-to-end with mock Azure client
- `test_index_creation.py`: Test index management

### 6.3 Test Data
- Sample JSON files with various structures
- Edge cases: Empty chunks, missing fields, invalid types
- Large file tests (10k+ documents)

---

## 7. Configuration Files

### 7.1 `.env` Configuration
```env
# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
AZURE_SEARCH_API_KEY=your-api-key
AZURE_SEARCH_INDEX_NAME=insurance-documents-index

# Pipeline Settings
SCHEMA_FILE_PATH=./project_plan/schema/aisearch_index_schema.json
BATCH_SIZE=1000
MAX_RETRIES=3
TIMEOUT_SECONDS=300

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=./logs/indexing_pipeline.log
```

### 7.2 `pipeline_config.yaml` (Alternative)
```yaml
azure_search:
  endpoint: ${AZURE_SEARCH_ENDPOINT}
  api_key: ${AZURE_SEARCH_API_KEY}
  default_index_name: insurance-documents-index

schema:
  file_path: ./project_plan/schema/aisearch_index_schema.json

pipeline:
  batch_size: 1000
  max_retries: 3
  timeout_seconds: 300
  checkpoint_enabled: true
  checkpoint_file: ./.pipeline_checkpoint.json

logging:
  level: INFO
  file_path: ./logs/indexing_pipeline.log
  console_output: true
```

---

## 8. Project Structure

```
src/
├── index/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── file_discovery.py      # File discovery and validation
│   ├── json_transformer.py    # JSON flattening logic
│   ├── schema_validator.py    # Schema validation
│   ├── index_manager.py       # Azure index management
│   ├── batch_uploader.py      # Document upload with batching
│   ├── pipeline.py            # Main orchestration
│   ├── progress_tracker.py    # Progress tracking utilities
│   ├── models.py              # Data models and types
│   └── utils.py               # Helper functions
│
├── cli/
│   └── index_pipeline.py      # CLI entry point
│
├── tests/
│   ├── unit/
│   │   ├── test_json_transformer.py
│   │   ├── test_schema_validator.py
│   │   ├── test_file_discovery.py
│   │   └── test_batch_uploader.py
│   ├── integration/
│   │   └── test_pipeline_e2e.py
│   └── fixtures/
│       └── sample_data/
│
├── logs/
│   └── .gitkeep
│
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 9. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Set up project structure
- [ ] Implement configuration management
- [ ] Implement file discovery module
- [ ] Write unit tests for file discovery

### Phase 2: Data Transformation (Week 1)
- [ ] Implement JSON flattener
- [ ] Implement schema validator
- [ ] Write unit tests for transformations
- [ ] Create test fixtures

### Phase 3: Azure Integration (Week 2)
- [ ] Implement index manager
- [ ] Implement batch uploader
- [ ] Add retry and error handling
- [ ] Write integration tests

### Phase 4: Progress & CLI (Week 2)
- [ ] Implement progress tracker
- [ ] Build CLI interface
- [ ] Add logging and reporting
- [ ] Create documentation

### Phase 5: Testing & Refinement (Week 3)
- [ ] End-to-end testing
- [ ] Performance optimization
- [ ] Error recovery testing
- [ ] User acceptance testing

---

## 10. Key Implementation Details

### 10.1 Document ID Generation
```python
def generate_chunk_doc_id(base_doc_id: str, chunk_index: int, page_number: Optional[int] = None) -> str:
    """
    Generate unique document ID for each chunk
    Format: {base_doc_id}_chunk_{index}_{page_number}
    Example: doc_af4a91050dde_chunk_0_1
    """
    if page_number is not None:
        return f"{base_doc_id}_chunk_{chunk_index}_p{page_number}"
    return f"{base_doc_id}_chunk_{chunk_index}"
```

### 10.2 Vector Field Handling
```python
def validate_vector_field(vector: List[float], expected_dimensions: int) -> bool:
    """
    Validate vector embeddings
    - Must be list of floats
    - Must match expected dimensions (1536 for text-embedding-3-small)
    - All values must be finite
    """
    if not isinstance(vector, list):
        return False
    if len(vector) != expected_dimensions:
        return False
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vector)
```

### 10.3 Batch Upload with Retry
```python
async def upload_batch_with_retry(
    client: SearchClient,
    batch: List[Dict],
    max_retries: int = 3
) -> BatchUploadResult:
    """
    Upload batch with exponential backoff retry
    """
    for attempt in range(max_retries):
        try:
            result = await client.upload_documents(documents=batch)
            return BatchUploadResult(
                success_count=len([r for r in result if r.succeeded]),
                failed_count=len([r for r in result if not r.succeeded]),
                errors=[r.error_message for r in result if not r.succeeded]
            )
        except HttpResponseError as e:
            if e.status_code == 429 and attempt < max_retries - 1:
                # Rate limited - exponential backoff
                wait_time = 2 ** attempt
                logger.warning(f"Rate limited. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
```

### 10.4 Date Format Conversion
```python
def convert_to_edm_datetime(value: str) -> str:
    """
    Convert datetime string to Edm.DateTimeOffset format
    Input: "2025-11-03T23:29:42.275716"
    Output: "2025-11-03T23:29:42.275716Z"
    """
    if not value.endswith('Z'):
        return f"{value}Z"
    return value
```

---

## 11. Monitoring & Observability

### 11.1 Logging Levels
- **DEBUG**: Detailed transformation steps, API calls
- **INFO**: File processing start/end, upload progress
- **WARNING**: Validation failures, retries
- **ERROR**: Upload failures, schema mismatches

### 11.2 Metrics to Track
- Total processing time
- Documents per second
- Success/failure rates
- Average batch upload time
- Memory usage (for large files)

### 11.3 Output Reports
```json
{
  "pipeline_run": {
    "start_time": "2025-11-06T10:00:00Z",
    "end_time": "2025-11-06T10:15:32Z",
    "duration_seconds": 932,
    "status": "completed"
  },
  "input": {
    "files_discovered": 15,
    "files_processed": 15,
    "files_failed": 0
  },
  "documents": {
    "total_documents": 45230,
    "successful_uploads": 45230,
    "failed_uploads": 0,
    "validation_errors": 0
  },
  "performance": {
    "avg_docs_per_second": 48.5,
    "total_batches": 46,
    "avg_batch_time_ms": 1250
  },
  "errors": []
}
```

---

## 12. Security Considerations

- **API Key Management**: Use Azure Key Vault or environment variables (never hardcode)
- **Input Validation**: Sanitize all file paths to prevent path traversal
- **Logging**: Mask sensitive data in logs (API keys, personal information)
- **Network**: Use HTTPS for all Azure API calls
- **Access Control**: Ensure service principal has minimum required permissions

---

## 13. Performance Optimization

### 13.1 Batching Strategy
- Default batch size: 1000 documents
- Adjust based on document size (reduce for large documents with vectors)
- Use async/await for concurrent batch uploads

### 13.2 Memory Management
- Stream large JSON files instead of loading entirely
- Process files one at a time
- Clear document batches after upload

### 13.3 Network Optimization
- Use connection pooling
- Enable compression for API calls
- Implement rate limiting to avoid 429 errors

---

## 14. Future Enhancements

- [ ] Support for incremental updates (only changed documents)
- [ ] Delta detection based on file hash or timestamp
- [ ] Parallel file processing (multi-threading)
- [ ] Web UI for monitoring pipeline runs
- [ ] Integration with Azure Data Factory
- [ ] Support for other document formats (CSV, Parquet)
- [ ] Automatic schema migration for index updates
- [ ] Slack/Teams notifications for pipeline completion

---

## 15. Success Criteria

✅ **Functional Requirements**:
- Pipeline successfully processes multiple JSON files
- Index is created if it doesn't exist
- Documents are correctly flattened and mapped to schema
- Upload progress is clearly displayed
- Error handling allows pipeline to continue on failures

✅ **Performance Requirements**:
- Process at least 10,000 documents per minute
- Handle files up to 500MB in size
- Memory usage remains below 2GB during processing

✅ **Quality Requirements**:
- 95%+ test coverage for core modules
- Zero data loss during processing
- All validation errors are logged with details
- Pipeline can resume from checkpoint on failure

---

## Appendix A: Sample Commands

### A.1 Development Testing
```bash
# Run with sample data
python src/cli/index_pipeline.py \
  --input ./tests/fixtures/sample_data/ \
  --dry-run \
  --log-level DEBUG

# Run single file
python src/cli/index_pipeline.py \
  --input ./artifact/etl_output/sample_extraction.json \
  --index test-index \
  --batch-size 100
```

### A.2 Production Usage
```bash
# Full pipeline run
python src/cli/index_pipeline.py \
  --input ./artifact/etl_output/ \
  --recursive \
  --index insurance-documents-index \
  --output-report ./reports/pipeline_run_$(date +%Y%m%d_%H%M%S).json

# Resume from checkpoint
python src/cli/index_pipeline.py \
  --input ./artifact/etl_output/ \
  --resume-from-checkpoint
```

---

## Appendix B: Dependencies

### B.1 Python Packages
```txt
# Azure SDK
azure-search-documents>=11.4.0
azure-identity>=1.15.0

# Progress & UI
rich>=13.7.0
tqdm>=4.66.0

# Config & Validation
pydantic>=2.5.0
pydantic-settings>=2.1.0
python-dotenv>=1.0.0

# Logging
loguru>=0.7.2

# Testing
pytest>=7.4.3
pytest-asyncio>=0.21.1
pytest-mock>=3.12.0

# Utilities
click>=8.1.7  # CLI framework
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-06
**Author**: AI Search Indexing Team
