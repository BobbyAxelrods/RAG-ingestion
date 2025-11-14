# Agentic RAG ETL & Indexing

This repository contains an ETL pipeline that chunks and analyzes documents, extracts entities and Q&A pairs via a chat-capable LLM, and produces nested ETL JSON ready for review and later indexing into Azure AI Search. Indexing is intentionally a separate step so extraction can be audited before publishing.

## Overview
- Ingestion: load documents and metadata from `artifact/matched_downloads`.
- Chunking: split documents into page-aligned chunks.
- LLM extraction: generate entities and Q&A using the configured chat deployment.
- Output: write nested ETL JSON via `ExtractionWriter` into `artifact/etl_output`.
- Indexing: read the reviewed ETL JSON and push to Azure AI Search (run separately).

## Project Structure
```
artifact/           # local data, dev scripts, and ETL outputs (gitignored)
logs/               # pipeline logs (gitignored)
project_plan/       # docs, schemas, samples
src/                # ETL, services, models, indexing code
```
Key files:
- `src/etl/services/openai_service.py`: initializes chat and embeddings clients.
- `src/etl/services/page_insights_service.py`: uses chat client to extract entities and Q&A.
- `artifact/_tmp_run_local_doc.py`: local runner to generate nested ETL JSON via `ExtractionWriter`.

## Prerequisites
- Python 3.11+ recommended
- Azure resources: Azure OpenAI (chat deployment), Azure AI Search (for indexing)
- Recommended: a virtual environment

## Setup
1. Create and activate a virtual environment
```
python -m venv .venv
.\.venv\Scripts\activate
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Configure environment
Create `.env` in repo root (kept out of git by `.gitignore`).

Chat (LLM) configuration:
```
CHAT_MODEL_ENDPOINT=https://<your-aoai-resource>.openai.azure.com/
CHAT_MODEL_API_KEY=<your-azure-openai-key>
CHAT_MODEL=<your-chat-deployment-name>
```
Embeddings configuration (for vector generation):
```
AZURE_OPENAI_ENDPOINT=https://<your-aoai-resource>.openai.azure.com/
AZURE_OPENAI_KEY=<your-azure-openai-key>
AZURE_OPENAI_DEPLOYMENT_NAME=<your-embeddings-deployment-name>
AZURE_OPENAI_API_VERSION=2024-02-01
```
Runtime settings:
```
RUNTIME_OFFLINE_MODE=false
```

## Local ETL Run
Use the local runner to process a document and write nested ETL output.
```
python artifact/_tmp_run_local_doc.py
```
This produces `artifact/etl_output/<doc>_extraction.json` with:
- `chunk_data[].chunk_metadata.chunk_entities`: entity categories extracted via chat LLM
- `chunk_data[].chunk_metadata.page_qna_pairs`: grounded Q&A pairs per page

## Review Before Indexing
- Inspect `artifact/etl_output/<doc>_extraction.json` and log output in `logs/pipeline.log`.
- Validate Q&A count and entity coverage. Adjust prompts/configs if needed.
- Only proceed to indexing when extraction quality meets your bar.

## Indexing to Azure AI Search
Indexing is separated from ETL. Typical flow:
- Configure Azure AI Search index schema (`project_plan/schema/aisearch_index_schema.json`).
- Implement or use existing indexing scripts in `src/index/` to read ETL JSON and push documents.
- Validate index population via Azure portal or API queries.

## Security Notes
- Secrets: `.env`, keys, and credentials are ignored by git. Do not commit secrets.
- Artifacts: raw documents and ETL outputs under `artifact/` are ignored by git by default.
- Logs: pipeline logs can contain model metadata; review before sharing.
- Principle of least privilege: provision only necessary Azure keys and roles.

## Troubleshooting
- Zero Q&A output: confirm chat deployment is configured and not an embeddings model.
- 400 OperationNotSupported: ensure `CHAT_MODEL` points to a chat-capable deployment.
- Offline mode: set `RUNTIME_OFFLINE_MODE=false` to enable live LLM calls.
- Parsing issues: strict JSON parsing can drop outputs; consider logging raw LLM responses.

## Development Workflow
- Keep extraction and indexing separate to allow review and QA.
- Use `ExtractionWriter` for nested ETL JSON to maintain consistent schema.
- Add targeted logs to `logs/pipeline.log` for visibility into drops and filters.

## License
Internal project. Do not distribute without authorization.

## Regression Testing (Vector, Hybrid, Simple)

Run the regression script to evaluate all three search methods (vector, hybrid, simple) against a set of questions from an Excel file. Each question generates two JSON outputs (English and Traditional Chinese) containing the combined results from the three search methods.

Prerequisites:
- Set search and embeddings environment variables (place in `.env` or export them in your shell):
  - `SEARCH_SERVICE_ENDPOINT`, `SEARCH_SERVICE_KEY`
  - `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_API_VERSION`
  - `SEARCH_INDEX_NAME` (or pass `--index` on the CLI)

Full run (top 10 per method):
- `python src/search_diagnostic_prod/run_benchmark_regression.py --excel "artifact\data_library\evaluation\benchmark_ii ( steven)\reviewed_pru.xlsx" --top 10 --out results_json --index index_full_rev_1`

Single-sample run (quick check):
- `python src/search_diagnostic_prod/run_benchmark_regression.py --excel "artifact\data_library\evaluation\benchmark_ii ( steven)\reviewed_pru.xlsx" --top 10 --out results_json_test --index index_full_rev_1 --limit 1`

Options:
- `--sheet` selects a specific Excel sheet
- `--index` overrides the index name
- `--top` sets top-k per search method
- `--out` sets output root directory
- `--limit` caps number of questions processed
- `--probe` prints available fields for the active index

Output structure:
- Per question folder under the specified output root (e.g., `results_json\<QuestionID>\`)
- Files: `<QuestionID>_en.json` and `<QuestionID>_tc.json`
- Each file contains `runs` with three entries: `vector`, `hybrid`, `simple`, each including `query`, `search_method`, `total_result`, and a `results[]` array with normalized fields (`id`, `document_id`, `title_name_en`, `title_name_tc`, `content_en`, `content_tc`, `filename`, `page_number`, `score`, `content_chunk`).