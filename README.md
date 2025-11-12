# myRAG - Financial Document QA System

A Retrieval-Augmented Generation (RAG) system for question answering on financial documents.

## Installation

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Create `.env` file with API keys
```bash
TOGETHER_API_KEY=your_together_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

3. Add PDF documents to `data/test_files/finance-bench-pdfs/` from `https://github.com/patronus-ai/financebench/tree/main/pdfs`

## Quick Start

```bash
# Build vector database from PDFs
python src/create_database.py

# Interactive query interface
python src/retrieval.py

# Bulk testing on FinanceBench
python src/bulk_testing.py --dataset financebench
```

## Project Structure

```
myRAG/
├── src/                    # Core source code
│   ├── create_database.py  # Build vector database
│   ├── retrieval.py        # Interactive queries
│   └── bulk_testing.py     # Bulk evaluation
├── dataset_adapters/       # Dataset loading adapters
├── evaluation/             # Evaluation metrics
├── tests/                  # Testing utilities
├── data/                   # Input PDF documents
├── chroma/                 # Vector database (auto-created)
└── bulk_runs/              # Test results (auto-created)
```

## How It Works

### 1. Database Creation (`src/create_database.py`)

**Purpose:** Process PDFs and build searchable vector database

**Process:**
1. `load_documents()`: Load PDFs from `data/test_files/finance-bench-pdfs/`
   - Uses UnstructuredPDFLoader to extract text
   - Handles errors for individual files
   - Returns list of Document objects

2. `split_text(documents)`: Chunk documents into smaller pieces
   - Uses RecursiveCharacterTextSplitter
   - Chunk size: 1000 characters
   - Overlap: 200 characters (preserves context across boundaries)
   - Returns list of text chunks with metadata (source, start_index)

3. `save_to_chroma(chunks)`: Store chunks in vector database
   - Embeds each chunk using BAAI/bge-base-en-v1.5 (Together API)
   - Saves to ChromaDB at `chroma/`
   - Processes in batches of 500 for memory efficiency

**Run:** `python src/create_database.py`

### 2. Query Interface (`src/retrieval.py`)

**Purpose:** Interactive question-answering

**Process:**
1. `load_database()`: Load ChromaDB from disk
   - Connects to existing database at `chroma/`
   - Loads embedding function (BAAI/bge-base-en-v1.5)

2. `create_retriever()`: Create retriever with top-k=5
   - Wraps database in retriever interface
   - Returns 5 most similar chunks for any query

3. `main()`: Interactive loop
   - Get user question
   - Retrieve top-5 relevant chunks using semantic search
   - Combine chunks into context string
   - Send context + question to LLM (Llama 3.1 70B via Together API)
   - Display answer and source documents

**Run:** `python src/retrieval.py`

### 3. Bulk Testing (`src/bulk_testing.py`)

**Purpose:** Evaluate system on benchmark datasets

**Key Components:**

**BulkTestConfig:**
- Dataclass storing all settings (model, top-k, temperature, paths)
- Generates descriptive filenames from config
- Resolves paths relative to project root

**BulkTestRunner:**

1. `initialize_framework()`:
   - Load ChromaDB once (reuse for all questions)
   - Create retriever with specified top-k
   - Initialize Together API client

2. `process_single_question(question, question_id)`:
   - **Retrieval**: Get top-k chunks, time the operation
   - **Generation**: Send context + question to LLM, time the operation
   - **Error handling**: Catch API errors, return error message if fails
   - Returns: predicted_answer, sources, timing, error

3. `run_bulk_test(adapter)`:
   - Load dataset using adapter (e.g., FinanceBenchAdapter)
   - Initialize framework once
   - Loop through all questions with progress bar (tqdm)
   - For each question:
     - Process through RAG pipeline
     - Calculate semantic similarity vs gold answer
     - Store results (question, answer, similarity, timing, sources, errors)
   - Handle interruptions (Ctrl+C, rate limits) gracefully
   - Return results DataFrame

4. `save_results(results_df, adapter)`:
   - Calculate aggregate metrics (mean/min/max similarity, by question type)
   - Save CSV with all question-level results
   - Save JSON with summary statistics
   - Print formatted summary to terminal

**Run:** `python src/bulk_testing.py --dataset financebench --top-k 5`

### 4. Dataset Adapters (`dataset_adapters/`)

**Purpose:** Modular interface for loading different datasets

**BaseDatasetAdapter:**
- Abstract class defining required methods
- `load_dataset()`: Load dataset as DataFrame
- `get_question_column()`: Return column name for questions
- `get_answer_column()`: Return column name for gold answers
- `get_question_type_column()`: Return column name for question types
- `get_metadata_columns()`: Return additional columns to include
- `get_dataset_name()`: Return short abbreviation for filenames

**FinanceBenchAdapter:**
- Loads PatronusAI/financebench from HuggingFace (150 questions)
- Maps columns: question, answer, question_type, company, doc_name, doc_type
- Returns dataset abbreviation: "fb"

**Adding new datasets:** Create new adapter implementing BaseDatasetAdapter

### 5. Evaluation Metrics (`evaluation/metrics.py`)

**Functions:**

1. `normalize_text(text)`:
   - Lowercase, strip whitespace, remove edge punctuation
   - Used for text comparison

2. `embedding_similarity(predicted, gold, embeddings)`:
   - Embed both answers using BAAI/bge-base-en-v1.5
   - Calculate cosine similarity between vectors
   - Returns score 0.0 (different) to 1.0 (identical)
   - Better than exact match (captures meaning, not just words)

3. `calculate_aggregate_metrics(results_df)`:
   - Filter successful vs failed predictions
   - Calculate mean/min/max semantic similarity
   - Group by question type and calculate per-type stats
   - Calculate average timing (retrieval, generation)
   - Return summary dictionary

4. `format_metrics_summary(metrics)`:
   - Format metrics as readable string
   - Display overall stats, per-type breakdown, timing info

### 6. Testing Utilities (`tests/`)

**test_bulk_system.py:**
- Quick validation test (1 question)
- Verifies full pipeline works end-to-end
- Run: `python tests/test_bulk_system.py`

**check_db.py:**
- Display database chunk count and progress
- Run: `python tests/check_db.py`

**view_database.py:**
- Inspect database contents (sources, sample documents)
- Run: `python tests/view_database.py`

## Configuration

### Command-Line Options (bulk_testing.py)

```bash
--dataset financebench     # Dataset name
--top-k 5                  # Number of chunks to retrieve
--temperature 0.0          # Generation randomness (0=deterministic)
--max-tokens 512           # Max response length
--model [model_name]       # LLM model to use
```

### Editing Settings

**Database creation** (`src/create_database.py`):
- `CHROMA_PATH`: Database location
- `DATA_PATH`: PDF source directory
- Chunk size: 1000 characters
- Chunk overlap: 200 characters

**Retrieval** (`src/retrieval.py`):
- Top-k: 5 documents (default in `create_retriever()`)
- Embedding model: BAAI/bge-base-en-v1.5

**Generation** (`src/bulk_testing.py`):
- Model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
- Temperature: 0.0 (deterministic)
- Max tokens: 512

## Output Files

Results saved to `bulk_runs/` with format:
```
{timestamp}_{dataset}_{model}_k{top_k}_t{temp}.csv
{timestamp}_{dataset}_{model}_k{top_k}_t{temp}.json
```

Example: `2025-11-09_14-30-22_fb_llama31-70b_k5_t0.csv`

**CSV columns:**
- question_id, question, gold_answer, predicted_answer
- semantic_similarity (0.0-1.0)
- retrieval_time_ms, generation_time_ms
- sources (retrieved document names)
- question_type, company, doc_name (metadata)
- error (if prediction failed)

**JSON contains:**
- Overall statistics (avg/min/max similarity)
- Per-question-type breakdown
- Success/failure counts
- Average timing
- Configuration used

## Current Performance

**FinanceBench baseline (150 questions):**
- Average Semantic Similarity: 53.7%
- Success Rate: 100%
- Main issue: Cross-document contamination (retrieves wrong company/year)

## Improvement Suggestions

**Quick Changes That Can Be Done:**
1. Increase top-k (test k=10, 15, 20)
2. Adjust temperature and max_tokens

**Medium effort:**
3. Hybrid search (BM25 + semantic)
4. Reranking with cross-encoder
5. Query enhancement (extract entities, expand terms)

**Requires DB rebuild:**
6. Better chunking (larger chunks, semantic boundaries)

## Troubleshooting

**Database not found:**
- Run `python src/create_database.py` first

**Module not found errors:**
- Run scripts from project root: `cd /path/to/myRAG`

**Rate limit errors:**
- System saves partial results automatically
- Check Together API quota

**Low similarity scores:**
- Verify PDFs match dataset expectations
- Try increasing top-k
- Check retrieved sources in CSV output

**Import errors in moved files:**
- All imports updated to work from new structure
- If issues persist, check sys.path modifications in file headers


**Recommended experiments:**
1. Baseline: k=5, semantic search only
2. Vary k: Test k=10, 15, 20
3. Compare improvements: hybrid search, reranking
4. Test on multiple datasets for generalization
5. Implementing different retrieval methods as tools

**Metrics to report:**
- Semantic similarity (primary)
- Success rate
- Per-question-type breakdown
- Timing statistics
