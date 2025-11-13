# Migration Guide: Project Reorganization

This document explains the changes made to reorganize the PennyBot codebase for better maintainability and readability.

## Summary of Changes

### 1. **Directory Restructuring**

The project has been reorganized into a cleaner structure:

**Before:**
```
PennyBot_LLM_Agentic_RAG/
├── all_questions_tagged.csv
├── build_index.py
├── chat_cli.py
├── etl.py
├── financebench_open_source.jsonl
├── generate_corpus.py
├── ingest_and_filter.py
├── rag_agent_library.py
└── requirements.txt (with merge conflict)
```

**After:**
```
PennyBot_LLM_Agentic_RAG/
├── src/                          # Source code
│   ├── rag_agent_library.py
│   ├── chat_cli.py
│   └── utils/
│       └── etl.py
├── scripts/                      # Utility scripts
│   ├── build_index.py
│   ├── evaluate.py (NEW)
│   ├── generate_corpus.py
│   └── ingest_and_filter.py
├── data/                         # Data files
│   ├── raw/
│   │   ├── all_questions_tagged.csv
│   │   └── financebench_open_source.jsonl
│   └── processed/
├── config/                       # Configuration
│   └── .env.example (NEW)
├── docs/                         # Documentation
│   └── MIGRATION.md (this file)
└── tests/                        # Tests (empty, for future use)
```

### 2. **Critical Bug Fixes**

#### Duplicate Code Removed
- **File:** `src/rag_agent_library.py`
- **Issue:** Functions `_format_chat_history`, `_format_docs_with_citations`, and `create_rag_pipeline` were defined twice (lines 270-340 were incomplete duplicates)
- **Fix:** Removed duplicate definitions, kept the complete implementation

#### Merge Conflict Resolved
- **File:** `requirements.txt`
- **Issue:** Unresolved git merge conflict (lines 1-29)
- **Fix:** Created clean requirements.txt combining both branches with proper versioning

#### Missing File Created
- **File:** `scripts/evaluate.py`
- **Issue:** Referenced by Dockerfile but didn't exist
- **Fix:** Created comprehensive evaluation harness with EM, F1, TTFT, and hallucination detection

### 3. **Import Path Updates**

All import statements have been updated to work with the new structure:

**src/chat_cli.py:**
```python
# Before
from rag_agent_library import ...

# After
from src.rag_agent_library import ...
```

**scripts/ingest_and_filter.py:**
```python
# Before
from etl import etl_auto

# After
from src.utils.etl import etl_auto
```

**scripts/build_index.py:**
- Completely rewritten to use the new structure
- Now uses `get_pinecone_vectorstore` from rag_agent_library

### 4. **New Files Added**

| File | Purpose |
|------|---------|
| `config/.env.example` | Template for environment variables with API keys |
| `scripts/evaluate.py` | Comprehensive evaluation harness |
| `run_all.sh` | Cross-platform setup script (replaces run_all.bat) |
| `docs/MIGRATION.md` | This migration guide |
| `src/__init__.py` | Python package marker |
| `src/utils/__init__.py` | Python package marker |
| `tests/__init__.py` | Python package marker |

### 5. **Configuration Improvements**

#### Updated .gitignore
Added patterns for:
- Processed data files
- Evaluation results
- Additional IDE files
- Test coverage reports

#### Enhanced .dockerignore
Improved Docker build efficiency by excluding unnecessary files.

#### Updated Dockerfile
- Better layer caching (copy requirements.txt first)
- Updated CMD to point to `scripts/evaluate.py`
- Added comments for clarity

### 6. **Documentation Updates**

#### New README.md
The README has been completely rewritten with:
- Clear project structure diagram
- Step-by-step quick start guide
- Mathematical foundations explained
- Cost analysis with estimates
- Usage examples (CLI and programmatic)
- Docker deployment instructions
- Evaluation metrics documentation

#### .env.example
Comprehensive template with:
- Required API keys (Together AI, Pinecone)
- Optional configuration options
- Model selection guide
- Comments explaining each variable

## Migration Checklist

If you have an existing installation, follow these steps:

### ☑️ Step 1: Backup Your Work
```bash
# Create a backup
cp -r PennyBot_LLM_Agentic_RAG PennyBot_LLM_Agentic_RAG.backup
```

### ☑️ Step 2: Pull Latest Changes
```bash
git pull origin main
```

### ☑️ Step 3: Update Your .env File
```bash
# Copy the new template
cp config/.env.example .env

# Migrate your existing API keys to the new .env format
```

### ☑️ Step 4: Reinstall Dependencies
```bash
# Activate your virtual environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Reinstall with the updated requirements.txt
pip install --upgrade pip
pip install -r requirements.txt
```

### ☑️ Step 5: Update Import Statements (if you have custom code)
If you've written custom scripts that import from PennyBot:

```python
# Old imports
from rag_agent_library import ...
from etl import ...

# New imports
from src.rag_agent_library import ...
from src.utils.etl import ...
```

### ☑️ Step 6: Test the Installation
```bash
# Verify Python syntax
python -m py_compile src/*.py scripts/*.py src/utils/*.py

# Run the setup script
./run_all.sh  # On Unix/Mac
# or
bash run_all.sh  # On Windows with Git Bash
```

### ☑️ Step 7: Run Evaluation (Optional)
```bash
# Test with a small subset
python scripts/evaluate.py --limit 10

# Full evaluation
python scripts/evaluate.py
```

## Breaking Changes

### Command Changes

| Old Command | New Command |
|-------------|-------------|
| `python chat_cli.py` | `python src/chat_cli.py` |
| `python build_index.py` | `python scripts/build_index.py` |
| `python ingest_and_filter.py` | `python scripts/ingest_and_filter.py` |
| N/A | `python scripts/evaluate.py` (NEW) |

### Data File Locations

| Old Path | New Path |
|----------|----------|
| `./all_questions_tagged.csv` | `data/raw/all_questions_tagged.csv` |
| `./financebench_open_source.jsonl` | `data/raw/financebench_open_source.jsonl` |
| N/A | `data/processed/` (for cleaned data) |

### Configuration

| Old Path | New Path |
|----------|----------|
| `./.env` | `./.env` (unchanged, but see config/.env.example) |
| N/A | `config/.env.example` (NEW template) |

## Benefits of the Reorganization

### 1. **Better Separation of Concerns**
- Source code in `src/`
- Utility scripts in `scripts/`
- Data in `data/`
- Config in `config/`

### 2. **Improved Maintainability**
- Easier to locate files
- Clear project structure
- Reduced code duplication

### 3. **Enhanced Developer Experience**
- Comprehensive README with examples
- .env.example template for easy setup
- Migration guide (this document)

### 4. **Better Testing Infrastructure**
- `tests/` directory prepared for unit tests
- Evaluation harness for benchmarking

### 5. **Production Readiness**
- Clean requirements.txt
- Updated Dockerfile
- Cross-platform run script

## Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:** Make sure you're running scripts from the project root:
```bash
cd /path/to/PennyBot_LLM_Agentic_RAG
python src/chat_cli.py  # ✅ Correct
```

### Data Not Found

**Problem:** `FileNotFoundError: data/raw/all_questions_tagged.csv`

**Solution:** Verify data files are in the correct location:
```bash
ls -la data/raw/
# Should show: all_questions_tagged.csv and financebench_open_source.jsonl
```

### Environment Variables

**Problem:** API key errors

**Solution:** Verify your .env file is in the project root with correct keys:
```bash
cat .env
# Should contain:
# TOGETHER_API_KEY=...
# PINECONE_API_KEY=...
```

## Questions or Issues?

If you encounter any problems during migration:

1. Check this migration guide
2. Review the updated README.md
3. Open an issue on GitHub with details about the error

## Version History

- **v2.0.0** (2025-01-13) - Major reorganization
  - Restructured project directories
  - Fixed critical bugs (duplicates, merge conflicts)
  - Added evaluation harness
  - Updated all documentation

- **v1.0.0** (Previous) - Initial release
  - Basic RAG functionality
  - CLI interface
  - Pinecone integration
