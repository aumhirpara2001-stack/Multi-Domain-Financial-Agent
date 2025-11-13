import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.utils.etl import etl_auto
from langchain_core.documents import Document

RAW_PATH = "./data/raw/all_questions_tagged.csv"
CLEAN_PATH = "./data/processed/clean_questions.csv"

def to_langchain_documents(df: pd.DataFrame):
    documents = []
    for _, row in df.iterrows():
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        context = str(row.get("context", "")).strip()

        if not question or not answer:
            continue

        page_content = f"Question: {question}\nAnswer: {answer}\nContext: {context}"
        metadata = {
            "id": row.get("id", ""),
            "source_dataset": row.get("source_dataset", ""),
            "primary_tag": row.get("primary_tag", ""),
            "secondary_tags": row.get("secondary_tags", "")
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    return documents

if __name__ == "__main__":
    df_raw = pd.read_csv(RAW_PATH)
    df_clean, _ = etl_auto(df_raw)
    df_clean.to_csv(CLEAN_PATH, index=False)

    docs = to_langchain_documents(df_clean)
    print(f"✅ Ingested and filtered {len(docs)} clean documents.")
