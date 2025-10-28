from datasets import load_dataset
import pandas as pd

def load_financebench(split="train"):
    dataset = load_dataset("PatronusAI/financebench", split=split)
    df = pd.DataFrame(dataset)

    # Preview the actual structure
    print("📊 Columns in FinanceBench:", df.columns.tolist())
    print("🧪 Sample rows:\n", df[["question", "answer", "justification"]].head(5).to_string())

    # Use 'justification' instead of 'context'
    return df[["question", "justification", "answer"]].rename(columns={"justification": "context"})

def load_finder(split="train"):
    dataset = load_dataset("Linq-AI-Research/FinDER", split=split)
    df = pd.DataFrame(dataset)

    # Preview FinDER structure
    print("📊 Columns in FinDER:", df.columns.tolist())
    print("🧪 Sample rows:\n", df[["question", "context", "answer"]].head(5).to_string())

    return df[["question", "context", "answer"]]