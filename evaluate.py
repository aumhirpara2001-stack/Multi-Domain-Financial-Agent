import re
import numpy as np
from sklearn.metrics import f1_score
from data_loader import load_financebench, load_finder
from embedder import build_vector_store
from retriever import get_retriever
from rag_chain import build_rag_chain

def normalize(text):
    return re.sub(r'\W+', ' ', text.lower()).strip()

def exact_match(pred, gold):
    return normalize(pred) == normalize(gold)

def token_f1(pred, gold):
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def is_multi_hop_or_temporal(question):
    keywords = ["compare", "difference", "trend", "change", "growth", "decline", "between", "across", "over time", "historical"]
    return any(kw in question.lower() for kw in keywords)

def evaluate_model(chain, test_df, max_samples=100):
    em_scores, f1_scores, hallucinations = [], [], []
    complex_em_scores, complex_f1_scores = [], []

    for i, row in test_df.iterrows():
        if i >= max_samples:
            break
        query = row["question"]
        gold = row["answer"]
        result = chain({"query": query})
        pred = result["result"]

        em = exact_match(pred, gold)
        f1 = token_f1(pred, gold)

        em_scores.append(em)
        f1_scores.append(f1)

        # Chunk-aware hallucination check
        retrieved_chunks = [normalize(doc.page_content[:300]) for doc in result["source_documents"]]
        hallucinated = not any(normalize(gold) in chunk for chunk in retrieved_chunks)
        hallucinations.append(hallucinated)

        # Multi-hop/temporal tagging
        is_complex = is_multi_hop_or_temporal(query)
        if is_complex:
            complex_em_scores.append(em)
            complex_f1_scores.append(f1)

        # Print trace
        print(f"\nQ: {query}")
        print(f"PRED: {pred}")
        print(f"GOLD: {gold}")
        print(f"EM: {em}, F1: {f1:.2f}, Hallucinated: {hallucinated}, Multi-hop/Temporal: {is_complex}")
        print("Retrieved Chunks:")
        for idx, chunk in enumerate(retrieved_chunks):
            print(f"  Chunk {idx+1}: {chunk[:200]}...")

    # Summary
    print("\n--- Evaluation Summary ---")
    print(f"Exact Match: {np.mean(em_scores):.2f}")
    print(f"F1 Score: {np.mean(f1_scores):.2f}")
    print(f"Hallucination Rate: {np.mean(hallucinations):.2f}")
    print(f"Multi-hop EM: {np.mean(complex_em_scores):.2f}")
    print(f"Multi-hop F1: {np.mean(complex_f1_scores):.2f}")

if __name__ == "__main__":
    print("Choose dataset for evaluation:")
    print("1. FinanceBench")
    print("2. FinDER")
    choice = input("Enter 1 or 2: ")

    if choice == "1":
        test_df = load_financebench("test")
    elif choice == "2":
        test_df = load_finder("test")
    else:
        print("Invalid choice. Defaulting to FinanceBench.")
        test_df = load_financebench("test")

    vectorstore = build_vector_store(test_df)
    retriever = get_retriever(vectorstore)
    rag_chain = build_rag_chain(retriever)
    evaluate_model(rag_chain, test_df)
