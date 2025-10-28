# app.py
"""
CLI RAG agent.
Flow:
  Text -> Splitter -> Embeddings -> FAISS VectorStore -> Retriever -> LLM -> Answer
"""

import os
from dotenv import load_dotenv
import pandas as pd

from embedder import build_vector_store

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate    

# Load environment variables
load_dotenv()


def load_sample_data() -> pd.DataFrame:
    """Load a small sample dataset with a 'text' column."""
    try:
        from datasets import load_dataset
        ds = load_dataset("ag_news", split="train[:1%]")
        texts = []
        for ex in ds:
            if "title" in ex and "description" in ex:
                texts.append(f"{ex['title']}\n\n{ex['description']}")
            else:
                texts.append(" ".join(str(v) for v in ex.values()))
        return pd.DataFrame({"text": texts})
    except Exception:
        return pd.DataFrame(
            {
                "text": [
                    "Python is a programming language emphasizing readability.",
                    "LangChain helps build LLM-powered applications with retrievers and chains.",
                    "FAISS enables efficient similarity search over dense vectors.",
                    "OpenAI provides embeddings and chat models like gpt-3.5-turbo and gpt-4.",
                ]
            }
        )


def build_retriever(vectorstore):
    """Wrap FAISS vectorstore in a retriever."""
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def build_rag_chain(retriever):
    """Construct RetrievalQA chain with ChatOpenAI and a custom prompt."""
    model_name = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
    llm = ChatOpenAI(model=model_name, temperature=0)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are a helpful assistant. Use the provided context to answer the question.\n\n"
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer clearly and concisely."
        ),
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def main():
    # 1. Load dataset
    df = load_sample_data()

    # 2. Build vector store
    print("Building vector store...")
    vectorstore = build_vector_store(df)

    # 3. Build retriever + RAG chain
    retriever = build_retriever(vectorstore)
    qa_chain = build_rag_chain(retriever)

    # 4. Interactive CLI loop
    print("RAG agent ready. Type 'exit' to quit.")
    while True:
        query = input("\nAsk a question: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("👋 Exiting. See you next time!")
            break

        result = qa_chain({"query": query})
        print("\nAnswer:", result["result"])
        print("Sources:", [doc.metadata for doc in result["source_documents"]])


if __name__ == "__main__":
    main()