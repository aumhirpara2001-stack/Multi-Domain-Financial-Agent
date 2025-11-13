import pandas as pd
from ingest_and_filter import to_langchain_documents
from langchain_community.vectorstores import Pinecone
from langchain.embeddings import HuggingFaceEmbeddings

df = pd.read_csv("./data/processed/clean_questions.csv")
docs = to_langchain_documents(df)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Pinecone.from_documents(docs, embeddings, index_name="your_index_name")

print(f"✅ Pinecone index populated with {len(docs)} documents.")
