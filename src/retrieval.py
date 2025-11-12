# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_classic.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_together import TogetherEmbeddings
from langchain.prompts import PromptTemplate
from langchain_together import Together
from langchain.chains import LLMChain
import openai
import os
from pathlib import Path

load_dotenv()

# path to chromadb persistence directory (relative to project root)
BASE_DIR = Path(__file__).parent.parent
CHROMA_PATH = str(BASE_DIR / "chroma")
embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5") # this is for embedding with TogetherAPI rather than OpenAI

def main():


    # get user query
    user_query = input("Enter your question: ")

    # retrieve relevant chunks
    retriever = create_retriever()
    docs = retriever.invoke(user_query) 

    # if no relevant documents found, print message and return
    if not docs:
        print("No relevant documents found.")
        return

    # combine retrieved text into context (combine chunks into one context block)
    context = "\n\n".join(d.page_content for d in docs)

    # initialize Together client (uses OpenAI-compatible SDK) - from Together Docs
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
    )

    # generate grounded answer using context + question
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",  # current answer gen model
        messages=[
            {"role": "system", "content": "You are a helpful research assistant."},
            {
                "role": "user",
                "content": f"Use the following context to answer the question concisely and accurately.\n\nContext:\n{context}\n\nQuestion:\n{user_query}",
            },
        ],
        max_tokens=512,
        temperature=0,
    )

    # display result
    print("\nAnswer: \n", response.choices[0].message.content)
    print("\nSources:\n")
    for doc in docs:
        print(doc.metadata)


def load_database():
    # load existing chromadb instance from persistent directory using openai embeddings
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )
    return db


def create_retriever():
    # create retriever from loaded database for querying documents
    db = load_database()
    retriever = db.as_retriever()
    return retriever

if __name__ == "__main__":
    main()