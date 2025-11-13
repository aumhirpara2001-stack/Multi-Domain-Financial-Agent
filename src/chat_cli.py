
# Agentic RAG Chat CLI  

"""
chat_cli.py - The Interactive Chat Agent

This is the file you will run from your terminal.
It imports the "brain" and "memory" functions from
our `rag_agent_library.py` file and creates the
interactive chat in the CLI.
"""

import sys
from pathlib import Path
from typing import List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the core components from our "brain" library
from src.rag_agent_library import (
    get_pinecone_vectorstore,
    create_rag_pipeline,
    LLM_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    PINECONE_INDEX_NAME
)
from langchain_together import ChatTogether, TogetherEmbeddings


def main():
    """
    Main function to initialize and run the chat agent.
    """
    print("--- Initializing Conversational RAG Agent ---")

    # 1. Initialize Models
    print(f"Using LLM: {LLM_MODEL_NAME}")
    llm = ChatTogether(
        model=LLM_MODEL_NAME,
        temperature=0.1
    )

    print(f"Using Embeddings: {EMBEDDING_MODEL_NAME}")
    embeddings = TogetherEmbeddings(model=EMBEDDING_MODEL_NAME)

    # 2. Get the Vector Store ("Memory")
    # This will connect to Pinecone and, if it's the very first run,
    # it will automatically load and seed your 10,000+ document chunks.
    # On all future runs, it will just connect instantly.
    # Set force_reseed=False for production use.
    # Set force_reseed=True to delete and re-upload everything.
    vectorstore = get_pinecone_vectorstore(embeddings, force_reseed=False)

    if vectorstore is None:
        print("Failed to initialize vector store. Exiting.")
        return

    print(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")

    # 3. Create the RAG Pipeline ("Brain")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3} # Retrieve top 3 chunks
    )
    rag_agent = create_rag_pipeline(retriever, llm)

    print("-------------------------------------------------")
    print("✅ Agent is ready! Type 'exit' or 'quit' to end.")
    print("-------------------------------------------------")

    # 4. Run the Interactive Chat Loop
    chat_history: List[Tuple[str, str]] = []

    while True:
        try:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print("\nAI: Goodbye!")
                break

            if not query:
                continue

            # Run the full RAG pipeline
            print("\nAI: Thinking...")
            response = rag_agent.invoke({
                "question": query,
                "chat_history": chat_history
            })

            # Print the answer
            print(f"\nAI: {response['answer']}")

            # Print the citations (optional)
            citations = set()
            for doc in response['retrieved_docs']:
                citations.add(f"Source: {doc.metadata.get('source_dataset', 'N/A')}, ID: {doc.metadata.get('id', 'N/A')}")

            if citations:
                print("\n--- Citations ---")
                for c in citations:
                    print(c)
            print("-----------------")

            # Update chat history
            chat_history.append((query, response['answer']))

        except KeyboardInterrupt:
            print("\nAI: Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()