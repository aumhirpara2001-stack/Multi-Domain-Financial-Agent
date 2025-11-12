# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path

# load environment variables from .env file
load_dotenv()

# get the directory where this script is located and project root
BASE_DIR = Path(__file__).parent.parent
# construct absolute path to chroma database directory
CHROMA_PATH = str(BASE_DIR / "chroma")


def main():
    # main entry point that displays database information
    view_database_info()


def view_database_info():
    # load database, retrieve all documents, and print comprehensive statistics
    db = load_database()
    collection = db._collection
    results = get_all_documents(collection)
    
    print_database_stats(collection)
    print_source_files(results)
    print_sample_documents(results)


def load_database():
    # load existing chromadb instance from persistent directory using openai embeddings
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings()
    )
    return db


def get_all_documents(collection):
    # retrieve all documents from the collection including ids, documents, and metadata
    results = collection.get()
    return results


def print_database_stats(collection):
    # print basic statistics about the database collection
    count = collection.count()
    print(f"Total documents: {count}")
    print(f"Collection name: {collection.name}")
    
    # print collection-level metadata if it exists
    if collection.metadata:
        print(f"Collection metadata: {collection.metadata}")
    print()


def print_source_files(results):
    # extract and display unique source file paths from document metadata
    if not results['metadatas']:
        return
    
    # collect unique source file paths from all document metadata
    sources = set()
    for metadata in results['metadatas']:
        if 'source' in metadata:
            sources.add(metadata['source'])
    
    print(f"Unique source files: {len(sources)}")
    for source in sorted(sources):
        print(f"  - {source}")
    print()


def print_sample_documents(results, num_samples=3):
    # display sample documents with their content preview and metadata
    count = len(results['ids'])
    if count == 0:
        print("No documents found in database.")
        return
    
    print(f"Sample documents (showing first {min(num_samples, count)}):")
    print("-" * 60)
    
    # iterate through sample documents and print their details
    for i in range(min(num_samples, count)):
        print(f"\nDocument {i+1} (ID: {results['ids'][i][:30]}...)")
        
        # print content preview (first 200 characters) if available
        if results['documents']:
            content = results['documents'][i]
            preview_length = 200
            if len(content) > preview_length:
                print(f"Content preview: {content[:preview_length]}...")
            else:
                print(f"Content: {content}")
        
        # print document metadata including source file and start index
        if results['metadatas']:
            print(f"Metadata: {results['metadatas'][i]}")
        
        print("-" * 60)


if __name__ == "__main__":
    main()