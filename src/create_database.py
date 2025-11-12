from langchain_community.document_loaders import DirectoryLoader, UnstructuredPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_together import TogetherEmbeddings
# from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import openai
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
import shutil
from pathlib import Path

# load environment variables from .env file
load_dotenv()
# set openai api key from environment variablef
openai.api_key = os.environ['OPENAI_API_KEY']

# path to chromadb persistence directory (relative to project root)
BASE_DIR = Path(__file__).parent.parent
CHROMA_PATH = str(BASE_DIR / "chroma")
# path to directory containing source documents
DATA_PATH = str(BASE_DIR / "data/test_files/finance-bench-pdfs")

embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")

def main():
    # main entry point that orchestrates the data store generation
    generate_data_store()


def generate_data_store():
    # load documents from source directory, split into chunks, and save to vector database
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    # load all pdf files from the data directory and return as document objects
    # with error handling and progress tracking
    pdf_files = list(Path(DATA_PATH).glob("*.pdf"))
    total_files = len(pdf_files)
    documents = []
    failed_files = []

    print(f"\n{'='*60}")
    print(f"Loading {total_files} PDF files from {DATA_PATH}")
    print(f"{'='*60}\n")

    for idx, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"[{idx}/{total_files}] Processing: {pdf_path.name}...", end=" ")
            loader = UnstructuredPDFLoader(str(pdf_path))
            docs = loader.load()
            documents.extend(docs)
            print(f"✓ ({len(docs)} pages)")
        except Exception as e:
            print(f"✗ FAILED")
            failed_files.append({
                'file': pdf_path.name,
                'error': str(e)
            })
            continue

    print(f"\n{'='*60}")
    print(f"Successfully loaded: {len(documents)} pages from {total_files - len(failed_files)} files")
    if failed_files:
        print(f"Failed files: {len(failed_files)}")
        print(f"{'='*60}")
        print("\nFailed files details:")
        for fail in failed_files:
            print(f"  - {fail['file']}")
            print(f"    Error: {fail['error'][:100]}...")
    print(f"{'='*60}\n")

    return documents


def split_text(documents: list[Document]):
    # split documents into smaller chunks with overlap for better context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # maximum characters per chunk (increased for better context)
        chunk_overlap=200,  # characters to overlap between chunks
        length_function=len,  # function to measure text length
        add_start_index=True,  # add start index to metadata for tracking position in source
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # debug output to show sample chunk and its metadata
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # remove existing database directory if it exists to start fresh
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # create new vector database with batched processing for better performance and progress tracking
    print(f"\n{'='*60}")
    print(f"Embedding and saving {len(chunks)} chunks to ChromaDB")
    print(f"{'='*60}\n")

    # Process in batches to show progress and avoid memory issues
    batch_size = 500  # chunks per batch
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    db = None
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = (i // batch_size) + 1

        print(f"[Batch {batch_num}/{total_batches}] Embedding {len(batch)} chunks...", end=" ", flush=True)

        try:
            if db is None:
                # Create database with first batch
                db = Chroma.from_documents(
                    batch, embeddings, persist_directory=CHROMA_PATH
                )
            else:
                # Add subsequent batches to existing database
                db.add_documents(batch)
            print(f"✓ ({i + len(batch)}/{len(chunks)} total)")
        except Exception as e:
            print(f"✗ FAILED: {str(e)[:100]}")
            continue

    print(f"\n{'='*60}")
    print(f"✓ Saved {len(chunks)} chunks to {CHROMA_PATH}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()