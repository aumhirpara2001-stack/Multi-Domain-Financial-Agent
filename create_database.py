from langchain_community.document_loaders import DirectoryLoader
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

# load environment variables from .env file
load_dotenv()
# set openai api key from environment variable
openai.api_key = os.environ['OPENAI_API_KEY']

# path to chromadb persistence directory
CHROMA_PATH = "chroma"
# path to directory containing source documents
DATA_PATH = "data/test_files"

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
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    # split documents into smaller chunks with overlap for better context preservation
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # maximum characters per chunk
        chunk_overlap=100,  # characters to overlap between chunks
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

    # create new vector database from chunks using TogetherAPI embeddings and persist to disk
    db = Chroma.from_documents(
        chunks, embeddings, persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()