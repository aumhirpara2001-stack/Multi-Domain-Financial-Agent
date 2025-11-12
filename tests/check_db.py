from langchain_chroma import Chroma
from langchain_together import TogetherEmbeddings
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# Path relative to project root
BASE_DIR = Path(__file__).parent.parent
CHROMA_PATH = str(BASE_DIR / "chroma")
embeddings = TogetherEmbeddings(model="BAAI/bge-base-en-v1.5")

# Load existing database
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

# Get collection count
collection = db._collection
count = collection.count()

print(f"\n{'='*60}")
print(f"Current chunks in ChromaDB: {count:,} / 255556")
print(f"Progress: {(count/255556)*100:.2f}%")
print(f"{'='*60}\n")
