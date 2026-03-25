import chromadb
from sentence_transformers import SentenceTransformer
from ingest import extract_text
from chunker import chunk_text

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

text = extract_text("sample.pdf")
chunks = chunk_text(text)
print(f"Chunks to embed: {len(chunks)}")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="documents")

print("Embedding and storing chunks...")
for i, chunk in enumerate(chunks):
    embedding = model.encode(chunk).tolist()

    collection.add(
        ids=[f"chunk_{i}"],
        embeddings=[embedding],
        documents=[chunk],
        metadatas=[{"chunk_index": i}]
    )

print(f"\nStored {collection.count()} chunks in ChromaDB")
print("Vector database saved to ./chroma_db")
