import chromadb
import os
from sentence_transformers import SentenceTransformer
from ingest import extract_text
from chunker import chunk_text


model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")


def embed_document(pdf_path):
    filename = os.path.basename(pdf_path)

    existing = collection.get(where={"source": filename})
    if existing["ids"]:
        print(f"'{filename}' already embedded ({len(existing['ids'])} chunks). Skipping.")
        return

    print(f"Embedding '{filename}'...")
    text = extract_text(pdf_path)
    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        chunk_id = f"{filename}_chunk_{i}"

        collection.add(
            ids=[chunk_id],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{
                "source": filename,
                "chunk_index": i
            }]
        )

    print(f"Stored {len(chunks)} chunks from '{filename}'")


if __name__ == "__main__":
    pdf_files = [f for f in os.listdir(".") if f.endswith(".pdf")]

    if not pdf_files:
        print("No PDF files found in current directory.")
    else:
        print(f"Found {len(pdf_files)} PDF(s): {pdf_files}\n")
        for pdf in pdf_files:
            embed_document(pdf)

    print(f"\nTotal chunks in database: {collection.count()}")