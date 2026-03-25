import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")

def retrieve(query, k=3):
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(k, collection.count())
    )

    chunks = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    return list(zip(chunks, distances, metadatas))


if __name__ == "__main__":
    query = "What is this document about?"

    print(f"Query: {query}\n")
    results = retrieve(query)

    for i, (chunk, distance) in enumerate(results):
        print(f"--- Result {i+1} | chunk_index: {meta['chunk_index']} | (distance: {distance:.4f}) ---")
        print(chunk[:300])
        print()