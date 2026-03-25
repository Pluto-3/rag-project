import ollama
import chromadb
import time
from sentence_transformers import SentenceTransformer

# --- Setup ---
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="documents")


# --- Query Rewriting ---
def rewrite_query(question):
    prompt = f"""You are a search query optimizer. 
Rewrite the following question into a concise search query that will retrieve relevant information from a formal document.
Return ONLY the rewritten query, nothing else. No explanation, no preamble.

Question: {question}
Rewritten query:"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 1024}
    )

    return response["message"]["content"].strip()


# --- Retrieval ---
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


# --- Answer Generation ---
def ask(question):
    # Step 1 — rewrite the query
    rewritten = rewrite_query(question)
    print(f"\n[Query rewritten]: {rewritten}")
    time.sleep(2)

    # Step 2 — retrieve using rewritten query
    retrieved = retrieve(rewritten)

    # Step 3 — build context
    context = "\n\n".join([chunk for chunk, _, _ in retrieved])

    # Step 4 — generate answer
    prompt = f"""You are a helpful assistant. Use the context below to answer the question.
If the context doesn't contain enough information, say so honestly rather than making things up.

Context:
{context}

Question:
{question}

Answer:"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 2048}
    )

    answer = response["message"]["content"]

    return answer, rewritten, retrieved


# --- Main loop ---
if __name__ == "__main__":
    print("RAG system ready. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()

        if question.lower() == "quit":
            break

        if not question:
            continue

        answer, rewritten, sources = ask(question)

        print(f"\n--- Answer ---")
        print(answer)

        print(f"\n--- Sources used ---")
        for i, (chunk, distance, meta) in enumerate(sources):
            print(f"[{i+1}] source: {meta['source']} | chunk: {meta['chunk_index']} | distance: {distance:.4f}")
            print(f"    {chunk[:150]}...")

        print()