def chunk_text(text, chunk_size=400, overlap=75):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    from ingest import extract_text

    text = extract_text("sample.pdf")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")
    print(f"\n--- Chunk 1 ---")
    print(chunks[0])
    print(f"\n--- Chunk 2 ---")
    print(chunks[1])
    print(f"\n--- Overlap check (last 10 words of chunk 1) ---")
    print(" ".join(chunks[0].split()[-10:]))
    print(f"\n--- (first 10 words of chunk 2) ---")
    print(" ".join(chunks[1].split()[:10]))