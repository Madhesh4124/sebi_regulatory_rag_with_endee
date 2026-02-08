import json
import math
from sentence_transformers import SentenceTransformer
from typing import List, Dict

# =====================================================
# CONFIG
# =====================================================
CHUNKS_FILE = "chunks.json"
OUTPUT_FILE = "chunks_with_embeddings.json"  

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

BATCH_SIZE = 32
NORMALIZE_EMBEDDINGS = True

# =====================================================
# 1. LOAD CHUNKS
# =====================================================
def load_chunks(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert isinstance(data, list), "chunks.json must be a list"
    assert "text" in data[0], "Each chunk must have a 'text' field"
    assert "metadata" in data[0], "Each chunk must have 'metadata'"
    assert data[0]["text"].strip(), "Chunk text is empty"

    return data

# =====================================================
# 2. LOAD MODEL
# =====================================================
def load_model(name: str) -> SentenceTransformer:
    print(f"Loading embedding model: {name}")
    return SentenceTransformer(name)

# =====================================================
# 3. BATCHED EMBEDDING (RAM SAFE)
# =====================================================
def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    normalize: bool
):
    embeddings = []

    total = len(texts)
    batches = math.ceil(total / batch_size)

    for i in range(batches):
        start = i * batch_size
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]

        print(f"Embedding batch {i+1}/{batches} ({start}–{end})")

        vecs = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=normalize
        )

        embeddings.extend(vecs)

    return embeddings

# =====================================================
# 4. MAIN PIPELINE
# =====================================================
def main():
    chunks = load_chunks(CHUNKS_FILE)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    model = load_model(MODEL_NAME)

    vectors = embed_texts(
        model=model,
        texts=texts,
        batch_size=BATCH_SIZE,
        normalize=NORMALIZE_EMBEDDINGS
    )

    assert len(vectors) == len(chunks), "Embedding count mismatch"

    output = []
    for i in range(len(chunks)):
        output.append({
            "text": chunks[i]["text"],        
            "metadata": metadatas[i],
            "embedding": vectors[i].tolist()
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nEmbedding complete")
    print(f"Chunks embedded: {len(output)}")
    print(f"Vector dimension: {len(output[0]['embedding'])}")
    print(f"Output written to: {OUTPUT_FILE}")

    # Sanity check
    print("\n--- SAMPLE CHECK ---")
    print("Clause:", output[0]["metadata"]["clause_id"])
    print("Text length:", len(output[0]["text"]))
    print("Vector (first 10 dims):", output[0]["embedding"][:10])

# =====================================================
# 5. RUN
# =====================================================
if __name__ == "__main__":
    main()
