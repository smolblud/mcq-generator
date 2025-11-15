import faiss
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import re

# --------------------------
# PATHS & MODEL
# --------------------------
INDEX_PATH = "index/faiss_index.bin"       # FAISS index path
METADATA_PATH = "index/metadata.json"      # Metadata path
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # stronger embeddings

# --------------------------
# LOAD INDEX & METADATA
# --------------------------
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# --------------------------
# LOAD EMBEDDING MODEL
# --------------------------
model = SentenceTransformer(MODEL_NAME)

# --------------------------
# QUERY FUNCTION
# --------------------------
def query_index(query_text, top_k=3, subject_filter=None):
    """
    Return top_k most relevant chunks for a query.
    Optional: filter by subject ('math', 'physics', etc.)
    """
    query_vec = model.encode([query_text], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k * 5)  # search more for filtering

    results = []
    for i, dist in zip(indices[0], distances[0]):
        chunk_meta = metadata[i]

        # Apply subject filter if given
        if subject_filter and chunk_meta["subject"].lower() != subject_filter.lower():
            continue

        results.append({
            "chunk_id": chunk_meta["id"],
            "subject": chunk_meta["subject"],
            "topic": chunk_meta["topic"],
            "source_file": chunk_meta["source_file"],
            "distance": float(dist),
            "text_preview": chunk_meta.get("text", "")[:400]  # show first 400 chars
        })

        if len(results) >= top_k:
            break

    return results

# --------------------------
# MAIN: DYNAMIC QUERY
# --------------------------
if __name__ == "__main__":
    query_text = input("Enter your query: ")
    subject_filter = input("Enter subject filter (leave blank for all subjects): ").strip() or None
    top_k = 3

    results = query_index(query_text, top_k=top_k, subject_filter=subject_filter)

    print(f"\nTop {len(results)} results for query: '{query_text}'\n")
    for i, chunk in enumerate(results, start=1):
        print(f"Result #{i}")
        print(f"Chunk ID   : {chunk['chunk_id']}")
        print(f"Subject    : {chunk['subject']}")
        print(f"Topic      : {chunk['topic']}")
        print(f"Source File: {chunk['source_file']}")
        print(f"Distance   : {chunk['distance']:.4f}")
        print(f"Text Preview:\n{chunk['text_preview']}...")
        print("-" * 80)
