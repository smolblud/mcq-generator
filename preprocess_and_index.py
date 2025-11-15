import os
import re
import json
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------- CONFIG ----------------
DATA_DIR = Path("data")  # Folder containing txt files
CHUNK_SIZE = 300         # smaller chunk size for better accuracy
CHUNK_OVERLAP = 50       # overlap between chunks
INDEX_DIR = Path("index")  # where FAISS index & metadata will be saved
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # better embeddings
TOPIC_BLUEPRINT_CSV = Path(r"C:\Users\User\OneDrive\Desktop\mcq-generator-main\mcq-generator\final_topic_blueprint.csv")
# ----------------------------------------

def load_topic_blueprint(csv_path):
    df = pd.read_csv(csv_path)
    df["start_page"] = pd.to_numeric(df["start_page"], errors="coerce").fillna(0).astype(int)
    df["end_page"] = pd.to_numeric(df["end_page"], errors="coerce").fillna(0).astype(int)
    return df

def assign_topic_by_page(subject_blueprint, page_num):
    """Return the topic that contains the given page number."""
    for _, row in subject_blueprint.iterrows():
        if row["start_page"] <= page_num <= row["end_page"]:
            return row["topic"]
    return "General"

def chunk_text_by_pages(text, subject, filename, blueprint_df):
    """Split text into pages and assign topics based on page numbers."""
    pages = re.split(r'--- PAGE (\d+) ---', text)
    chunks = []
    chunk_id = 1

    # Create (page_num, page_text) pairs
    page_pairs = []
    for i in range(1, len(pages), 2):
        try:
            page_num = int(pages[i])
            page_text = pages[i+1].strip()
            page_pairs.append((page_num, page_text))
        except (IndexError, ValueError):
            continue

    # Filter blueprint for this subject
    subject_blueprint = blueprint_df[blueprint_df["subject"].str.lower() == subject.lower()]
    subject_blueprint = subject_blueprint.sort_values("start_page")

    # Process each page
    for page_num, page_text in page_pairs:
        topic = assign_topic_by_page(subject_blueprint, page_num)
        words = page_text.split()
        i = 0
        while i < len(words):
            chunk_words = words[i:i + CHUNK_SIZE]
            if len(chunk_words) < 20:
                break
            chunk_text = " ".join(chunk_words)

            topic_clean = re.sub(r"[^A-Za-z0-9]+", "_", topic)[:40]
            chunk_label = f"{subject.upper()}_{topic_clean}_{chunk_id:03d}"

            chunks.append({
                "id": chunk_label,
                "subject": subject,
                "topic": topic,
                "source_file": filename,
                "text": chunk_text  # keep text for verification & querying
            })
            chunk_id += 1
            i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks

def load_txt_files(data_dir):
    txt_files = list(data_dir.rglob("*.txt"))
    print(f"Found {len(txt_files)} text files.")
    return txt_files

def preprocess():
    all_chunks = []
    txt_files = load_txt_files(DATA_DIR)
    blueprint_df = load_topic_blueprint(TOPIC_BLUEPRINT_CSV)

    for txt_file in txt_files:
        subject = txt_file.parent.name
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text_by_pages(text, subject, txt_file.stem, blueprint_df)
        print(f"{txt_file.name}: {len(chunks)} chunks created")
        all_chunks.extend(chunks)

    return all_chunks

def build_index(chunks):
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode([c["text"] for c in chunks], show_progress_bar=True, convert_to_numpy=True)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors of dimension {dim}")

    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_DIR / "faiss_index.bin"))

    # Save metadata with text for query verification
    metadata = [{k: c[k] for k in ["id", "subject", "topic", "source_file", "text"]} for c in chunks]
    with open(INDEX_DIR / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved for {len(metadata)} chunks")
    return index, metadata

if __name__ == "__main__":
    print("=== Preprocessing & Chunking ===")
    chunks = preprocess()
    print(f"Total chunks created: {len(chunks)}")

    print("\n=== Building FAISS Index ===")
    build_index(chunks)

    print("\nAll done! Index saved in 'index/' folder.")
