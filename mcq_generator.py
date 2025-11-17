# mcq_generator_gemini.py

import os
import json
from typing import List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Config
# -----------------------------

# Path to your existing FAISS index and chunk metadata
FAISS_INDEX_PATH = "index/faiss_index.bin"
CHUNKS_META_PATH = "index/metadata.json"

# SentenceTransformer model you used when building the FAISS index
EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # change if you used a different one

# Gemini model name
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

# Make sure you set this in your environment, e.g.:
# export GOOGLE_API_KEY="your_key_here"
from dotenv import load_dotenv
load_dotenv()  # <-- THIS loads your .env file

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")


# -----------------------------
# Load embeddings + FAISS index + metadata
# -----------------------------

print("Loading embedding model...")
emb_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading chunk metadata...")
with open(CHUNKS_META_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)  # expected: list of dicts or list of texts


# -----------------------------
# LangChain: Gemini LLM
# -----------------------------

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
)


# -----------------------------
# Prompt template for MCQ generation
# -----------------------------

mcq_prompt = ChatPromptTemplate.from_template(
    """
You are an exam MCQ generator.

Use ONLY the context below to create {n_questions} multiple-choice questions
for the given subject and topic. Do not invent facts that are not supported by the context.

Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}

Requirements for each question:
- 1 clear stem
- 4 options labeled A, B, C, D
- Exactly ONE correct option
- A short explanation/solution
- At least one citation referencing the source file and topic, formatted as: [SourceFile_Topic]

<context>
{context}
</context>

Return the questions as a JSON array. Each element should have:
- "stem": string
- "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}
- "answer": one of "A", "B", "C", "D"
- "explanation": string
- "citations": list of strings (e.g. ["chunk 1", "chunk 2"])

Strictly output ONLY valid JSON.
"""
)



# -----------------------------
# Retrieval helper using FAISS
# -----------------------------

def retrieve_context(subject: str, topic: str, k: int = 5, search_k: int = 50) -> str:
    """
    Retrieve top-k chunks from FAISS and format them with [source_file_topic] tags.
    First tries to filter by subject/topic from metadata; if nothing matches,
    falls back to using the top-k chunks without filtering.
    """
    # Build retrieval query
    query = f"{subject} - {topic}"

    # Embed query
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype="float32")

    faiss.normalize_L2(q_emb)

    # Search more results than we finally need, then filter
    distances, indices = index.search(q_emb, search_k)
    indices = indices[0]

    # ---------- First pass: try with metadata filters ----------
    filtered_parts: List[str] = []

    for idx in indices:
        if idx < 0 or idx >= len(chunks):
            continue

        meta = chunks[idx]

        if isinstance(meta, dict):
            text = meta.get("text", "")
            source_file = meta.get("source_file", "unknown")
            chunk_subject = (meta.get("subject", "") or "").strip()
            chunk_topic = (meta.get("topic", "") or "").strip()
        else:
            text = str(meta)
            source_file = "unknown"
            chunk_subject = ""
            chunk_topic = ""

        # Apply filters ONLY if metadata is present
        if chunk_subject and chunk_subject.lower() != subject.lower():
            continue

        if chunk_topic and (
            topic.lower() not in chunk_topic.lower()
            and chunk_topic.lower() not in topic.lower()
        ):
            continue

        topic_name = chunk_topic if chunk_topic else topic
        citation_tag = f"[{source_file}_{topic_name}]"

        filtered_parts.append(f"{citation_tag} {text}")

        if len(filtered_parts) >= k:
            break

    # If we got something with filters, use that
    if filtered_parts:
        return "\n\n".join(filtered_parts)

    # ---------- Second pass: fallback with NO filters ----------
    fallback_parts: List[str] = []

    for idx in indices:
        if idx < 0 or idx >= len(chunks):
            continue

        meta = chunks[idx]

        if isinstance(meta, dict):
            text = meta.get("text", "")
            source_file = meta.get("source_file", "unknown")
            chunk_topic = (meta.get("topic", "") or "").strip()
        else:
            text = str(meta)
            source_file = "unknown"
            chunk_topic = ""

        topic_name = chunk_topic if chunk_topic else topic
        citation_tag = f"[{source_file}_{topic_name}]"

        fallback_parts.append(f"{citation_tag} {text}")

        if len(fallback_parts) >= k:
            break

    return "\n\n".join(fallback_parts)



# -----------------------------
# MCQ generation function
# -----------------------------


def generate_mcqs(
    subject: str,
    topic: str,
    difficulty: str = "Medium",
    n_questions: int = 5,
) -> str:
    """
    Generate MCQs using Gemini + LangChain + FAISS index.
    Returns the raw JSON string from the model.
    """
    # Retrieve context from FAISS using subject + topic
    context = retrieve_context(subject=subject, topic=topic, k=4)

    if not context.strip():
        raise ValueError("No relevant context retrieved from FAISS for this topic.")

    # Format prompt messages
    messages = mcq_prompt.format_messages(
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        n_questions=n_questions,
        context=context,
    )

    # Call Gemini via LangChain
    response = llm.invoke(messages)

    # response.content should be a JSON string
    return response.content


# -----------------------------
# Demo usage (CLI-style)
# -----------------------------

if __name__ == "__main__":

    subject = "math"
    topic = "Sequences and Series"
    difficulty = "Medium"
    n_questions = 3

    print(f"Generating {n_questions} MCQs for {subject} – {topic}...")
    json_output = generate_mcqs(
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        n_questions=n_questions,
    )

    print("\nRaw model output (JSON):\n")
    print(json_output)
