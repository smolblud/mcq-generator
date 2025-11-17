import os
import re
import json
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# -----------------------------
# Config
# -----------------------------
FAISS_INDEX_PATH = "index/faiss_index.bin"
CHUNKS_META_PATH = "index/metadata.json"
MCQ_OUTPUT_DIR = Path("MCQ_output")
CSV_PATH = "docs/a.csv"

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

# Load .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

# -----------------------------
# Load embedding model, FAISS index, and metadata
# -----------------------------
print("Loading embedding model...")
emb_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)
print("Index dimension:", index.d)
print("Embedding model dimension:", emb_model.get_sentence_embedding_dimension())

print("Loading chunk metadata...")
with open(CHUNKS_META_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -----------------------------
# LangChain Gemini LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# -----------------------------
# Helper functions
# -----------------------------
def slugify(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"+", "", text)
    return text.strip("_") or "na"

def make_citation_key(meta: dict) -> str:
    source_file = str(meta.get("source_file", "unknown_source")).strip()
    topic = str(meta.get("topic", "")).strip()
    source_slug = slugify(source_file)
    topic_slug = slugify(topic)
    if topic_slug:
        return f"{source_slug}_{topic_slug}"
    else:
        return source_slug or "unknown_source"

def clean_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith(""):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith(""):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text

def retrieve_context(query: str, k: int = 5) -> str:
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)
    distances, indices = index.search(q_emb, k)
    indices = indices[0]
    context_parts: List[str] = []
    for idx in indices:
        if idx < 0 or idx >= len(chunks):
            continue
        meta = chunks[idx]
        text = meta.get("text", "") if isinstance(meta, dict) else str(meta)
        citation_key = make_citation_key(meta)
        context_parts.append(f"[{citation_key}]\n{text}")
    return "\n\n".join(context_parts)

def save_mcqs_to_file(mcqs: list[dict], subject: str, chapter: str | None, topic: str | None, n_questions: int) -> Path:
    MCQ_OUTPUT_DIR.mkdir(exist_ok=True)
    subject_slug = slugify(subject)
    chapter_slug = slugify(chapter) if chapter else "no_chapter"
    topic_slug = slugify(topic) if topic else "all_topics"
    filename = f"{subject_slug}{chapter_slug}{topic_slug}{n_questions}_mcqs.json"
    out_path = MCQ_OUTPUT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mcqs, f, indent=2, ensure_ascii=False)
    return out_path

# -----------------------------
# Prompt templates
# -----------------------------
mcq_prompt_stem = ChatPromptTemplate.from_template(
    """
You are an exam MCQ generator for STEM subjects (Mathematics and Physics).
Use ONLY the context below to create {n_questions} multiple-choice questions
for the given subject and chapter. Do not invent facts that are not supported by the context.

Subject: {subject}
Chapter: {chapter}
Difficulty (overall target): {difficulty}

In the context, you will see lines like:
[<citation_key>] some text...

Requirements for EACH question:
- 1 clear conceptual or computational stem
- 4 options labeled A, B, C, D
- Exactly ONE correct option
- A worked solution written in LaTeX (no surrounding $$, just raw LaTeX)
- A difficulty tag for that question: one of "Easy", "Medium", "Hard"
- At least one citation using the EXACT citation_key shown in the context (without brackets).

<context>
{context}
</context>

Return the questions as a JSON array.
Each element must have:
- "stem": string
- "options": {"A": "...", "B": "...", "C": "...", "D": "..."}
- "answer": one of "A", "B", "C", "D"
- "solution_latex": string
- "difficulty": "Easy" | "Medium" | "Hard"
- "citations": list of citation keys
Strictly output ONLY valid JSON.
"""
)

mcq_prompt_sat = ChatPromptTemplate.from_template(
    """
You are an exam MCQ generator for SAT English (Reading & Writing).
Use ONLY the context below to create {n_questions} multiple-choice questions.

Subject: {subject}
Chapter: {chapter}
Topic: {topic}
Difficulty (overall target): {difficulty}

<context>
{context}
</context>

Return the questions as a JSON array.
Each element must have:
- "question_type": "reading" | "grammar" | "vocab"
- "stem": string
- "options": {"A": "...", "B": "...", "C": "...", "D": "..."}
- "answer": one of "A", "B", "C", "D"
- "evidence_span": string
- "citations": list of citation keys
Strictly output ONLY valid JSON.
"""
)

# -----------------------------
# MCQ generation
# -----------------------------
def generate_mcqs(subject: str, chapter: str | None = None, topic: str | None = None, difficulty: str = "Medium", n_questions: int = 5) -> list[dict]:
    parts = [subject]
    if chapter:
        parts.append(chapter)
    if topic:
        parts.append(topic)
    query = " - ".join(parts)
    context = retrieve_context(query, k=8)
    if not context.strip():
        raise ValueError(f"No context retrieved for {query}")
    prompt = mcq_prompt_sat if "sat" in subject.lower() or "english" in subject.lower() else mcq_prompt_stem
    messages = prompt.format_messages(
        subject=subject,
        chapter=chapter or "",
        topic=topic or "All topics",
        difficulty=difficulty,
        n_questions=n_questions,
        context=context,
    )
    response = llm.invoke(messages)
    cleaned = clean_json_text(response.content)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, list):
        raise ValueError("Expected JSON array of MCQs.")
    return parsed

if __name__ == "__main__":
    TOTAL_MCQS = 30
    df = pd.read_csv(CSV_PATH)
    num_rows = len(df)

    # Compute roughly equal number of MCQs per row
    base_mcqs = TOTAL_MCQS // num_rows
    extra = TOTAL_MCQS % num_rows  # distribute remainder

    total_mcqs_generated = 0

    for idx, row in df.iterrows():
        subject = row["subject"]
        chapter = row["unit"]
        topic = row["topic"] if not pd.isna(row["topic"]) else None

        # Add 1 extra MCQ to the first 'extra' rows to reach exactly 30
        n_questions = base_mcqs + (1 if idx < extra else 0)

        print(f"Generating {n_questions} MCQs for Subject='{subject}', Chapter='{chapter}', Topic='{topic or 'All'}'...")
        mcqs = generate_mcqs(subject, chapter, topic, difficulty="Medium", n_questions=n_questions)
        total_mcqs_generated += len(mcqs)

        out_path = save_mcqs_to_file(mcqs, subject, chapter, topic, n_questions)
        print(f"Saved {len(mcqs)} MCQs to: {out_path.resolve()}\n")

    print(f"Total MCQs generated across all topics: {total_mcqs_generated}")