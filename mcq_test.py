import os
import re
import json
import random
from pathlib import Path
from typing import List, Dict, Any

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
TOPIC_BLUEPRINT_CSV = Path("docs/topic_blueprint.csv")

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

TOTAL_QUESTIONS = 30
DEFAULT_DIFFICULTY = "Medium"

# -----------------------------
# Environment & API key
# -----------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY environment variable is not set.")

# -----------------------------
# Load models & index
# -----------------------------
print("Loading embedding model...")
emb_model = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading FAISS index...")
index = faiss.read_index(FAISS_INDEX_PATH)

print("Loading chunk metadata...")
with open(CHUNKS_META_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# -----------------------------
# Load topic blueprint
# -----------------------------
def load_topic_blueprint(csv_path: Path = TOPIC_BLUEPRINT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"subject", "unit", "topic", "start_page", "end_page", "source_file"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Topic blueprint missing columns: {missing}")
    return df

# -----------------------------
# Gemini LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL_NAME,
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
)

# -----------------------------
# Prompt template
# -----------------------------
mcq_prompt = ChatPromptTemplate.from_template(
    """
You are an exam MCQ generator.

Use ONLY the context below to create {n_questions} multiple-choice questions
for the given subject and topic.

Subject: {subject}
Topic: {topic}
Difficulty: {difficulty}

Requirements:
- 1 clear stem
- 4 options (A–D)
- Exactly ONE correct option
- Explanation
- At least one citation like: [SourceFile_Topic]
- difficulty must be "{difficulty}"

<context>
{context}
</context>

Return ONLY valid JSON (array of objects).
"""
)

# -----------------------------
# Retrieval
# -----------------------------
def retrieve_context(subject: str, topic: str, k: int = 4, search_k: int = 50) -> str:
    query = f"{subject} {topic}"
    q_emb = emb_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    _, indices = index.search(q_emb, search_k)
    indices = indices[0]

    parts = []
    for idx in indices:
        if idx < 0 or idx >= len(chunks):
            continue
        meta = chunks[idx]
        text = meta.get("text", "")
        source_file = meta.get("source_file", "unknown")
        chunk_topic = meta.get("topic", topic)
        citation = f"[{source_file}_{chunk_topic}]"
        parts.append(f"{citation} {text}")
        if len(parts) >= k:
            break
    return "\n\n".join(parts)

# -----------------------------
# JSON parsing
# -----------------------------
def parse_llm_json(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.I)
        text = re.sub(r"```$", "", text)
    return json.loads(text)

# -----------------------------
# Generate MCQs for ONE topic
# -----------------------------
def generate_mcqs_for_topic(subject: str, topic: str, n_questions: int, difficulty: str) -> List[Dict[str, Any]]:
    context = retrieve_context(subject, topic)
    if not context:
        return []

    messages = mcq_prompt.format_messages(
        subject=subject,
        topic=topic,
        difficulty=difficulty,
        n_questions=n_questions,
        context=context,
    )

    response = llm.invoke(messages)
    try:
        questions = parse_llm_json(response.content)
    except Exception as e:
        print(f"[ERROR] JSON parse failed for {topic}: {e}")
        return []

    # Add metadata & log
    for q in questions:
        q["topic"] = topic
        q["subject"] = subject
        stem_snippet = (q.get("stem", "")[:60] + "...") if q.get("stem") else ""
        print(f"[MCQ GENERATED] Subject: {subject} | Topic: {topic} | Stem snippet: {stem_snippet}")

    return questions

# -----------------------------
# RANDOM blueprint-aware generator
# -----------------------------
def generate_mcqs_random(difficulty: str = DEFAULT_DIFFICULTY) -> List[Dict[str, Any]]:
    df = load_topic_blueprint()
    all_questions: List[Dict[str, Any]] = []

    while len(all_questions) < TOTAL_QUESTIONS:
        row = df.sample(1).iloc[0]  # pick random topic
        subject = row["subject"]
        topic = row["topic"]

        n_q = random.choice([1, 1, 2, 2, 3])  # random MCQs per topic
        remaining = TOTAL_QUESTIONS - len(all_questions)
        n_q = min(n_q, remaining)  # do not exceed TOTAL_QUESTIONS

        questions = generate_mcqs_for_topic(subject, topic, n_q, difficulty)
        all_questions.extend(questions)

    return all_questions[:TOTAL_QUESTIONS]

# -----------------------------
# CLI entrypoint
# -----------------------------
if __name__ == "__main__":
    difficulty = "Hard"
    print(f"Generating {TOTAL_QUESTIONS} random {difficulty} MCQs...")

    questions = generate_mcqs_random(difficulty)

    OUTPUT_FOLDER = Path("MCQ_test_output")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    output_file = OUTPUT_FOLDER / f"generated_mcqs_{difficulty.lower()}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(questions)} MCQs to {output_file}")