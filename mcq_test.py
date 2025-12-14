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

# Topic blueprint CSV (with columns: subject, unit, topic, start_page, end_page, source_file)
TOPIC_BLUEPRINT_CSV = Path("docs/topic_blueprint.csv")

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Total MCQs to generate (global)
TOTAL_QUESTIONS = 30
# Per-topic upper bound (we'll generate up to this many per topic)
MAX_QUESTIONS_PER_TOPIC = 5
DEFAULT_DIFFICULTY = "Medium"


# -----------------------------
# Environment & API key
# -----------------------------

load_dotenv()
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
# Load topic blueprint
# -----------------------------

def load_topic_blueprint(csv_path: Path = TOPIC_BLUEPRINT_CSV) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"subject", "unit", "topic", "start_page", "end_page", "source_file"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Topic blueprint is missing columns: {missing}")
    return df


def iter_random_unique_topics(df: pd.DataFrame):
    """
    Yield (subject, topic) pairs in random order, without repetition.
    """
    # Shuffle rows
    df_shuffled = df.sample(frac=1.0, random_state=None).reset_index(drop=True)
    seen = set()
    for _, row in df_shuffled.iterrows():
        key = (str(row["subject"]).strip(), str(row["topic"]).strip())
        if key in seen:
            continue
        seen.add(key)
        yield key


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
- Set "difficulty" to exactly "{difficulty}" in the JSON.

<context>
{context}
</context>

    Return the questions as a JSON array. Each element should have:
    - "stem": string
    - "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}
    - "answer": one of "A", "B", "C", "D"
    - "explanation": string
    - "citations": list of strings (e.g. ["chunk 1", "chunk 2"])
    - "difficulty": string (exactly "{difficulty}")
    
    Strictly output ONLY valid JSON.

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
    query = f"{subject} - {topic}"

    # Embed query
    q_emb = emb_model.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)

    # Search
    distances, indices = index.search(q_emb, search_k)
    indices = indices[0]

    # ---------- First pass: with metadata filters ----------
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

    if filtered_parts:
        return "\n\n".join(filtered_parts)

    # ---------- Second pass: no filters ----------
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
# Helper: robust JSON parsing
# -----------------------------

def parse_llm_json(raw: str) -> List[Dict[str, Any]]:
    """
    Try to parse the LLM output as JSON.
    Handles the case where it comes wrapped in ```json ... ``` fences.
    """
    text = raw.strip()

    # Strip code fences if present
    if text.startswith("```"):
        # Remove starting ```json or ``` and ending ```
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    return json.loads(text)


# -----------------------------
# MCQ generation per topic
# -----------------------------
def generate_mcqs_for_topic(
    subject: str,
    topic: str,
    n_questions: int,
    difficulty: str | None = None
) -> List[Dict[str, Any]]:
    """
    Generate MCQs for a given subject/topic using Gemini + FAISS.
    Returns a list of question dicts with 'topic' field set.
    """
    context = retrieve_context(subject=subject, topic=topic, k=4)

    if not context.strip():
        print(f"[WARN] No context retrieved for topic: {subject} – {topic}. Skipping.")
        return []

    level = (difficulty or DEFAULT_DIFFICULTY).strip()

    messages = mcq_prompt.format_messages(
        subject=subject,
        topic=topic,
        difficulty=level,
        n_questions=n_questions,
        context=context,
    )

    response = llm.invoke(messages)
    raw_output = response.content

    try:
        questions = parse_llm_json(raw_output)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON for topic {subject} – {topic}: {e}")
        print("Raw output was:\n", raw_output)
        return []

    if not isinstance(questions, list):
        print(f"[ERROR] LLM output for topic {subject} – {topic} is not a JSON array.")
        return []

    # -----------------------------
    # Inject 'topic' field so evaluation works
    # -----------------------------
    for q in questions:
        q['topic'] = topic  # ensures blueprint matching works

    return questions

# -----------------------------
# Main generator: 30 random MCQs from blueprint
# -----------------------------

def generate_mcqs_balanced(difficulty: str = DEFAULT_DIFFICULTY) -> List[Dict[str, Any]]:
    """
    Generate TOTAL_QUESTIONS MCQs distributed across all topics in the blueprint.
    """
    df = load_topic_blueprint()
    all_questions: List[Dict[str, Any]] = []

    total_topics = len(df)
    if total_topics == 0:
        print("[ERROR] Blueprint has no topics.")
        return []

    # Base allocation
    per_topic_quota = TOTAL_QUESTIONS // total_topics
    remainder = TOTAL_QUESTIONS % total_topics

    for i, row in df.iterrows():
        subject = row['subject']
        topic = row['topic']

        # Handle remainder by giving one extra question to first 'remainder' topics
        n_questions_for_topic = min(MAX_QUESTIONS_PER_TOPIC, per_topic_quota + (1 if i < remainder else 0))

        if n_questions_for_topic <= 0:
            continue

        print(f"[INFO] Generating {n_questions_for_topic} MCQs for {subject} – {topic} at difficulty {difficulty}")

        topic_questions = generate_mcqs_for_topic(
            subject,
            topic,
            n_questions=n_questions_for_topic,
            difficulty=difficulty
        )

        all_questions.extend(topic_questions)

        if len(all_questions) >= TOTAL_QUESTIONS:
            break

    if len(all_questions) < TOTAL_QUESTIONS:
        print(f"[WARN] Only generated {len(all_questions)} questions, fewer than requested {TOTAL_QUESTIONS}.")

    return all_questions[:TOTAL_QUESTIONS]


# -----------------------------
# CLI entrypoint
# -----------------------------

if __name__ == "__main__":
    difficulty = "Hard"   # or "Easy" / "Medium" or read from CLI/env
    print(f"Generating {TOTAL_QUESTIONS} {difficulty} MCQs from random topics in the blueprint...")
    
    # -----------------------------
# Save generated MCQs
# -----------------------------
    OUTPUT_FOLDER = Path("MCQ_test_output")
    OUTPUT_FOLDER.mkdir(exist_ok=True)

    output_file = OUTPUT_FOLDER / f"generated_mcqs_{difficulty.lower()}.json"
    with open(output_file, "w", encoding="utf-8") as f:
     json.dump(questions, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(questions)} MCQs to {output_file}")


    print("\nFinal MCQ JSON array:\n")
    print(json.dumps(questions, indent=2, ensure_ascii=False))

