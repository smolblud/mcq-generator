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
    difficulty: str | None = None  # 👈 NEW
) -> List[Dict[str, Any]]:
    """
    Generate MCQs for a given subject/topic using Gemini + FAISS.
    Returns a list of question dicts.
    """
    context = retrieve_context(subject=subject, topic=topic, k=4)

    if not context.strip():
        print(f"[WARN] No context retrieved for topic: {subject} – {topic}. Skipping.")
        return []

    # Normalize difficulty
    level = (difficulty or DEFAULT_DIFFICULTY).strip()
    # If your prompt expects lowercase:
    # level = level.lower()

    messages = mcq_prompt.format_messages(
        subject=subject,
        topic=topic,
        difficulty=level,          # 👈 use UI-provided difficulty
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

    # Ensure it's a list
    if not isinstance(questions, list):
        print(f"[ERROR] LLM output for topic {subject} – {topic} is not a JSON array.")
        return []

    return questions



# -----------------------------
# NEW: Subject-wise Generation
# -----------------------------

def get_topics_for_subject(subject: str) -> List[str]:
    """Return a list of unique topics for a given subject from the blueprint."""
    df = load_topic_blueprint()
    # Filter by subject (case-insensitive)
    topics = df[df["subject"].str.strip().str.lower() == subject.strip().lower()]["topic"].unique()
    return [str(t).strip() for t in topics]

def generate_mcqs_for_subject(
    subject: str,
    n_questions: int,
    difficulty: str = "Medium"
) -> List[Dict[str, Any]]:
    """
    Generate MCQs for a specific subject by selecting random topics 
    within that subject.
    """
    all_topics = get_topics_for_subject(subject)
    
    if not all_topics:
        print(f"[WARN] No topics found for subject: {subject}")
        return []

    # Strategy: Don't pick 1 question from 10 topics (too much context switching).
    # Instead, aim for ~3-5 questions per topic to utilize context window better.
    questions_per_topic_target = 3
    
    # Calculate how many topics we need to reach n_questions
    num_topics_needed = max(1, int(np.ceil(n_questions / questions_per_topic_target)))
    
    # Select random topics (without replacement if possible)
    if num_topics_needed > len(all_topics):
        selected_topics = np.random.choice(all_topics, num_topics_needed, replace=True)
    else:
        selected_topics = np.random.choice(all_topics, num_topics_needed, replace=False)

    print(f"[INFO] Selected topics for {subject}: {selected_topics}")

    all_questions = []
    questions_remaining = n_questions

    for i, topic in enumerate(selected_topics):
        if questions_remaining <= 0:
            break
        
        # Distribute remaining questions roughly evenly
        topics_left = len(selected_topics) - i
        # Simple integer division distribution
        n_for_this = int(np.ceil(questions_remaining / topics_left))
        
        print(f"[INFO] Generating {n_for_this} MCQs for {subject} - {topic}")
        
        qs = generate_mcqs_for_topic(
            subject=subject,
            topic=topic,
            n_questions=n_for_this,
            difficulty=difficulty
        )
        
        all_questions.extend(qs)
        questions_remaining = n_questions - len(all_questions)

    # Scramble the final list so questions from the same topic aren't strictly adjacent
    random.shuffle(all_questions)
    
    # Trim to exact number requested
    return all_questions[:n_questions]

# -----------------------------
# Main generator: 30 random MCQs from blueprint
# -----------------------------

def generate_random_mcqs_from_blueprint(
    difficulty: str = DEFAULT_DIFFICULTY,  # 👈 NEW
) -> List[Dict[str, Any]]:
    """
    Load the topic blueprint and generate a total of 30 MCQs
    from randomly selected, different topics.
    """
    df = load_topic_blueprint()
    all_questions: List[Dict[str, Any]] = []

    questions_remaining = TOTAL_QUESTIONS

    for subject, topic in iter_random_unique_topics(df):
        if questions_remaining <= 0:
            break

        n_for_topic = min(MAX_QUESTIONS_PER_TOPIC, questions_remaining)
        print(f"\n[INFO] Generating {n_for_topic} MCQs for: {subject} – {topic} "
              f"at difficulty {difficulty}")
        
        topic_questions = generate_mcqs_for_topic(
            subject,
            topic,
            n_questions=n_for_topic,
            difficulty=difficulty,  # 👈 pass it through
        )

        all_questions.extend(topic_questions)
        questions_remaining = TOTAL_QUESTIONS - len(all_questions)
        print(f"[INFO] Total questions so far: {len(all_questions)}; remaining: {questions_remaining}")

        if questions_remaining <= 0:
            break

    if len(all_questions) < TOTAL_QUESTIONS:
        print(
            f"[WARN] Only generated {len(all_questions)} questions "
            f"from available topics, fewer than requested {TOTAL_QUESTIONS}."
        )

    return all_questions[:TOTAL_QUESTIONS]



# -----------------------------
# CLI entrypoint
# -----------------------------

if __name__ == "__main__":
    difficulty = "Hard"   # or "Easy" / "Medium" or read from CLI/env
    print(f"Generating {TOTAL_QUESTIONS} {difficulty} MCQs from random topics in the blueprint...")
    questions = generate_random_mcqs_from_blueprint(difficulty=difficulty)

    print("\nFinal MCQ JSON array:\n")
    print(json.dumps(questions, indent=2, ensure_ascii=False))

