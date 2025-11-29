import json
import re
import numpy as np
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Load embedding model
# -----------------------------
print("Loading embedding model...")
emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# -----------------------------
# Validation Helpers
# -----------------------------
def check_uniqueness(mcqs, threshold=0.85):
    stems = [q["stem"] for q in mcqs]
    print(f"[DEBUG] Checking uniqueness for {len(stems)} stems...")
    if len(stems) < 2:
        print("[DEBUG] Not enough stems to check uniqueness.")
        return []

    embeddings = emb_model.encode(stems)
    sim_matrix = cosine_similarity(embeddings)
    duplicates = []

    n = len(stems)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] > threshold:
                duplicates.append((i, j, sim_matrix[i][j]))
    print(f"[DEBUG] Found {len(duplicates)} duplicate pairs based on threshold {threshold}")
    return duplicates

def check_distractors(mcqs):
    issues = []
    print(f"[DEBUG] Checking distractors for {len(mcqs)} MCQs...")
    for idx, q in enumerate(mcqs):
        options = q.get("options", {})
        answer = q.get("answer")
        if len(options) != 4:
            issues.append((idx, "Incorrect number of options"))
        if answer not in options:
            issues.append((idx, "Correct answer not in options"))
    print(f"[DEBUG] Found {len(issues)} distractor issues")
    return issues

def check_citations(mcqs):
    issues = []
    print(f"[DEBUG] Checking citations for {len(mcqs)} MCQs...")
    for idx, q in enumerate(mcqs):
        citations = q.get("citations", [])
        if not citations:
            issues.append((idx, "No citations provided"))
    print(f"[DEBUG] Found {len(issues)} citation issues")
    return issues

def check_math_consistency(mcqs):
    issues = []
    print(f"[DEBUG] Checking math consistency for {len(mcqs)} MCQs...")
    for idx, q in enumerate(mcqs):
        solution = q.get("solution_latex", "")
        options = q.get("options", {})
        answer_key = q.get("answer")

        if not solution or not any(c.isdigit() for c in solution):
            print(f"[DEBUG] Skipping MCQ {idx} (non-math or no numeric solution)")
            continue

        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", solution)
        if not numbers:
            issues.append((idx, "No numeric expression found"))
            continue

        try:
            opt_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", options[answer_key])[0])
            if not any(np.isclose(float(n), opt_val, atol=1e-2) for n in numbers):
                issues.append((idx, "No number matches correct option"))
            else:
                print(f"[DEBUG] MCQ {idx}: math check passed, matched value {opt_val}")
        except Exception as e:
            issues.append((idx, f"Math parsing skipped ({e})"))

    print(f"[DEBUG] Found {len(issues)} math issues")
    return issues

def validate_mcqs(mcqs, uniqueness_threshold=0.85):
    if not mcqs:
        print("[DEBUG] No MCQs to validate.")
        return {
            "duplicates": [],
            "distractor_issues": [],
            "citation_issues": [],
            "math_issues": []
        }

    print("[DEBUG] Starting full MCQ validation...")
    results = {
        "duplicates": check_uniqueness(mcqs, threshold=uniqueness_threshold),
        "distractor_issues": check_distractors(mcqs),
        "citation_issues": check_citations(mcqs),
        "math_issues": check_math_consistency(mcqs),
    }
    print("[DEBUG] Validation complete.\n")
    return results

# -----------------------------
# Main: validate all MCQs in folder
# -----------------------------
if __name__ == "__main__":
    mcq_folder = Path("MCQ_output")
    all_mcqs = []

    # Load all JSON files in the folder
    for file_path in mcq_folder.glob("*.json"):
        print(f"\n--- Loading {file_path.name} ---")
        with open(file_path, "r", encoding="utf-8") as f:
            mcqs = json.load(f)
            print(f"[DEBUG] Loaded {len(mcqs)} MCQs from {file_path.name}")
            all_mcqs.extend(mcqs)

    # --- Add two hardcoded wrong MCQs to test the suite ---
    wrong_mcqs = [
        # Duplicate stem (will trigger uniqueness)
        {
            "stem": all_mcqs[0]["stem"] if all_mcqs else "Duplicate stem example",
            "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
            "answer": "A",
            "solution_latex": "1",
            "citations": ["dummy"]
        },
        # Wrong math answer (will trigger math issue)
        {
            "stem": "MCQ with wrong math answer",
            "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
            "answer": "B",
            "solution_latex": "3",
            "citations": ["dummy"]
        }
    ]
    all_mcqs.extend(wrong_mcqs)
    print(f"\n[DEBUG] Total MCQs including hardcoded wrong examples: {len(all_mcqs)}\n")

    # Run validation
    results = validate_mcqs(all_mcqs)

    # Print summary
    print("\nValidation Results Summary:")
    for key, issues in results.items():
        if issues:
            print(f"{key}: {len(issues)} issues")
            for issue in issues:
                print("  ", issue)
        else:
            print(f"{key}: No issues found ✅")
