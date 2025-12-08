import json
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading embedding model...")
emb_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")


# -----------------------------
# Validation Functions
# -----------------------------
def compute_duplicates(mcqs, threshold=0.85):
    stems = [q["stem"] for q in mcqs]
    embeddings = emb_model.encode(stems)
    sim_matrix = cosine_similarity(embeddings)
    duplicate_map = {}
    n = len(stems)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] > threshold:
                duplicate_map.setdefault(i, []).append({"index": j, "similarity": float(sim_matrix[i][j])})
                duplicate_map.setdefault(j, []).append({"index": i, "similarity": float(sim_matrix[i][j])})
    return duplicate_map


def validate_distractors(mcq):
    options = mcq.get("options", {})
    answer = mcq.get("answer")
    result = {"options_valid": True, "options_msg": "", "distractor_plausible": True, "distractor_msg": ""}

    if len(options) != 4:
        result["options_valid"] = False
        result["options_msg"] = f"Incorrect number of options ({len(options)})"
    if answer not in options:
        result["options_valid"] = False
        result["options_msg"] += " | Correct answer not in options"

    # Only check numeric plausibility for math/physics MCQs
    if mcq.get("solution_latex"):
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", mcq["solution_latex"])
        if numbers:
            option_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", options[answer])
            if option_numbers:
                correct_val = float(option_numbers[0])
                implausible = []
                for k, v in options.items():
                    if k != answer:
                        v_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", v)
                        if any(np.isclose(correct_val, float(n), atol=1e-2) for n in v_numbers):
                            implausible.append(k)
                if implausible:
                    result["distractor_plausible"] = False
                    result["distractor_msg"] = f"Options {implausible} too similar to correct answer"
    return result


def validate_math_physics(mcq):
    result = {"math_physics_valid": True, "math_physics_msg": ""}
    solution = mcq.get("solution_latex", "")
    options = mcq.get("options", {})
    answer = mcq.get("answer")

    if solution and options and answer in options:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", solution)
        if numbers:
            try:
                correct_val = float(re.findall(r"[-+]?\d*\.\d+|\d+", options[answer])[0])
                if not any(np.isclose(float(n), correct_val, atol=1e-2) for n in numbers):
                    result["math_physics_valid"] = False
                    result["math_physics_msg"] = "No number matches correct option"
            except Exception as e:
                result["math_physics_valid"] = False
                result["math_physics_msg"] = f"Math parsing error ({e})"
    return result


def validate_citations(mcq):
    return {"citations_valid": bool(mcq.get("citations", []))}


def validate_reading(mcq):
    if mcq.get("question_type") == "reading":
        if "evidence_span" in mcq and mcq["evidence_span"]:
            return {"reading_evidence_valid": True, "reading_msg": ""}
        else:
            return {"reading_evidence_valid": False, "reading_msg": "Missing evidence_span"}
    return {"reading_evidence_valid": True, "reading_msg": ""}


# -----------------------------
# Main Validation Loop
# -----------------------------
def run_validation(mcqs, uniqueness_threshold=0.85):
    duplicates_map = compute_duplicates(mcqs, threshold=uniqueness_threshold)
    validated_mcqs = []

    for idx, mcq in enumerate(mcqs):
        mcq_result = {"index": idx, "stem": mcq.get("stem")}

        # Uniqueness
        if idx in duplicates_map:
            mcq_result["duplicate"] = True
            mcq_result["duplicate_of"] = duplicates_map[idx]
        else:
            mcq_result["duplicate"] = False
            mcq_result["duplicate_of"] = []

        # Distractors
        mcq_result.update(validate_distractors(mcq))

        # Math/Physics
        mcq_result.update(validate_math_physics(mcq))

        # Citations
        mcq_result.update(validate_citations(mcq))

        # Reading evidence
        mcq_result.update(validate_reading(mcq))

        validated_mcqs.append(mcq_result)
    return validated_mcqs


# -----------------------------
# Run on all JSON MCQs in folder
# -----------------------------
if __name__ == "__main__":
    mcq_folder = Path("MCQ_output")
    all_mcqs = []

    for file_path in mcq_folder.glob("*.json"):
        print(f"Loading {file_path.name}...")
        with open(file_path, "r", encoding="utf-8") as f:
            mcqs = json.load(f)
            all_mcqs.extend(mcqs)

    results = run_validation(all_mcqs)

    # -----------------------------
    # Save JSON with metadata
    # -----------------------------
    output_file = "validation_results.json"

    metadata = {
        "description": "This file contains MCQs with validation results from the validation suite.",
        "fields": {
            "index": "Unique identifier of the MCQ in the dataset",
            "stem": "The question text or prompt",
            "duplicate": "Boolean indicating if the MCQ is a duplicate",
            "duplicate_of": "List of indices of MCQs that this one duplicates (if any)",
            "options_valid": "Boolean indicating if all answer options are valid",
            "options_msg": "Message explaining any issues with options",
            "distractor_plausible": "Boolean indicating if distractors are plausible",
            "distractor_msg": "Message explaining distractor issues",
            "math_physics_valid": "Boolean indicating if any math/physics calculations parse correctly",
            "math_physics_msg": "Message explaining math/physics issues",
            "citations_valid": "Boolean indicating if citation references are valid",
            "reading_evidence_valid": "Boolean indicating if reading-based evidence supports the answer",
            "reading_msg": "Message explaining any reading/evidence issues"
        }
    }

    output_data = {
        "metadata": metadata,
        "mcqs": results
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Validation complete. Detailed results with metadata saved to {output_file}")
