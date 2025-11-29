#!/usr/bin/env python3
"""
adaptive_engine_full.py

Adaptive engine with:
 - theta updates
 - difficulty control
 - topic balancing
 - interactive or simulated student responses
 - avoids repeating items until exhausted
 - session logging to JSON

Usage:
  python adaptive_engine_full.py --simulate --steps 10
  python adaptive_engine_full.py --interactive --steps 10
"""

import os
import json
import random
import argparse
import math
from pathlib import Path
from typing import List, Dict, Any, Optional

# -----------------------
# Configuration / params
# -----------------------
THETA_STEP = 0.15         # how much theta moves on correct/incorrect in simple update
THETA_MIN = -2.0
THETA_MAX = 2.0

# logistic scale for simulated probability
LOGISTIC_SCALE = 1.7

# difficulty -> item difficulty parameter (b). Higher b -> harder
DIFF_TO_B = {
    "Easy": -0.5,
    "Medium": 0.0,
    "Hard": 0.5
}

# -----------------------
# Helpers
# -----------------------
def clamp_theta(theta: float) -> float:
    return max(THETA_MIN, min(THETA_MAX, theta))

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# -----------------------
# Load MCQs and infer topic
# -----------------------
def infer_topic_from_filename(fname: str) -> str:
    # fname without extension
    base = Path(fname).stem
    parts = base.split("_")
    # parts like ["physicsunit","18diffraction","and","interference1","mcqs"] maybe
    # heuristic: drop first two parts if they look like subject/unit, and drop trailing "mcqs" or numbers
    # simpler: take everything between the first two underscores and the trailing number token
    if len(parts) >= 3:
        # remove tokens that look like file suffix (ending with 'mcqs' or containing 'mcq')
        clean = [p for p in parts[2:] if "mcq" not in p.lower()]
        # remove trailing tokens that are just digits or end with digits
        clean = [p for p in clean if not p.rstrip("0123456789").isalnum() or any(ch.isalpha() for ch in p)]
        if not clean:
            # fallback: join from 2:-1
            return "_".join(parts[2:-1]) if len(parts) > 3 else parts[2]
        return "_".join(clean)
    return "misc"

def load_all_mcqs(folder: str = "MCQ_output") -> List[Dict[str, Any]]:
    mcqs = []
    folderp = Path(folder)
    if not folderp.exists():
        print(f"[ERROR] MCQ folder '{folder}' not found.")
        return mcqs

    for fp in sorted(folderp.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            topic = infer_topic_from_filename(fp.name)
            for q in data:
                # normalize shape
                q_norm = dict(q)  # copy
                q_norm.setdefault("difficulty", q_norm.get("difficulty", "Medium"))
                q_norm.setdefault("topic", q_norm.get("topic", topic))
                q_norm.setdefault("source_file", fp.name)
                # ensure options is a dict
                q_norm["options"] = dict(q_norm.get("options", {}))
                # ensure answer exists (if not set to None)
                q_norm["answer"] = q_norm.get("answer")
                mcqs.append(q_norm)
        except Exception as e:
            print(f"[LOAD ERROR] {fp.name}: {e}")
    print(f"[DEBUG] Loaded {len(mcqs)} MCQs from '{folder}'")
    return mcqs

# -----------------------
# Difficulty & topic utilities
# -----------------------
def target_difficulty(theta: float) -> str:
    if theta <= -0.5:
        return "Easy"
    elif theta >= 0.5:
        return "Hard"
    else:
        return "Medium"

def select_topic_least_asked(topics_asked: Dict[str, int], available_topics: List[str]) -> Optional[str]:
    # prefer topics with zero counts first
    for t in available_topics:
        if topics_asked.get(t, 0) == 0:
            return t
    # otherwise pick the least asked
    sorted_topics = sorted(available_topics, key=lambda x: topics_asked.get(x, 0))
    return sorted_topics[0] if sorted_topics else None

# -----------------------
# Select next MCQ
# -----------------------
def select_next_mcq(mcq_pool: List[Dict[str, Any]],
                    theta: float,
                    topics_asked: Dict[str, int],
                    asked_ids: set) -> Dict[str, Any]:
    desired_diff = target_difficulty(theta)
    available_topics = list({q["topic"] for q in mcq_pool})
    target_topic = select_topic_least_asked(topics_asked, available_topics)

    # Filter out already asked items
    candidates_pool = [ (i,q) for i,q in enumerate(mcq_pool) if i not in asked_ids ]

    # preference 1: difficulty + topic
    candidates = [ (i,q) for (i,q) in candidates_pool if q.get("difficulty")==desired_diff and q.get("topic")==target_topic ]
    if candidates:
        i,q = random.choice(candidates)
        return {"index": i, "mcq": q}

    # preference 2: difficulty only
    candidates = [ (i,q) for (i,q) in candidates_pool if q.get("difficulty")==desired_diff ]
    if candidates:
        i,q = random.choice(candidates)
        return {"index": i, "mcq": q}

    # preference 3: topic only
    candidates = [ (i,q) for (i,q) in candidates_pool if q.get("topic")==target_topic ]
    if candidates:
        i,q = random.choice(candidates)
        return {"index": i, "mcq": q}

    # fallback: any not-asked
    if candidates_pool:
        i,q = random.choice(candidates_pool)
        return {"index": i, "mcq": q}

    # if everything exhausted, reset asked_ids and pick random from whole pool
    asked_ids.clear()
    i,q = random.choice(list(enumerate(mcq_pool)))
    return {"index": i, "mcq": q}

# -----------------------
# Simulate correctness using logistic model
# -----------------------
def simulate_correctness(theta: float, difficulty_label: str) -> bool:
    b = DIFF_TO_B.get(difficulty_label, 0.0)
    x = LOGISTIC_SCALE * (theta - b)
    p = sigmoid(x)
    r = random.random()
    # debug print included where function used
    return r < p

# -----------------------
# Theta update (simple)
# -----------------------
def update_theta_simple(theta: float, correct: bool) -> float:
    old = theta
    theta = theta + THETA_STEP if correct else theta - THETA_STEP
    theta = clamp_theta(theta)
    print(f"[DEBUG] θ updated: {old:.2f} -> {theta:.2f} (correct={correct})")
    return theta

# -----------------------
# Run adaptive session
# -----------------------
def run_adaptive_session(mcqs: List[Dict[str, Any]],
                         steps: int = 10,
                         simulate: bool = True,
                         sim_seed: Optional[int] = None) -> Dict[str, Any]:
    if sim_seed is not None:
        random.seed(sim_seed)

    state = {
        "theta": 0.0,
        "history": [],       # list of dicts per item
        "topics_asked": {},
        "asked_ids": set()
    }

    # quick map of topics
    all_topics = sorted({q["topic"] for q in mcqs})

    print("\n==================== ADAPTIVE SESSION START ====================\n")
    for step in range(steps):
        print(f"--- STEP {step+1} ---")
        pick = select_next_mcq(mcqs, state["theta"], state["topics_asked"], state["asked_ids"])
        idx = pick["index"]
        q = pick["mcq"]

        # mark asked
        state["asked_ids"].add(idx)
        topic = q.get("topic", "misc")
        diff = q.get("difficulty", "Medium")

        # show question
        print(f"[DEBUG] Selected (idx={idx}) from {q.get('source_file')}, topic='{topic}', difficulty='{diff}'")
        print("Q:", q.get("stem"))
        for k,v in q.get("options", {}).items():
            print(f"  {k}: {v}")

        # Get correctness: interactive or simulate
        if simulate:
            correct = simulate_correctness(state["theta"], diff)
            print(f"[DEBUG] Simulated correctness probability model -> θ={state['theta']:.2f}, diff_b={DIFF_TO_B.get(diff):.2f}, selected={'Correct' if correct else 'Wrong'}")
        else:
            # interactive input
            student_answer = input("Type your answer (A/B/C/D) or 'skip': ").strip().upper()
            if student_answer == "SKIP" or student_answer == "":
                print("[DEBUG] Student skipped question -> treated as incorrect")
                correct = False
            else:
                correct = (student_answer == q.get("answer"))
                print(f"[DEBUG] Student answered '{student_answer}', correct option = '{q.get('answer')}' -> {'Correct' if correct else 'Wrong'}")

        # Update theta and topic counts
        state["theta"] = update_theta_simple(state["theta"], correct)
        state["topics_asked"][topic] = state["topics_asked"].get(topic, 0) + 1

        # record history
        state["history"].append({
            "step": step+1,
            "idx": idx,
            "source_file": q.get("source_file"),
            "topic": topic,
            "difficulty": diff,
            "stem": q.get("stem"),
            "student_correct": bool(correct),
            "theta_after": state["theta"]
        })

        print(f"[DEBUG] Topic counts: {state['topics_asked']}\n")

    print("\n==================== ADAPTIVE SESSION END ====================\n")
    print("Final θ:", state["theta"])
    print("Topic distribution:", state["topics_asked"])
    print("History length:", len(state["history"]))

    # save session
    out = {
        "final_theta": state["theta"],
        "topics_asked": state["topics_asked"],
        "history": state["history"]
    }
    Path("adaptive_session.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("[DEBUG] Session saved to 'adaptive_session.json'")

    return out

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Adaptive engine (interactive or simulated)")
    p.add_argument("--mcq-folder", type=str, default="MCQ_output", help="Folder containing MCQ JSON files")
    p.add_argument("--steps", type=int, default=10, help="Number of items in adaptive session")
    p.add_argument("--simulate", action="store_true", help="Simulate student responses (default behavior)")
    p.add_argument("--interactive", action="store_true", help="Ask user for answers interactively")
    p.add_argument("--seed", type=int, default=None, help="Random seed for simulation reproducibility")
    return p.parse_args()

# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    args = parse_args()
    mcqs = load_all_mcqs(args.mcq_folder)
    if not mcqs:
        print("[ERROR] No MCQs loaded. Exiting.")
        exit(1)

    # prefer interactive if requested
    simulate_mode = True
    if args.interactive:
        simulate_mode = False
    elif args.simulate:
        simulate_mode = True

    session = run_adaptive_session(mcqs, steps=args.steps, simulate=simulate_mode, sim_seed=args.seed)
    print("\nSession summary saved to adaptive_session.json")
