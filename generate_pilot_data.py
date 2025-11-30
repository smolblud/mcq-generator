"""
generate_pilot_data.py

Simulates multiple student sessions using adaptive_engine.py to generate
pilot data (items and responses) for testing metrics.py.
"""

import pandas as pd
import random
import adaptive_engine
import sys
import os
from contextlib import contextmanager

# Suppress stdout to keep console clean during simulation
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def main():
    # Force UTF-8 for stdout to avoid encoding errors on Windows consoles
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            # Python < 3.7 or other environments
            pass

    print("Loading MCQs...")
    # Load all MCQs using the engine's function
    # Note: adaptive_engine.load_all_mcqs prints debug info, let it show.
    mcqs = adaptive_engine.load_all_mcqs("MCQ_output")
    
    if not mcqs:
        print("No MCQs found in MCQ_output. Exiting.")
        return

    print(f"Loaded {len(mcqs)} MCQs.")

    # 1. Generate Items Metadata (pilot_items.csv)
    print("Generating items metadata...")
    items_data = []
    for i, q in enumerate(mcqs):
        item_id = f"item_{i}"
        # Ensure we have an answer. If missing, skip or warn?
        # The engine might fail if answer is missing, but let's check.
        ans = q.get("answer")
        if not ans:
            ans = "A" # Fallback if missing, though it shouldn't be for valid MCQs
        
        items_data.append({
            "item_id": item_id,
            "topic": q.get("topic", "Unknown"),
            "correct_option": ans,
            "options": list(q.get("options", {}).keys())
        })
    
    items_df = pd.DataFrame(items_data)
    # We don't need the options list in the CSV for metrics.py, just correct_option
    items_df[["item_id", "topic", "correct_option"]].to_csv("pilot_items.csv", index=False)
    print("Saved pilot_items.csv")

    # 2. Simulate Students (pilot_responses.csv)
    n_students = 50
    steps_per_student = 20
    print(f"Simulating {n_students} students ({steps_per_student} steps each)...")
    
    responses_data = []
    
    for s_idx in range(n_students):
        student_id = f"student_{s_idx+1}"
        
        # Run session with suppressed stdout
        with suppress_stdout():
            # Random seed for variability
            seed = random.randint(0, 100000)
            session = adaptive_engine.run_adaptive_session(
                mcqs, 
                steps=steps_per_student, 
                simulate=True, 
                sim_seed=seed
            )
        
        # Process history
        for step in session["history"]:
            item_idx = step["idx"]
            is_correct = step["student_correct"]
            item_id = f"item_{item_idx}"
            
            # Get item details to synthesize selected_option
            # We need the correct option and distractor options
            # item_idx corresponds to index in mcqs list
            q = mcqs[item_idx]
            correct_opt = q.get("answer", "A")
            options_keys = list(q.get("options", {}).keys())
            
            if not options_keys:
                options_keys = ["A", "B", "C", "D"] # Fallback
            
            if is_correct:
                selected = correct_opt
            else:
                # Pick a wrong option
                distractors = [o for o in options_keys if o != correct_opt]
                if not distractors:
                    # If no distractors (e.g. only 1 option?), pick correct (shouldn't happen if wrong)
                    # Or just pick random from all
                    distractors = options_keys
                selected = random.choice(distractors)
            
            responses_data.append({
                "student_id": student_id,
                "item_id": item_id,
                "selected_option": selected
            })
            
        if (s_idx + 1) % 10 == 0:
            print(f"  Simulated {s_idx + 1} students...")

    responses_df = pd.DataFrame(responses_data)
    responses_df.to_csv("pilot_responses.csv", index=False)
    print(f"Saved pilot_responses.csv with {len(responses_df)} rows.")

if __name__ == "__main__":
    main()
