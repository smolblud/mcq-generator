"""
evaluation_metrics.py

Computes extended evaluation metrics for MCQ data:
- Faithfulness (match to correct answers)
- Topic coverage vs blueprint (only generated topics)
- Adaptive gains per student
- Latency/Cost estimation (mocked)
- Saves results to JSON and CSV
"""

import pandas as pd
import numpy as np
import time
import argparse
import os
import glob
import json
import random

# ----------------------------
# Load MCQs from folder
# ----------------------------
def load_mcqs(mcq_folder):
    mcq_files = glob.glob(os.path.join(mcq_folder, "*.json"))
    mcqs = []
    for f in mcq_files:
        with open(f, 'r', encoding='utf-8') as jf:
            data = json.load(jf)
            for i, q in enumerate(data):
                q['item_id'] = f"{os.path.splitext(os.path.basename(f))[0]}_{i+1}"
                if 'topic' not in q:
                    q['topic'] = os.path.splitext(os.path.basename(f))[0]  # filename as topic
                mcqs.append(q)
    return mcqs

def build_items_df(mcqs):
    items_data = []
    for q in mcqs:
        items_data.append({
            "item_id": q['item_id'],
            "topic": q.get('topic', 'Unknown'),
            "correct_option": q['answer']
        })
    return pd.DataFrame(items_data)

# ----------------------------
# Load Responses (or simulate)
# ----------------------------
def load_or_simulate_responses(items_df, responses_file=None, n_students=50):
    if responses_file and os.path.exists(responses_file):
        responses_df = pd.read_csv(responses_file)
    else:
        responses_data = []
        for s_idx in range(n_students):
            student_id = f"student_{s_idx+1}"
            for _, row in items_df.iterrows():
                correct = row['correct_option']
                if random.random() < 0.7:
                    selected = correct
                else:
                    options = ['A', 'B', 'C', 'D']
                    options = [o for o in options if o != correct]
                    selected = random.choice(options)
                responses_data.append({
                    "student_id": student_id,
                    "item_id": row['item_id'],
                    "selected_option": selected
                })
        responses_df = pd.DataFrame(responses_data)
    return responses_df

# ----------------------------
# Load Blueprint (optional)
# ----------------------------
def load_blueprint(blueprint_file=None):
    blueprint_dict = {}
    if blueprint_file and os.path.exists(blueprint_file):
        bp = pd.read_csv(blueprint_file)
        for _, row in bp.iterrows():
            topic = row['topic']
            blueprint_dict[topic] = blueprint_dict.get(topic, 0) + 1
    return blueprint_dict

# ----------------------------
# Faithfulness
# ----------------------------
def compute_faithfulness(items_df, responses_df):
    merged = responses_df.merge(items_df[['item_id', 'correct_option']], on='item_id')
    merged['is_correct'] = merged['selected_option'] == merged['correct_option']
    return merged['is_correct'].mean()

# ----------------------------
# Topic Coverage (only generated topics)
# ----------------------------
def compute_topic_coverage(items_df, blueprint_dict=None):
    counts = items_df['topic'].value_counts().to_dict()
    coverage_data = []

    all_topics = set(counts.keys())
    if blueprint_dict:
        all_topics |= set(blueprint_dict.keys())

    for topic in all_topics:
        actual = counts.get(topic, 0)
        expected = blueprint_dict.get(topic, actual) if blueprint_dict else actual
        pct = (actual / expected * 100) if expected > 0 else (100 if actual > 0 else 0)
        status = "OK"
        if actual < expected:
            status = f"Missing {expected - actual}"
        elif actual > expected:
            status = f"Excess {actual - expected}"
        coverage_data.append({
            "topic": topic,
            "actual_count": actual,
            "expected_count": expected,
            "coverage_pct": pct,
            "status": status
        })
    return pd.DataFrame(coverage_data).sort_values('topic')

# ----------------------------
# Adaptive Gains
# ----------------------------
def compute_adaptive_gains(responses_df, items_df):
    gains = []
    for student_id, group in responses_df.groupby('student_id'):
        group = group.sort_values('item_id')
        first = group.iloc[0]
        last = group.iloc[-1]
        first_correct = int(first['selected_option'] == items_df.loc[items_df['item_id']==first['item_id'], 'correct_option'].values[0])
        last_correct = int(last['selected_option'] == items_df.loc[items_df['item_id']==last['item_id'], 'correct_option'].values[0])
        gains.append(last_correct - first_correct)
    return np.mean(gains)

# ----------------------------
# Latency/Cost (mocked)
# ----------------------------
def compute_latency_cost(responses_df):
    latency_per_response = 1.0
    cost_per_response = 0.01
    n_responses = len(responses_df)
    return n_responses * latency_per_response, n_responses * cost_per_response

# ----------------------------
# Main Evaluation
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcq_folder", default="MCQ_output", help="Folder with MCQ JSON files")
    parser.add_argument("--responses", default=None, help="CSV with student responses (optional)")
    parser.add_argument("--blueprint", default=None, help="CSV file with topic blueprint (optional)")
    parser.add_argument("--n_students", type=int, default=50, help="Number of simulated students if no responses file")
    parser.add_argument("--save_path", default="evaluation_results", help="Folder to save results")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    start_time = time.time()

    mcqs = load_mcqs(args.mcq_folder)
    items_df = build_items_df(mcqs)
    responses_df = load_or_simulate_responses(items_df, args.responses, args.n_students)
    blueprint_dict = load_blueprint(args.blueprint)

    faithfulness = compute_faithfulness(items_df, responses_df)
    coverage_df = compute_topic_coverage(items_df, blueprint_dict)
    adaptive_gain = compute_adaptive_gains(responses_df, items_df)
    latency, cost = compute_latency_cost(responses_df)

    # ----------------------------
    # Print results
    # ----------------------------
    print(f"\nFaithfulness: {faithfulness:.3f}\n")
    print("Topic Coverage:\n", coverage_df, "\n")
    print(f"Average Adaptive Gain per student: {adaptive_gain:.3f}\n")
    print(f"Estimated Total Latency: {latency:.2f} sec")
    print(f"Estimated Cost: ${cost:.2f}")

    elapsed = time.time() - start_time
    print(f"\nEvaluation finished in {elapsed:.2f} sec")

    # ----------------------------
    # Save results
    # ----------------------------
    coverage_csv_path = os.path.join(args.save_path, "topic_coverage.csv")
    coverage_df.to_csv(coverage_csv_path, index=False)

    results_summary = {
        "faithfulness": faithfulness,
        "average_adaptive_gain": adaptive_gain,
        "estimated_latency_sec": latency,
        "estimated_cost_usd": cost,
        "total_mcqs": len(items_df),
        "total_responses": len(responses_df),
    }

    results_json_path = os.path.join(args.save_path, "evaluation_summary.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\nResults saved to:\n- {coverage_csv_path}\n- {results_json_path}")


if __name__ == "__main__":
    main()
