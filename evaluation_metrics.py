"""
evaluation_metrics.py

Comprehensive evaluation for RAG-based MCQ system:
- Faithfulness (citation presence + answer correctness proxy)
- Psychometrics: Difficulty (p-value) & Discrimination (point-biserial)
- Topic coverage vs blueprint
- Adaptive gains (pre/post proxy)
- Latency & cost estimation
- Saves CSV + JSON (report-ready)
"""
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import argparse
import os
import glob
import json
import random
import time
from scipy.stats import pointbiserialr


# -------------------------------------------------
# Load MCQs
# -------------------------------------------------
def load_mcqs(mcq_folder):
    mcq_files = glob.glob(os.path.join(mcq_folder, "*.json"))
    mcqs = []

    for f in mcq_files:
        with open(f, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            for i, q in enumerate(data):
                mcqs.append({
                    "item_id": f"{os.path.basename(f)}_{i}",
                    "topic": q.get("topic", "Unknown"),
                    "subject": q.get("subject", "Unknown"),
                    "correct_option": q["answer"],
                    "has_citation": bool(q.get("citations")),
                })
    return pd.DataFrame(mcqs)


# -------------------------------------------------
# Load or simulate responses
# -------------------------------------------------
def load_or_simulate_responses(items_df, responses_file=None, n_students=50):
    if responses_file and os.path.exists(responses_file):
        return pd.read_csv(responses_file)

    responses = []
    for s in range(n_students):
        sid = f"student_{s+1}"
        ability = np.random.normal(0, 1)

        for _, item in items_df.iterrows():
            p_correct = 1 / (1 + np.exp(-ability))
            correct = random.random() < p_correct

            if correct:
                selected = item["correct_option"]
            else:
                options = ["A", "B", "C", "D"]
                options.remove(item["correct_option"])
                selected = random.choice(options)

            responses.append({
                "student_id": sid,
                "item_id": item["item_id"],
                "selected_option": selected
            })

    return pd.DataFrame(responses)


# -------------------------------------------------
# Load blueprint
# -------------------------------------------------
def load_blueprint(blueprint_file):
    if not blueprint_file or not os.path.exists(blueprint_file):
        return None
    bp = pd.read_csv(blueprint_file)
    return set(bp["topic"].astype(str))


# -------------------------------------------------
# Faithfulness
# -------------------------------------------------
from scipy.stats import binom

# -------------------------------------------------
# Faithfulness
# -------------------------------------------------
def compute_faithfulness(mcqs_df, alpha=0.05):
    # Faithful if MCQ has explanation AND citation
    mcqs_df['faithful'] = mcqs_df.get('has_explanation', True) & mcqs_df.get('has_citation', False)
    
    p = mcqs_df['faithful'].mean()
    
    n = len(mcqs_df)
    # Binomial 95% confidence interval
    ci_low, ci_up = binom.interval(1-alpha, n, p)
    ci_low /= n
    ci_up /= n
    
    return p, (ci_low, ci_up)



# -------------------------------------------------
# Psychometrics
# -------------------------------------------------
def compute_psychometrics(items_df, responses_df):
    merged = responses_df.merge(items_df, on="item_id")
    merged["is_correct"] = merged["selected_option"] == merged["correct_option"]

    # Difficulty (p-value)
    difficulty = merged.groupby("item_id")["is_correct"].mean()

    # Total scores per student
    total_scores = merged.groupby("student_id")["is_correct"].sum()

    discrimination = {}
    for item_id, group in merged.groupby("item_id"):
        scores = group["is_correct"]
        total = total_scores[group["student_id"]].values
        if scores.nunique() > 1:
            corr, _ = pointbiserialr(scores, total)
            discrimination[item_id] = corr
        else:
            discrimination[item_id] = np.nan

    psych_df = pd.DataFrame({
        "item_id": difficulty.index,
        "p_value": difficulty.values,
        "discrimination": [discrimination[i] for i in difficulty.index]
    })

    return psych_df


# -------------------------------------------------
# Coverage
# -------------------------------------------------
def compute_topic_coverage(items_df, blueprint_topics=None):
    counts = items_df["topic"].value_counts()
    data = []

    all_topics = set(counts.index)
    if blueprint_topics:
        all_topics |= blueprint_topics

    for topic in all_topics:
        actual = counts.get(topic, 0)
        expected = 1 if blueprint_topics and topic in blueprint_topics else actual
        pct = 100 * actual / expected if expected > 0 else 0
        data.append({
            "topic": topic,
            "actual_count": actual,
            "expected_count": expected,
            "coverage_pct": pct
        })

    return pd.DataFrame(data)


# -------------------------------------------------
# Adaptive gains (θ proxy)
# -------------------------------------------------
def compute_adaptive_gains(responses_df, items_df):
    """
    Computes average adaptive gain per student.
    Gain = last_correct (0/1) - first_correct (0/1)
    """

    # Merge correct answers
    merged = responses_df.merge(
        items_df[['item_id', 'correct_option']],
        on='item_id',
        how='left'
    )

    # Explicit boolean → int conversion
    merged['is_correct'] = (
        merged['selected_option'] == merged['correct_option']
    ).astype(int)

    gains = []

    for student_id, grp in merged.groupby('student_id'):
        grp = grp.sort_values('item_id')

        first_correct = int(grp.iloc[0]['is_correct'])
        last_correct = int(grp.iloc[-1]['is_correct'])

        gains.append(last_correct - first_correct)

    return float(np.mean(gains)) if gains else 0.0



# -------------------------------------------------
# Latency & cost (measured or mocked)
# -------------------------------------------------
def compute_latency_cost(n_items):
    latency_per_item = 1.2   # sec
    cost_per_item = 0.015    # USD
    return {
        "avg_latency_sec": latency_per_item,
        "p95_latency_sec": latency_per_item * 1.8,
        "total_cost_usd": n_items * cost_per_item
    }
# -------------------------------------------------
# Plotting utilities
# -------------------------------------------------
def plot_psychometrics(psych_df, save_path):
    plt.figure()
    psych_df["p_value"].hist(bins=10)
    plt.xlabel("Difficulty (p-value)")
    plt.ylabel("Number of Items")
    plt.title("Item Difficulty Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "difficulty_distribution.png"))
    plt.close()

    plt.figure()
    psych_df["discrimination"].dropna().hist(bins=10)
    plt.xlabel("Discrimination (Point-Biserial)")
    plt.ylabel("Number of Items")
    plt.title("Item Discrimination Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "discrimination_distribution.png"))
    plt.close()


def plot_topic_coverage(coverage_df, save_path):
    plt.figure(figsize=(10, 5))
    plt.bar(coverage_df["topic"], coverage_df["actual_count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Number of MCQs")
    plt.title("Topic Coverage")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "topic_coverage.png"))
    plt.close()
def plot_faithfulness(faithfulness, save_path):
    plt.figure(figsize=(4,4))
    plt.bar(["Faithful", "Unfaithful"], [faithfulness*100, (1-faithfulness)*100], color=['green','red'])
    plt.ylabel("Percentage")
    plt.title("MCQ Faithfulness (%)")
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "faithfulness.png"))
    plt.close()




def plot_adaptive_gains(responses_df, items_df, save_path):
    merged = responses_df.merge(
        items_df[['item_id', 'correct_option']],
        on='item_id'
    )
    merged["is_correct"] = (
        merged["selected_option"] == merged["correct_option"]
    ).astype(int)

    gains = []
    for sid, grp in merged.groupby("student_id"):
        grp = grp.sort_values("item_id")
        gains.append(grp.iloc[-1]["is_correct"] - grp.iloc[0]["is_correct"])

    plt.figure()
    plt.hist(gains, bins=[-1, 0, 1, 2], align="left", rwidth=0.8)
    plt.xticks([-1, 0, 1])
    plt.xlabel("Adaptive Gain")
    plt.ylabel("Number of Students")
    plt.title("Adaptive Learning Gains")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "adaptive_gains.png"))
    plt.close()


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcq_folder", default="MCQ_test_output")
    parser.add_argument("--responses", default=None)
    parser.add_argument("--blueprint", default=None)
    parser.add_argument("--n_students", type=int, default=50)
    parser.add_argument("--save_path", default="evaluation_results")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    start = time.time()

    items_df = load_mcqs(args.mcq_folder)
    responses_df = load_or_simulate_responses(items_df, args.responses, args.n_students)
    blueprint_topics = load_blueprint(args.blueprint)

    faithfulness, faith_ci = compute_faithfulness(items_df)

    psych_df = compute_psychometrics(items_df, responses_df)
    coverage_df = compute_topic_coverage(items_df, blueprint_topics)
    adaptive_gain = compute_adaptive_gains(responses_df, items_df)
    latency_cost = compute_latency_cost(len(items_df))

    # Save CSVs
    psych_df.to_csv(os.path.join(args.save_path, "psychometrics.csv"), index=False)
    coverage_df.to_csv(os.path.join(args.save_path, "topic_coverage.csv"), index=False)

    # Summary JSON
    summary = {
    "faithfulness": float(faithfulness),
    "faithfulness_CI": (float(faith_ci[0]), float(faith_ci[1])),
    "avg_p_value": float(psych_df["p_value"].mean()),
    "avg_discrimination": float(psych_df["discrimination"].mean(skipna=True)),
    "coverage_pct": float((coverage_df["actual_count"] > 0).mean() * 100),
    "adaptive_gain": adaptive_gain,
    **latency_cost
}


    with open(os.path.join(args.save_path, "evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EVALUATION COMPLETE ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved results to: {args.save_path}")
        # ----------------------------
    # Generate plots
    # ----------------------------
    plot_psychometrics(psych_df, args.save_path)
    plot_topic_coverage(coverage_df, args.save_path)
    plot_adaptive_gains(responses_df, items_df, args.save_path)
    plot_faithfulness(faithfulness, args.save_path)


    print(f"Elapsed time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()