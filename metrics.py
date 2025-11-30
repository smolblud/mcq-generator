"""
metrics.py

Standalone module for computing pilot-test metrics for MCQ items.
Includes calculation of p-values (difficulty), discrimination (point-biserial),
and topic coverage.

Assumes input data is provided as pandas DataFrames or lists of dictionaries.

commit: Add preprocessing, CLI, NaN-handling, plot saving, and small-n flags to metrics.py
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys
from typing import List, Dict, Union, Optional, Any

# -----------------------------------------------------------------------------
# Data Validation & Preparation
# -----------------------------------------------------------------------------

def ensure_dataframe(data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.DataFrame:
    """
    Ensures the input data is a pandas DataFrame.
    """
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        return data.copy()
    else:
        raise ValueError("Data must be a pandas DataFrame or a list of dictionaries.")

def validate_response_data(responses_df: pd.DataFrame) -> None:
    """
    Validates that the response DataFrame has the necessary columns.
    Required columns: 'item_id', 'student_id', 'is_correct'
    """
    required_cols = {'item_id', 'student_id', 'is_correct'}
    if not required_cols.issubset(responses_df.columns):
        missing = required_cols - set(responses_df.columns)
        raise ValueError(f"Response data missing required columns: {missing}")

def map_responses_to_is_correct(items_df: pd.DataFrame, responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps raw responses to binary correctness (0/1).
    
    Inputs:
      items_df: DataFrame with 'item_id', 'correct_option', optional 'options'.
      responses_df: DataFrame with respondent_id/student_id, 'item_id', 'selected_option'.
      
    Returns:
      DataFrame with 'student_id', 'item_id', 'is_correct'.
    """
    items_df = ensure_dataframe(items_df)
    responses_df = ensure_dataframe(responses_df)
    
    # 1. Detect respondent column
    resp_cols = responses_df.columns
    student_col = None
    for candidate in ['student_id', 'respondent_id', 'user_id']:
        if candidate in resp_cols:
            student_col = candidate
            break
    
    if not student_col:
        raise ValueError("Responses must have 'student_id', 'respondent_id', or 'user_id' column.")
    
    # Normalize to 'student_id'
    out_df = responses_df.rename(columns={student_col: 'student_id'})[['student_id', 'item_id', 'selected_option']].copy()
    
    # Drop missing responses
    out_df.dropna(subset=['selected_option'], inplace=True)
    
    # Merge with correct answer
    # Ensure items_df has item_id
    if 'item_id' not in items_df.columns:
        raise ValueError("Items DataFrame must have 'item_id'.")
        
    # Prepare items lookup
    # We need correct_option. If not present, we can't score.
    if 'correct_option' not in items_df.columns:
        raise ValueError("Items DataFrame must have 'correct_option'.")
        
    items_lookup = items_df.set_index('item_id')[['correct_option']]
    if 'options' in items_df.columns:
        items_lookup['options'] = items_df.set_index('item_id')['options']
    
    # Join
    merged = out_df.merge(items_lookup, on='item_id', how='left')
    
    def check_correct(row):
        selected = row['selected_option']
        correct = row['correct_option']
        
        if pd.isna(correct):
            return np.nan # Cannot score
            
        # If selected is numeric
        if isinstance(selected, (int, float)):
            sel_int = int(selected)
            # Try 0-based match to A, B, C...
            # A=0, B=1...
            # If correct is "A", and selected is 0 -> Match
            # If correct is "A", and selected is 1 -> Mismatch (unless 1-based A?)
            # Heuristic: Convert numeric to letter (0->A) and compare
            # Also try 1->A
            
            # 0-based mapping
            char_0 = chr(ord('A') + sel_int)
            # 1-based mapping
            char_1 = chr(ord('A') + sel_int - 1) if sel_int > 0 else None
            
            if str(correct).upper() == char_0:
                return 1
            if char_1 and str(correct).upper() == char_1:
                return 1
            
            # If correct is also numeric (unlikely but possible)
            try:
                if float(correct) == float(selected):
                    return 1
            except:
                pass
                
            return 0
            
        # If selected is text
        sel_str = str(selected).strip()
        corr_str = str(correct).strip()
        
        # Direct match
        if sel_str.upper() == corr_str.upper():
            return 1
            
        # Check if selected text matches option text corresponding to correct option
        # (Not fully implemented as 'options' structure varies, assuming simple match for now)
        
        return 0

    merged['is_correct'] = merged.apply(check_correct, axis=1)
    
    # Drop unscoreable
    merged.dropna(subset=['is_correct'], inplace=True)
    merged['is_correct'] = merged['is_correct'].astype(int)
    
    return merged[['student_id', 'item_id', 'is_correct']]

# -----------------------------------------------------------------------------
# Metric Calculations
# -----------------------------------------------------------------------------

def calculate_p_values(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes p-values (difficulty) for each item.
    p_value = percentage of respondents who answered correctly.
    
    Returns a DataFrame with index 'item_id' and columns:
    - p_value: float
    - n_responses: int
    - difficulty_flag: str (comment if too easy/hard)
    """
    validate_response_data(responses_df)
    
    # Group by item_id and calculate mean of is_correct
    stats = responses_df.groupby('item_id')['is_correct'].agg(['mean', 'count'])
    stats.rename(columns={'mean': 'p_value', 'count': 'n_responses'}, inplace=True)
    
    def flag_difficulty(row):
        p = row['p_value']
        n = row['n_responses']
        
        flags = []
        
        if n < 5:
            # Set p to NaN if n is too small? 
            # Prompt says: set p_value = np.nan and add comment
            return "small n — interpret with caution"
            
        if p > 0.9:
            flags.append("Too Easy (>0.9)")
        elif p < 0.2:
            flags.append("Too Hard (<0.2)")
            
        if p == 0.0 or p == 1.0:
            flags.append("possible key issue")
            
        return "; ".join(flags)
    
    # Apply logic to set NaN and flags
    # We need to do this carefully. If we set p to NaN first, we lose the 0/1 check.
    # So calculate flag first.
    stats['difficulty_flag'] = stats.apply(flag_difficulty, axis=1)
    
    # Now set NaN for small n
    stats.loc[stats['n_responses'] < 5, 'p_value'] = np.nan
    
    return stats

def calculate_discrimination(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes discrimination (point-biserial correlation) for each item.
    Correlation between item correctness (0/1) and total test score EXCLUDING that item.
    
    Returns a DataFrame with index 'item_id' and columns:
    - point_biserial: float
    - n_pairs_used: int
    - discrimination_flag: str
    """
    validate_response_data(responses_df)
    
    # Calculate total score for each student
    student_scores = responses_df.groupby('student_id')['is_correct'].sum()
    
    item_stats = {}
    
    for item_id, group in responses_df.groupby('item_id'):
        # group contains responses for this item
        # join with student_scores
        merged = group.merge(student_scores.rename('total_score'), on='student_id')
        
        # adjusted score = total_score - is_correct (for that item)
        merged['adjusted_score'] = merged['total_score'] - merged['is_correct']
        
        n_pairs = len(merged)
        
        # Calculate correlation
        if n_pairs < 2 or merged['adjusted_score'].std() == 0 or merged['is_correct'].std() == 0:
            corr = np.nan # Undefined correlation
        else:
            corr = merged['is_correct'].corr(merged['adjusted_score'])
            
        item_stats[item_id] = {'point_biserial': corr, 'n_pairs_used': n_pairs}
        
    stats = pd.DataFrame.from_dict(item_stats, orient='index')
    if stats.empty:
         stats = pd.DataFrame(columns=['point_biserial', 'n_pairs_used'])
    stats.index.name = 'item_id'
    
    def flag_discrimination(row):
        r = row['point_biserial']
        n = row['n_pairs_used']
        
        if n < 5:
            return "small n — interpret with caution"
            
        if pd.isna(r):
            return "Insufficient Data"
            
        if r < 0:
            return "negative discrimination — check key or ambiguity"
        elif r < 0.2:
            return "Low Discrimination (<0.2)"
        return ""
        
    stats['discrimination_flag'] = stats.apply(flag_discrimination, axis=1)
    
    # Set NaN for small n
    stats.loc[stats['n_pairs_used'] < 5, 'point_biserial'] = np.nan
    
    return stats

def calculate_topic_coverage(items_df: pd.DataFrame, blueprint: Dict[str, int]) -> pd.DataFrame:
    """
    Computes topic coverage relative to a blueprint.
    
    items_df: DataFrame containing 'item_id' and 'topic'.
    blueprint: Dictionary {topic: expected_count}
    """
    if 'topic' not in items_df.columns:
        raise ValueError("Items DataFrame must contain 'topic' column.")
        
    counts = items_df['topic'].value_counts().to_dict()
    
    coverage_data = []
    all_topics = set(blueprint.keys()) | set(counts.keys())
    
    for topic in all_topics:
        actual = counts.get(topic, 0)
        expected = blueprint.get(topic, 0)
        diff = actual - expected
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

# -----------------------------------------------------------------------------
# Summary & Visualization
# -----------------------------------------------------------------------------

def generate_metrics_summary(responses_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines all metrics into a single summary DataFrame.
    """
    responses_df = ensure_dataframe(responses_df)
    items_df = ensure_dataframe(items_df)
    
    # Calculate metrics
    p_values = calculate_p_values(responses_df)
    discrimination = calculate_discrimination(responses_df)
    
    # Merge metrics
    metrics = p_values.join(discrimination, how='outer')
    
    # Merge with item metadata (topic)
    # Ensure items_df has item_id as index or column
    if 'item_id' in items_df.columns:
        items_indexed = items_df.set_index('item_id')
    else:
        items_indexed = items_df
    
    final_df = metrics.join(items_indexed[['topic']], how='left')
    
    # Combine flags
    def combine_flags(row):
        flags = []
        if pd.notna(row.get('difficulty_flag')) and row['difficulty_flag']:
            flags.append(row['difficulty_flag'])
        if pd.notna(row.get('discrimination_flag')) and row['discrimination_flag']:
            flags.append(row['discrimination_flag'])
        return "; ".join(flags)
    
    final_df['comments'] = final_df.apply(combine_flags, axis=1)
    
    # Reorder columns
    cols = ['topic', 'p_value', 'point_biserial', 'n_responses', 'n_pairs_used', 'comments']
    # Ensure all cols exist
    for c in cols:
        if c not in final_df.columns:
            final_df[c] = np.nan
            
    return final_df[cols].reset_index()

def plot_metrics(summary_df: pd.DataFrame, out_dir: Optional[str] = None):
    """
    Generates simple plots for p-values and discrimination.
    Uses matplotlib if available, otherwise prints text histograms.
    If out_dir is provided, saves plots to out_dir/figures.
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create output directory if needed
        if out_dir:
            fig_dir = os.path.join(out_dir, "figures")
            os.makedirs(fig_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # P-values histogram
        p_vals = summary_df['p_value'].dropna()
        if not p_vals.empty:
            axes[0].hist(p_vals, bins=10, color='skyblue', edgecolor='black')
        axes[0].set_title('Item Difficulty (p-value) Distribution')
        axes[0].set_xlabel('p-value')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(0.2, color='red', linestyle='--', label='Too Hard (<0.2)')
        axes[0].axvline(0.9, color='green', linestyle='--', label='Too Easy (>0.9)')
        axes[0].legend()
        
        # Discrimination histogram
        disc_vals = summary_df['point_biserial'].dropna()
        if not disc_vals.empty:
            axes[1].hist(disc_vals, bins=10, color='salmon', edgecolor='black')
        axes[1].set_title('Item Discrimination (Point-Biserial) Distribution')
        axes[1].set_xlabel('Correlation')
        axes[1].axvline(0.2, color='red', linestyle='--', label='Low Disc. (<0.2)')
        axes[1].axvline(0.0, color='black', linestyle='-', linewidth=1)
        axes[1].legend()
        
        plt.tight_layout()
        
        if out_dir:
            # Save individual plots or the combined one?
            # Prompt says: save figures/p_values.png and figures/discrimination.png
            # I'll save the combined one as 'metrics_overview.png' and individual ones as requested.
            # Actually, let's just save the combined one to save time, or split them.
            # Prompt is specific: "save figures/p_values.png", "figures/discrimination.png".
            
            # Save combined first (optional but good)
            # plt.savefig(os.path.join(fig_dir, "metrics_overview.png"))
            
            # Save P-values
            extent = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(fig_dir, "p_values.png"), bbox_inches=extent.expanded(1.2, 1.2))
            
            # Save Discrimination
            extent = axes[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(os.path.join(fig_dir, "discrimination.png"), bbox_inches=extent.expanded(1.2, 1.2))
            
            print(f"Plots saved to {fig_dir}")
        
        # plt.show() # Blocking call, skipping for demo safety
        print("Plots generated successfully (skipping plt.show() to avoid blocking).")
        plt.close(fig)
        
    except ImportError:
        print("\n[INFO] Matplotlib not found. Showing text summaries instead.")
        print("\n--- P-Value Distribution ---")
        print(summary_df['p_value'].describe())
        print("\n--- Discrimination Distribution ---")
        print(summary_df['point_biserial'].describe())

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCQ Metrics Calculation")
    parser.add_argument("--mcq", help="Path to MCQ items CSV/JSON (must have item_id, correct_option, topic)")
    parser.add_argument("--responses", required=True, help="Path to Responses CSV/JSON")
    parser.add_argument("--blueprint", help="Path to Blueprint text file (topic per line or topic: count)")
    parser.add_argument("--out", default="outputs", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output dir
    os.makedirs(args.out, exist_ok=True)
    
    # Load Responses
    print(f"Loading responses from {args.responses}...")
    if args.responses.endswith('.json'):
        responses_df = pd.read_json(args.responses)
    else:
        responses_df = pd.read_csv(args.responses)
        
    # Load Items (if provided)
    items_df = None
    if args.mcq:
        print(f"Loading items from {args.mcq}...")
        if args.mcq.endswith('.json'):
            items_df = pd.read_json(args.mcq)
        else:
            items_df = pd.read_csv(args.mcq)
            
        # Preprocess
        print("Mapping responses to correctness...")
        try:
            responses_df = map_responses_to_is_correct(items_df, responses_df)
        except Exception as e:
            print(f"Error mapping responses: {e}")
            sys.exit(1)
    else:
        print("No items file provided. Assuming responses contain 'is_correct' and 'student_id'.")
        # Create dummy items_df for topic merging if possible
        if 'item_id' in responses_df.columns:
            items_df = pd.DataFrame({'item_id': responses_df['item_id'].unique(), 'topic': 'Unknown'})
        else:
             items_df = pd.DataFrame(columns=['item_id', 'topic'])

    # Generate Summary
    print("Generating metrics summary...")
    summary = generate_metrics_summary(responses_df, items_df)
    
    # Save Summary
    out_path = os.path.join(args.out, "metrics_summary.csv")
    summary.to_csv(out_path, index=False)
    print(f"Summary saved to {out_path}")
    
    # Plot
    print("Generating plots...")
    plot_metrics(summary, out_dir=args.out)
    
    # Blueprint Report
    if args.blueprint:
        print(f"Processing blueprint from {args.blueprint}...")
        try:
            # Parse blueprint
            with open(args.blueprint, 'r') as f:
                lines = f.readlines()
            
            blueprint_dict = {}
            for line in lines:
                line = line.strip()
                if not line: continue
                if ':' in line:
                    k, v = line.split(':', 1)
                    blueprint_dict[k.strip()] = int(v.strip())
                else:
                    # Assume 1 per line
                    blueprint_dict[line] = blueprint_dict.get(line, 0) + 1
            
            coverage = calculate_topic_coverage(items_df, blueprint_dict)
            
            # Generate Report
            mean_p = summary['p_value'].mean()
            mean_r = summary['point_biserial'].mean()
            total_coverage = coverage['actual_count'].sum() / coverage['expected_count'].sum() * 100
            
            # Top 5 items to revise
            # Criteria: p < 0.2 (hard), p > 0.9 (easy), r < 0.2 (low disc)
            revise_mask = (summary['p_value'] < 0.2) | (summary['p_value'] > 0.9) | (summary['point_biserial'] < 0.2)
            revise_items = summary[revise_mask].head(5)['item_id'].tolist()
            
            report = f"""# Assignment 2 Report

- **Mean P-Value**: {mean_p:.2f}
- **Mean Point-Biserial**: {mean_r:.2f}
- **Topic Coverage**: {total_coverage:.1f}%

**Items to Revise (Top 5)**: {', '.join(map(str, revise_items)) if revise_items else 'None'}
"""
            report_path = os.path.join(args.out, "report_Assignment2.md")
            with open(report_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {report_path}")
            
        except Exception as e:
            print(f"Error processing blueprint: {e}")

if __name__ == "__main__":
    # If no args provided, run the old demo for backward compatibility/testing
    if len(sys.argv) == 1:
        print("No arguments provided. Running internal demo...")
        # Demo data
        items_data = [
            {'item_id': 'q1', 'topic': 'Math', 'correct_option': 'A'},
            {'item_id': 'q2', 'topic': 'Math', 'correct_option': 'B'},
            {'item_id': 'q3', 'topic': 'Science', 'correct_option': 'C'},
            {'item_id': 'q4', 'topic': 'Science', 'correct_option': 'D'},
            {'item_id': 'q5', 'topic': 'History', 'correct_option': 'A'},
        ]
        items_df = pd.DataFrame(items_data)
        
        responses_data = [
            {'student_id': 's1', 'item_id': 'q1', 'selected_option': 'A'}, # Correct
            {'student_id': 's1', 'item_id': 'q2', 'selected_option': 'B'}, # Correct
            {'student_id': 's1', 'item_id': 'q3', 'selected_option': 'C'}, # Correct
            {'student_id': 's1', 'item_id': 'q4', 'selected_option': 'A'}, # Wrong
            {'student_id': 's1', 'item_id': 'q5', 'selected_option': 'B'}, # Wrong
            
            {'student_id': 's2', 'item_id': 'q1', 'selected_option': 'A'},
            {'student_id': 's2', 'item_id': 'q2', 'selected_option': 'B'},
            {'student_id': 's2', 'item_id': 'q3', 'selected_option': 'A'},
            {'student_id': 's2', 'item_id': 'q4', 'selected_option': 'A'},
            {'student_id': 's2', 'item_id': 'q5', 'selected_option': 'A'},
            
            {'student_id': 's3', 'item_id': 'q1', 'selected_option': 'A'},
            {'student_id': 's3', 'item_id': 'q2', 'selected_option': 'A'},
            {'student_id': 's3', 'item_id': 'q3', 'selected_option': 'C'},
            {'student_id': 's3', 'item_id': 'q4', 'selected_option': 'A'},
            {'student_id': 's3', 'item_id': 'q5', 'selected_option': 'A'},
            
            {'student_id': 's4', 'item_id': 'q1', 'selected_option': 'A'},
            {'student_id': 's4', 'item_id': 'q2', 'selected_option': 'A'},
            {'student_id': 's4', 'item_id': 'q3', 'selected_option': 'A'},
            {'student_id': 's4', 'item_id': 'q4', 'selected_option': 'A'},
            {'student_id': 's4', 'item_id': 'q5', 'selected_option': 'A'},
        ]
        responses_df = pd.DataFrame(responses_data)
        
        print("\n--- Preprocessing ---")
        processed_resp = map_responses_to_is_correct(items_df, responses_df)
        print(processed_resp.head())
        
        print("\n--- Calculating Metrics ---")
        summary = generate_metrics_summary(processed_resp, items_df)
        print(summary.to_string())
        
        print("\n--- Plotting ---")
        plot_metrics(summary)
        
    else:
        main()
