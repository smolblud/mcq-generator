import gradio as gr
import json
from typing import List, Dict, Any
from pathlib import Path
from mcq_generator import (
    generate_mcqs_for_topic,
    generate_mcqs_for_subject,
    load_topic_blueprint
)
from metrics import generate_metrics_summary, plot_metrics
from adaptive_engine import select_next_mcq, update_theta_simple


MCQ_FOLDER = "MCQ_final_output"

Path(MCQ_FOLDER).mkdir(exist_ok=True)


# ------------------------
# Load and format topics
# ------------------------
def load_all_topics():
    """Load all topics from blueprint and return as list of (subject, topic) tuples"""
    df = load_topic_blueprint()
    topics = []
    seen = set()
    for _, row in df.iterrows():
        subject = str(row["subject"]).strip()
        topic = str(row["topic"]).strip()
        key = (subject, topic)
        if key not in seen:
            seen.add(key)
            topics.append(key)
    return topics

def load_all_subjects():
    """Load unique subjects from blueprint"""
    df = load_topic_blueprint()
    return sorted(df["subject"].unique().tolist())


def format_topics_for_display(topics):
    """Format topics as a list of strings for dropdown/selection"""
    return [f"{subject} - {topic}" for subject, topic in topics]


def get_topic_from_selection(selected_topic_str):
    """Parse selected topic string back to (subject, topic) tuple"""
    if not selected_topic_str or " - " not in selected_topic_str:
        return None, None
    parts = selected_topic_str.split(" - ", 1)
    return parts[0].strip(), parts[1].strip()


# ------------------------
# Generate MCQs for selected topic
# ------------------------
def generate_mcqs_ui_wrapper(
    mode,               # "By Topic" or "By Subject"
    selected_subject,   # For Subject Mode
    selected_topic_str, # For Topic Mode
    num_questions,
    difficulty
):
    """
    Unified entry point for generating MCQs based on mode.
    Returns: (status_msg, list_of_questions)
    """
    num_q = int(num_questions) if num_questions else 5
    
    # --- SUBJECT MODE ---
    if mode == "By Subject":
        if not selected_subject:
            return "Please select a Subject.", []
        
        try:
            print(f"[GUI] Generating {num_q} MCQs for Subject: {selected_subject}")
            questions = generate_mcqs_for_subject(
                selected_subject, 
                n_questions=num_q, 
                difficulty=difficulty
            )
            context_label = selected_subject
        except Exception as e:
            import traceback
            return f"Error: {str(e)}\n{traceback.format_exc()}", []

    # --- TOPIC MODE ---
    else: 
        if not selected_topic_str:
            return "Please select a Topic.", []
        
        subject, topic = get_topic_from_selection(selected_topic_str)
        if not subject or not topic:
            return "Invalid topic selection.", []
            
        try:
            print(f"[GUI] Generating {num_q} MCQs for {subject} - {topic}")
            questions = generate_mcqs_for_topic(
                subject,
                topic,
                n_questions=num_q,
                difficulty=difficulty,
            )
            context_label = f"{subject} - {topic}"
        except Exception as e:
            import traceback
            return f"Error: {str(e)}\n{traceback.format_exc()}", []

    # --- VALIDATION (Common) ---
    if not questions:
        return f"No MCQs generated for {context_label}.", []

    valid_questions = []
    for i, q in enumerate(questions):
        if not isinstance(q, dict): continue
        if "stem" not in q or "options" not in q: continue
        valid_questions.append(q)
    
    if not valid_questions:
        return f"No valid MCQs format received for {context_label}.", []
    
    # Save to file
    out_file = Path(MCQ_FOLDER) / "mcqs_generated.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(valid_questions, f, indent=2, ensure_ascii=False)
    
    return f"Generated {len(valid_questions)} MCQs for {context_label}", valid_questions


# ------------------------
# Format a single question for display
# ------------------------
def format_single_question(q, q_num, user_answer=None, show_feedback=False):
    """Format a single question as a clean HTML Card."""
    stem = q["stem"]
    difficulty = q.get("difficulty", "Medium")
    
    # Difficulty Color Logic
    diff_color = "#3b82f6" # Blue (Medium)
    if difficulty == "Easy": diff_color = "#10b981" # Green
    if difficulty == "Hard": diff_color = "#ef4444" # Red

    # 1. Main Card Container
    # We use rgba backgrounds to look good in both Dark and Light mode
    html = f"""
    <div style="
        border: 1px solid #e5e7eb40; 
        border-radius: 12px; 
        padding: 20px; 
        background: rgba(255,255,255,0.03); 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
        font-family: 'Segoe UI', sans-serif;
    ">
        <!-- Header: Question Number & Badge -->
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="
                font-weight: 700; 
                color: #9ca3af; 
                text-transform: uppercase; 
                font-size: 0.85rem; 
                letter-spacing: 0.05em;
            ">
                QUESTION {q_num}
            </span>
            <span style="
                background-color: {diff_color}20; 
                color: {diff_color}; 
                padding: 4px 10px; 
                border-radius: 99px; 
                font-size: 0.75rem; 
                font-weight: 700;
                border: 1px solid {diff_color}40;
            ">
                {difficulty.upper()}
            </span>
        </div>
        
        <!-- Question Stem -->
        <div style="
            font-size: 1.15rem; 
            font-weight: 500; 
            line-height: 1.6; 
            margin-bottom: 10px;
        ">
            {stem}
        </div>
    """

    # 2. Feedback Section (Only if submitted)
    if show_feedback and user_answer:
        correct_letter = q.get("answer", "")
        is_correct = (user_answer == correct_letter)
        
        status_color = "#10b981" if is_correct else "#ef4444"
        status_text = "✅ Correct Answer" if is_correct else f"❌ Incorrect. Correct answer: {correct_letter}"
        bg_color = f"{status_color}15" # 15% opacity
        
        explanation = q.get("explanation", "No explanation provided.")
        
        html += f"""
        <div style="
            margin-top: 15px; 
            padding: 15px; 
            background-color: {bg_color}; 
            border-left: 4px solid {status_color}; 
            border-radius: 4px;
        ">
            <div style="font-weight: bold; color: {status_color}; margin-bottom: 5px;">
                {status_text}
            </div>
            <div style="font-size: 0.95rem; opacity: 0.9;">
                <strong>Explanation:</strong> {explanation}
            </div>
        """
        if q.get("citations"):
            html += f"""
            <div style="margin-top: 8px; font-size: 0.85rem; opacity: 0.6; font-style: italic;">
                Source: {', '.join(q['citations'])}
            </div>
            """
        html += "</div>"

    html += "</div>"
    return html

def get_modern_header(page_num, start_idx, end_idx, total, submitted=False):
    sub_text = " (Submitted)" if submitted else ""
    return (
        f"<div style='margin:  20px; padding: 15px;'>"
        f"<strong>Page {page_num + 1} (Questions {start_idx + 1}-{end_idx})"
        f" — Submitted</strong></div>"
    )


def ensure_pool_for_next_step(session):
    """
    Checks if the adaptive pool has available questions for the user's current Theta.
    If not, generates a small batch (3) of specific difficulty and appends to pool.
    """
    theta = session.get("theta", 0.0)
    pool = session.get("mcq_pool", [])
    asked_ids = set(session.get("asked_ids", []))
    subject = session.get("subject")
    topic = session.get("topic")

    # 1. Determine required difficulty based on Theta
    if theta > 1.0:
        needed_diff = "Hard"
    elif theta < -1.0:
        needed_diff = "Easy"
    else:
        needed_diff = "Medium"

    # 2. Check availability
    # Count how many unasked questions of this difficulty exist
    available_count = 0
    for i, q in enumerate(pool):
        if i not in asked_ids and q.get("difficulty") == needed_diff:
            available_count += 1
    
    # 3. Refill if empty
    if available_count == 0:
        print(f"[ADAPTIVE] Pool exhausted for {needed_diff} (Theta: {theta:.2f}). Refilling...")
        
        try:
            # Generate small batch (3 questions) to minimize wait time
            new_qs = generate_mcqs_for_topic(
                subject, 
                topic, 
                n_questions=3, 
                difficulty=needed_diff
            )
            
            if new_qs:
                # Add metadata and append to pool
                for q in new_qs:
                    q_norm = dict(q)
                    q_norm.setdefault("topic", topic)
                    q_norm.setdefault("difficulty", needed_diff)
                    pool.append(q_norm)
                
                # Update session
                session["mcq_pool"] = pool
                print(f"[ADAPTIVE] Added {len(new_qs)} new {needed_diff} questions.")
            else:
                print("[ADAPTIVE] Failed to generate refill questions.")
                
        except Exception as e:
            print(f"[ADAPTIVE] Error refilling pool: {e}")

    return session

def build_adaptive_pool(subject: str, topic: str, per_level: int = 5) -> List[Dict[str, Any]]:
    """
    Build a mixed-difficulty pool of MCQs for adaptive mode:
    some Easy, some Medium, some Hard for the selected subject/topic.
    """
    pool: List[Dict[str, Any]] = []
    for level in ["Easy", "Medium", "Hard"]:
        print(f"[ADAPTIVE] Generating {per_level} {level} MCQs for {subject} - {topic}")
        qs = generate_mcqs_for_topic(
            subject,
            topic,
            n_questions=per_level,
            difficulty=level,
        )
        for q in qs:
            q_norm = dict(q)
            q_norm.setdefault("topic", topic)
            q_norm.setdefault("difficulty", level)
            pool.append(q_norm)

    if not pool:
        print(f"[ADAPTIVE] No MCQs generated for {subject} - {topic} in any difficulty.")
    else:
        print(f"[ADAPTIVE] Built adaptive pool of {len(pool)} MCQs for {subject} - {topic}")
    return pool

def start_adaptive_session_ui(selected_topic_str, num_steps):
    """Start an adaptive session for the chosen topic."""
    if not selected_topic_str:
        return "Please select a topic first.", {}, "<p>No question yet.</p>", gr.update(choices=[], value=None, interactive=False), ""

    subject, topic = get_topic_from_selection(selected_topic_str)
    if not subject or not topic:
        return "Invalid topic selection.", {}, "<p>No question yet.</p>", gr.update(choices=[], value=None, interactive=False), ""

    try:
        steps = int(num_steps) if num_steps else 10
    except ValueError:
        steps = 10

    # Build adaptive pool (mixed difficulty)
    pool = build_adaptive_pool(subject, topic, per_level=5)
    if not pool:
        return f"No MCQs available for adaptive mode for {subject} - {topic}.", {}, "<p>No question.</p>", gr.update(choices=[], value=None, interactive=False), ""

    # Initial theta, topics_asked, asked_ids, etc.
    theta = 0.0
    topics_asked = {}
    asked_ids = set()

    # Select first MCQ
    pick = select_next_mcq(pool, theta, topics_asked, asked_ids)
    idx = pick["index"]
    q = pick["mcq"]
    asked_ids.add(idx)

    # Prepare session state
    session = {
        "mode": "adaptive",
        "theta": theta,
        "history": [],
        "topics_asked": topics_asked,
        "asked_ids": list(asked_ids),    # store as list for JSON-compat
        "mcq_pool": pool,
        "current_idx": idx,
        "steps_done": 0,
        "max_steps": steps,
        "subject": subject,
        "topic": topic,
    }

    # Build HTML and radio choices for this question
    q_num = 1
    q_html = format_single_question(q, q_num, user_answer=None, show_feedback=False)
    options_list = [
        f"A: {q['options']['A']}",
        f"B: {q['options']['B']}",
        f"C: {q['options']['C']}",
        f"D: {q['options']['D']}",
    ]

    msg = f"Adaptive session started for {subject} - {topic}. Step 1 of {steps}."
    theta_text = f"{theta:.2f}"

    return (
        msg,
        session,
        q_html,
        gr.update(choices=options_list, value=None, interactive=True),
        theta_text,
    )
def adaptive_next_step_ui(session, selected_option):
    """Handle one step of the adaptive session."""
    if not session or session.get("mode") != "adaptive":
        return "Start the adaptive quiz first.", session, "<p>No question.</p>", gr.update(choices=[], value=None, interactive=False), ""

    # ... [Keep existing variable extraction logic] ...
    theta = float(session.get("theta", 0.0))
    topics_asked = dict(session.get("topics_asked", {}))
    asked_ids = set(session.get("asked_ids", []))
    idx = int(session.get("current_idx", 0))
    steps_done = int(session.get("steps_done", 0))
    max_steps = int(session.get("max_steps", 10))
    pool = session.get("mcq_pool", []) # Get current pool

    # 1. Check if Finished (Max Steps Reached)
    if steps_done >= max_steps:
        msg = f"Adaptive session finished. Final θ = {theta:.2f}"
        return msg, session, "<div style='padding:20px'><b>Session Completed.</b></div>", gr.update(visible=False), f"{theta:.2f}"

    # 2. Evaluate Previous Answer
    q = pool[idx]
    # ... [Keep your existing answer checking logic] ...
    if selected_option:
        answer_letter = selected_option.split(":")[0].strip()
    else:
        answer_letter = None
    
    correct = bool(answer_letter and answer_letter == q.get("answer"))
    explanation = q.get("explanation", "No explanation.")
    
    # 3. Update Theta
    theta = update_theta_simple(theta, correct)
    session["theta"] = theta # Update session immediately
    
    steps_done += 1
    session["steps_done"] = steps_done
    
    # ... [Keep History Recording Logic] ...
    session_history = list(session.get("history", []))
    session_history.append({
        "step": steps_done, "idx": idx, "stem": q.get("stem"),
        "student_answer": answer_letter, "correct": correct, "theta": theta
    })
    session["history"] = session_history
    
    # 4. Check for End of Quiz AGAIN (in case this was the last step)
    if steps_done >= max_steps:
         # Show feedback for last question
         q_html = format_single_question(q, steps_done, answer_letter, show_feedback=True)
         msg = f"{'✅ Correct' if correct else '❌ Incorrect'} | Final θ = {theta:.2f}"
         return msg, session, q_html, gr.update(visible=False), f"{theta:.2f}"

    # ============================================================
    # NEW: Refill Pool if needed BEFORE selecting next
    # ============================================================
    asked_ids.add(idx)
    session["asked_ids"] = list(asked_ids) # Update asked list so refill knows what's used
    
    # This checks theta, sees if we have questions, generates if missing
    session = ensure_pool_for_next_step(session) 
    
    # Refresh pool variable after refill
    pool = session["mcq_pool"] 
    # ============================================================

    # 5. Select Next Question
    pick = select_next_mcq(pool, theta, topics_asked, asked_ids)
    
    if not pick:
        # Emergency fallback if generation failed
        return "Error: No suitable questions found and generation failed.", session, "", gr.update(), f"{theta:.2f}"

    new_idx = pick["index"]
    new_q = pick["mcq"]
    session["current_idx"] = new_idx

    # 6. Format Output
    next_step_num = steps_done + 1
    
    # Show Feedback for PREVIOUS question + New Question
    # We can't show both easily in one block without custom HTML, 
    # so standard UI flow usually clears screen or shows a status message.
    
    msg = (
        f"Step {steps_done}: {'✅ Correct' if correct else '❌ Incorrect'}\n"
        f"Difficulty adjusted based on performance (θ={theta:.2f})"
    )
    
    q_html = format_single_question(new_q, next_step_num, user_answer=None, show_feedback=False)
    options_list = [f"{k}: {v}" for k, v in new_q['options'].items()]

    return (
        msg,
        session,
        q_html,
        gr.update(choices=options_list, value=None, interactive=True, visible=True),
        f"{theta:.2f}",
    )

# (format_questions_page is unused in UI, but update it for consistency)
def format_questions_page(questions, page_num, user_answers, questions_per_page=10):
    """Format a page of questions for display with feedback"""
    start_idx = page_num * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(questions))
    page_questions = questions[start_idx:end_idx]
    
    if not page_questions:
        return "No questions on this page."
    
    html = (
        f"<div style='"
        f"display: flex; align-items: center; justify-content: space-between; "
        f"background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px; "
        f"padding: 10px 16px; margin-bottom: 20px; font-family: sans-serif;'>"
        
        # Left side: The Page Badge
        f"  <div style='display: flex; align-items: center; gap: 12px;'>"
        f"    <span style='"
        f"      background-color: #2563eb; color: white; padding: 4px 12px; "
        f"      border-radius: 9999px; font-size: 0.85rem; font-weight: 600; "
        f"      letter-spacing: 0.025em; box-shadow: 0 1px 2px rgba(0,0,0,0.1);'>"
        f"      PAGE {page_num + 1}"
        f"    </span>"
        f"    <span style='color: #374151; font-weight: 600; font-size: 0.95rem;'>"
        f"      Questions {start_idx + 1} — {end_idx}"
        f"    </span>"
        f"  </div>"

        # Right side: Total Count (Subtle)
        f"  <div style='color: #6b7280; font-size: 0.85rem; font-weight: 500;'>"
        f"    Total: {len(questions)}"
        f"  </div>"
        f"</div>"
    )
    
    # Default: no feedback unless you pass show_feedback=True
    for i, q in enumerate(page_questions):
        q_num = start_idx + i + 1
        q_id = f"q_{start_idx + i}"
        user_answer = user_answers.get(q_id, None)
        html += format_single_question(q, q_num, user_answer, show_feedback=False)
    
    return html


# ------------------------
# Create radio components for questions on current page
# ------------------------
def create_question_radios(questions, page_num, user_answers, questions_per_page=10):
    """Create a list of Radio components for questions on current page"""
    start_idx = page_num * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(questions))
    page_questions = questions[start_idx:end_idx]
    
    radios = []
    for i, q in enumerate(page_questions):
        q_num = start_idx + i + 1
        q_id = f"q_{start_idx + i}"
        options_list = [
            f"A: {q['options']['A']}",
            f"B: {q['options']['B']}",
            f"C: {q['options']['C']}",
            f"D: {q['options']['D']}"
        ]
        user_answer = user_answers.get(q_id, None)
        selected_value = None
        if user_answer:
            for opt in options_list:
                if opt.startswith(user_answer + ":"):
                    selected_value = opt
                    break
        
        radio = gr.Radio(
            choices=options_list,
            label=(
                f"Question {q_num}: {q['stem'][:50]}..."
                if len(q['stem']) > 50
                else f"Question {q_num}: {q['stem']}"
            ),
            value=selected_value,
            interactive=True,
            visible=True
        )
        radios.append(radio)
    
    # Fill remaining slots with invisible radios (up to 10)
    while len(radios) < questions_per_page:
        radio = gr.Radio(choices=[], label="", visible=False, interactive=False)
        radios.append(radio)
    
    return radios


# ------------------------
# Update user answers when radio button changes
# ------------------------
def update_answer(session, question_idx, selected_option):
    """Update user answer when a radio button is selected"""
    if session is None or session == {}:
        return session
    
    if "user_answers" not in session:
        session["user_answers"] = {}
    
    if selected_option:
        # Extract answer letter (A, B, C, or D)
        answer_letter = selected_option.split(":")[0].strip()
        session["user_answers"][question_idx] = answer_letter
    
    return session


# ------------------------
# Start quiz / Generate MCQs with progress updates
# ------------------------
def start_quiz_ui(mode, subject_input, topic_input, num_questions, difficulty):
    # Yield loading state (Hide groups initially)
    loading_html = (
        "<div style='padding: 40px; text-align: center; background: #f5f5f5; border-radius: 10px;'>"
        "<h3>🔄 Generating MCQs...</h3>"
        "<p>Please wait while we retrieve content and generate questions.</p></div>"
    )
    
    # OUTPUTS: msg, session, main_html, prev, next, page_info, 10 GROUPS, 10 HTMLS, 10 RADIOS
    yield (
        "Generating...",
        {},
        loading_html,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value="Generating..."),
        *([gr.update(visible=False)] * 10), # Hide Groups
        *([gr.update(visible=False)] * 10), # Hide HTMLs
        *([gr.update(visible=False)] * 10)  # Hide Radios
    )
    
    # Generate
    msg, questions = generate_mcqs_ui_wrapper(
        mode, subject_input, topic_input, num_questions, difficulty
    )
    
    if not questions:
        error_html = f"<div style='padding: 20px; color: red;'><p>{msg}</p></div>"
        yield (
            msg, {}, error_html,
            gr.update(visible=False), gr.update(visible=False), gr.update(value="Error"),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10)
        )
        return

    # Initialize Session
    session = {
        "questions": questions,
        "current_page": 0,
        "questions_per_page": 10,
        "user_answers": {},
        "submitted": False
    }

    # Render First Page
    total_pages = (len(questions) + 9) // 10
    start_idx = 0
    end_idx = min(10, len(questions))
    
    group_updates = []
    question_html_updates = []
    radio_updates = []
    
    # Create Modern Header
    header_html = get_modern_header(1, start_idx+1, end_idx, len(questions))
    
    for i in range(10):
        if i < len(questions):
            q = questions[i]
            q_num = i + 1
            
# Attach header to first question block

            q_html = ""
            q_html += format_single_question(q, q_num, show_feedback=False)
            
            opts = [f"{k}: {v}" for k, v in q['options'].items()]
            
            group_updates.append(gr.update(visible=True)) # SHOW GROUP
            question_html_updates.append(gr.update(value=q_html, visible=True))
            radio_updates.append(gr.update(
                choices=opts,
                label=f"Select Answer",
                value=None,
                visible=True,
                interactive=True
            ))
        else:
            group_updates.append(gr.update(visible=False)) # HIDE GROUP
            question_html_updates.append(gr.update(value="", visible=False))
            radio_updates.append(gr.update(visible=False))
            
    yield (
        msg,
        session,
        gr.update(visible=False), # Hide main text
        gr.update(visible=False), # Prev
        gr.update(visible=(total_pages > 1)), # Next
        gr.update(value=f"Page 1 of {total_pages}"),
        *group_updates,
        *question_html_updates,
        *radio_updates
    )

# ------------------------
# Pagination functions
# ------------------------
def next_page_ui(session):
    """Updates Groups, HTML, and Radios for the current page"""
    if not session or "questions" not in session:
        return (
            session, gr.update(), gr.update(), gr.update(), gr.update(),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10)
        )   
    
    questions = session["questions"]
    questions_per_page = session.get("questions_per_page", 10)
    user_answers = session.get("user_answers", {})
    total_pages = (len(questions) + questions_per_page - 1) // questions_per_page
    current_page = session["current_page"]
    submitted = session.get("submitted", False)
    
    if current_page < total_pages - 1:
        session["current_page"] += 1
        new_page = session["current_page"]
        
        prev_btn_visible = True
        next_btn_visible = new_page < total_pages - 1
        
        start_idx = new_page * questions_per_page
        end_idx = min(start_idx + questions_per_page, len(questions))
        page_questions = questions[start_idx:end_idx]
        
        group_updates = [] 
        question_html_updates = []
        radio_updates = []
        
        header_html = get_modern_header(new_page+1, start_idx+1, end_idx, len(questions), submitted)
        
        for i in range(questions_per_page):
            if i < len(page_questions):
                q = page_questions[i]
                q_num = start_idx + i + 1
                q_id = f"q_{start_idx + i}"
                user_answer = user_answers.get(q_id, None)
                
                q_html = header_html if i == 0 else ""
                q_html += format_single_question(q, q_num, user_answer, show_feedback=submitted)
                
                options_list = [f"{k}: {v}" for k, v in q['options'].items()]
                selected_value = None
                if user_answer:
                    for opt in options_list:
                        if opt.startswith(user_answer + ":"):
                            selected_value = opt
                            break
                
                group_updates.append(gr.update(visible=True))
                question_html_updates.append(gr.update(value=q_html, visible=True))
                radio_updates.append(gr.update(
                    choices=options_list,
                    label=f"Select Answer",
                    value=selected_value,
                    visible=True,
                    interactive=not submitted
                ))
            else:
                group_updates.append(gr.update(visible=False))
                question_html_updates.append(gr.update(value="", visible=False))
                radio_updates.append(gr.update(visible=False))
        
        return (
            session,
            gr.update(visible=False),
            gr.update(visible=prev_btn_visible),
            gr.update(visible=next_btn_visible),
            gr.update(value=f"Page {new_page + 1} of {total_pages}"),
            *group_updates,
            *question_html_updates,
            *radio_updates
        )
    return (session, *([gr.update()] * 34)) # No change fallback

def prev_page_ui(session):
    """Navigate to previous page"""
    if not session or "questions" not in session:
        return (
            session, gr.update(), gr.update(), gr.update(), gr.update(),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10)
        )
    
    questions = session["questions"]
    questions_per_page = session.get("questions_per_page", 10)
    user_answers = session.get("user_answers", {})
    total_pages = (len(questions) + questions_per_page - 1) // questions_per_page
    current_page = session["current_page"]
    submitted = session.get("submitted", False)
    
    if current_page > 0:
        session["current_page"] -= 1
        new_page = session["current_page"]
        
        prev_btn_visible = new_page > 0
        next_btn_visible = True
        
        start_idx = new_page * questions_per_page
        end_idx = min(start_idx + questions_per_page, len(questions))
        page_questions = questions[start_idx:end_idx]
        
        group_updates = [] 
        question_html_updates = []
        radio_updates = []
        
        header_html = get_modern_header(new_page+1, start_idx+1, end_idx, len(questions), submitted)
        
        for i in range(questions_per_page):
            if i < len(page_questions):
                q = page_questions[i]
                q_num = start_idx + i + 1
                q_id = f"q_{start_idx + i}"
                user_answer = user_answers.get(q_id, None)
                
                q_html = header_html if i == 0 else ""
                q_html += format_single_question(q, q_num, user_answer, show_feedback=submitted)
                
                options_list = [f"{k}: {v}" for k, v in q['options'].items()]
                selected_value = None
                if user_answer:
                    for opt in options_list:
                        if opt.startswith(user_answer + ":"):
                            selected_value = opt
                            break
                
                group_updates.append(gr.update(visible=True))
                question_html_updates.append(gr.update(value=q_html, visible=True))
                radio_updates.append(gr.update(
                    choices=options_list,
                    label=f"Select Answer",
                    value=selected_value,
                    visible=True,
                    interactive=not submitted
                ))
            else:
                group_updates.append(gr.update(visible=False))
                question_html_updates.append(gr.update(value="", visible=False))
                radio_updates.append(gr.update(visible=False))
        
        return (
            session,
            gr.update(visible=False),
            gr.update(visible=prev_btn_visible),
            gr.update(visible=next_btn_visible),
            gr.update(value=f"Page {new_page + 1} of {total_pages}"),
            *group_updates,
            *question_html_updates,
            *radio_updates
        )
    return (session, *([gr.update()] * 34))


# ------------------------
# Submit Quiz
# ------------------------
def submit_quiz_ui(session):
    if session is None or session == {}:
        # Must return 36 items (6 fixed + 30 dynamic)
        return (
            "Please generate MCQs first.",
            session, gr.update(), gr.update(), gr.update(), gr.update(),
            *([gr.update()] * 30) # <--- Ensured 30 updates here
        )

    questions = session.get("questions", [])
    user_answers = session.get("user_answers", {})
    total_questions = len(questions)

    # Check for missing answers
    missing = [i + 1 for i in range(total_questions) if f"q_{i}" not in user_answers]
    if missing:
        missing_str = ", ".join(map(str, missing))
        msg = f"Please answer all questions. Missing: {missing_str}"
        # Must return 36 items
        return (
            msg,
            session, gr.update(), gr.update(), gr.update(), gr.update(),
            *([gr.update()] * 30) # <--- Ensured 30 updates here
        )

    # Mark submitted
    session["submitted"] = True
    correct_count = sum(1 for i,q in enumerate(questions) if user_answers.get(f"q_{i}")==q.get("answer"))

    msg = f"✅ Quiz submitted! Your score: {correct_count} / {total_questions}"

    questions_per_page = session.get("questions_per_page", 10)
    current_page = session.get("current_page", 0)
    total_pages = (len(questions) + questions_per_page - 1) // questions_per_page

    start_idx = current_page * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(questions))
    page_questions = questions[start_idx:end_idx]

    # Lists for the 3 dynamic components
    group_updates = []
    question_html_updates = []
    radio_updates = []

    header_html = get_modern_header(current_page+1, start_idx+1, end_idx, len(questions), True)

    for i in range(questions_per_page):
        if i < len(page_questions):
            q = page_questions[i]
            q_num = start_idx + i + 1
            qid = f"q_{start_idx + i}"
            user_answer = user_answers.get(qid)

            q_html = header_html if i == 0 else ""
            q_html += format_single_question(q, q_num, user_answer, show_feedback=True)

            options_list = [f"{k}: {v}" for k, v in q['options'].items()]
            selected_value = None
            if user_answer:
                for opt in options_list:
                    if opt.startswith(user_answer + ":"):
                        selected_value = opt
                        break

            # Add updates for all 3 lists
            group_updates.append(gr.update(visible=True))
            question_html_updates.append(gr.update(value=q_html, visible=True))
            radio_updates.append(gr.update(
                choices=options_list,
                label=f"Select Answer",
                value=selected_value,
                visible=True,
                interactive=False
            ))
        else:
            # Hide empty slots
            group_updates.append(gr.update(visible=False))
            question_html_updates.append(gr.update(value="", visible=False))
            radio_updates.append(gr.update(visible=False))

    prev_btn_visible = current_page > 0
    next_btn_visible = current_page < (total_pages - 1)

    # RETURN ALL 3 LISTS
    return (
        msg,
        session,
        gr.update(visible=False),
        gr.update(visible=prev_btn_visible),
        gr.update(visible=next_btn_visible),
        gr.update(value=f"Page {current_page + 1} of {total_pages}"),
        *group_updates,          # <--- 1. Groups (10 items)
        *question_html_updates,  # <--- 2. HTMLs (10 items)
        *radio_updates           # <--- 3. Radios (10 items)
    )
# ------------------------
# Metrics
# ------------------------
def generate_metrics_ui():
    resp_file = Path(MCQ_FOLDER) / "responses.json"
    mcq_file = Path(MCQ_FOLDER) / "mcqs_generated.json"

    if not resp_file.exists():
        return "No responses found.", None

    with open(resp_file, "r", encoding="utf-8") as f:
        responses = json.load(f)
    with open(mcq_file, "r", encoding="utf-8") as f:
        items = json.load(f)

    summary = generate_metrics_summary(responses, items)
    plot_metrics(summary, out_dir=MCQ_FOLDER)
    return "Metrics generated!", summary.to_dict(orient="records")


# ------------------------
# Gradio UI
# ------------------------
# Load topics on startup
all_topics = load_all_topics()
topic_options = format_topics_for_display(all_topics)
all_subjects = load_all_subjects() # Load subjects

with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Adaptive MCQ Quiz System")

    with gr.Tab("Quiz Generator"):
        gr.Markdown("### Generate Questions")
        
        # --- SELECTION AREA ---
        with gr.Group():
            with gr.Row():
                mode_radio = gr.Radio(
                    choices=["By Topic", "By Subject"],
                    value="By Topic",
                    label="Generation Mode"
                )
            
            # Use columns to swap visibility without layout jumping too much
            with gr.Row():
                topic_dropdown = gr.Dropdown(
                    choices=topic_options,
                    label="Select Topic",
                    visible=True,
                    interactive=True
                )
                subject_dropdown = gr.Dropdown(
                    choices=all_subjects,
                    label="Select Subject",
                    visible=False, # Hidden by default
                    interactive=True
                )
            
            with gr.Row():
                num_q_input = gr.Number(value=5, label="Num Questions", minimum=1, maximum=30)
                diff_input = gr.Dropdown(choices=["Easy", "Medium", "Hard"], value="Medium", label="Difficulty")
                
            gen_btn = gr.Button("🚀 Generate MCQs", variant="primary")

        # --- TOGGLE VISIBILITY ---
        def toggle_inputs(mode):
            if mode == "By Topic":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        mode_radio.change(toggle_inputs, inputs=[mode_radio], outputs=[topic_dropdown, subject_dropdown])

        # --- QUIZ DISPLAY AREA ---
        msg_box = gr.Textbox(label="Status", interactive=False)
        session_state = gr.State()
        page_info = gr.Textbox(label="Page Info", interactive=False)
        
        # Container for layout
        questions_container = gr.Markdown("### Questions appear here") # Placeholder

        # --- NEW LAYOUT: Capture Groups ---
        q_groups = []
        q_htmls = []
        q_radios = []
        
        with gr.Column(): 
            for i in range(10):
                # Initialize invisible
                with gr.Group(visible=False) as g: 
                    m = gr.HTML(visible=False)
                    r = gr.Radio(visible=False, label="Select Answer")
                    q_groups.append(g)
                    q_htmls.append(m)
                    q_radios.append(r)
        
        with gr.Row():
            btn_prev = gr.Button("◀ Prev", visible=False)
            btn_submit = gr.Button("Submit Quiz", visible=True)
            btn_next = gr.Button("Next ▶", visible=False)

        # --- OUTPUT LIST (Common for all buttons) ---
        # Order: msg, session, main_html, prev, next, page_info, GROUPS, HTMLS, RADIOS
        ui_outputs = [msg_box, session_state, questions_container, btn_prev, btn_next, page_info] + q_groups + q_htmls + q_radios

        gen_btn.click(
            start_quiz_ui,
            inputs=[mode_radio, subject_dropdown, topic_dropdown, num_q_input, diff_input],
            outputs=ui_outputs
        )

        btn_prev.click(prev_page_ui, inputs=[session_state], outputs=ui_outputs)
        btn_next.click(next_page_ui, inputs=[session_state], outputs=ui_outputs)
        btn_submit.click(submit_quiz_ui, inputs=[session_state], outputs=ui_outputs)

        # Radio change logic
        def make_radio_handler(idx):
            def handler(session, val):
                if not session: return session
                qs_per_page = session.get("questions_per_page", 10)
                curr_page = session.get("current_page", 0)
                abs_idx = curr_page * qs_per_page + idx
                qid = f"q_{abs_idx}"
                if val:
                    ans = val.split(":")[0].strip()
                    if "user_answers" not in session: session["user_answers"] = {}
                    session["user_answers"][qid] = ans
                return session
            return handler

        for i, r in enumerate(q_radios):
            r.change(make_radio_handler(i), inputs=[session_state, r], outputs=[session_state])
            
    with gr.Tab("Metrics"):
        metrics_btn = gr.Button("Generate Metrics")
        out_text = gr.Textbox()
        out_json = gr.JSON()
        metrics_btn.click(generate_metrics_ui, outputs=[out_text, out_json])

    with gr.Tab("Adaptive Quiz"):
        gr.Markdown("### Adaptive MCQ Quiz (Interactive, topic-based)")

        adaptive_topic_dropdown = gr.Dropdown(
            choices=topic_options,
            label="Select Topic",
            value=None,
            interactive=True
        )

        adaptive_steps = gr.Number(
            value=10,
            label="Number of Adaptive Steps",
            minimum=1,
            maximum=50
        )

        start_adaptive_btn = gr.Button("Start Adaptive Quiz", variant="primary")

        adaptive_msg_box = gr.Textbox(label="Adaptive System Message", interactive=False)
        adaptive_session_state = gr.State()

        adaptive_theta_box = gr.Textbox(label="Current Ability (θ)", interactive=False)

        adaptive_question_html = gr.Markdown("Question will appear here")
        adaptive_answer_radio = gr.Radio(
            choices=[],
            label="Your Answer",
            interactive=False
        )

        adaptive_next_btn = gr.Button("Submit Answer & Next", variant="secondary")

        # Wire up handlers
        start_adaptive_btn.click(
            start_adaptive_session_ui,
            inputs=[adaptive_topic_dropdown, adaptive_steps],
            outputs=[
                adaptive_msg_box,
                adaptive_session_state,
                adaptive_question_html,
                adaptive_answer_radio,
                adaptive_theta_box,
            ],
        )

        adaptive_next_btn.click(
            adaptive_next_step_ui,
            inputs=[adaptive_session_state, adaptive_answer_radio],
            outputs=[
                adaptive_msg_box,
                adaptive_session_state,
                adaptive_question_html,
                adaptive_answer_radio,
                adaptive_theta_box,
            ],
        )


demo.queue()  # Enable queueing for generators to work properly
demo.launch()
