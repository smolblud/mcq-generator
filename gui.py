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
    """Format a single question as Markdown (LaTeX-friendly)."""
    stem = q["stem"]
    opts = q["options"]
    difficulty = q.get("difficulty", "Medium")

    md = []
    md.append(f"### Question {q_num}\n")
    md.append(f"**{stem}**\n")

    # Options as a bulleted list
    md.append("")
    md.append(f"- A: {opts['A']}")
    md.append(f"- B: {opts['B']}")
    md.append(f"- C: {opts['C']}")
    md.append(f"- D: {opts['D']}")
    md.append("")

    if show_feedback and user_answer:
        correct_letter = q.get("answer", "")
        if user_answer == correct_letter:
            md.append("✅ **Correct!**")
        else:
            md.append(f"❌ **Incorrect.** Correct answer: **{correct_letter}**")

        explanation = q.get("explanation", "No explanation provided.")
        # Explanation may contain LaTeX like $\\sin x$, etc.
        md.append("")
        md.append(f"> **Explanation:** {explanation}")

        if q.get("citations"):
            citations = ", ".join(q["citations"])
            md.append(f"> *Citations:* {citations}")

    md.append("")
    md.append(f"*Difficulty: {difficulty}*")

    return "\n".join(md)


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
    """Handle one step of the adaptive session: evaluate answer, update theta, move to next."""
    if not session or session.get("mode") != "adaptive":
        return (
            "Start the adaptive quiz first.",
            session,
            "<p>No question.</p>",
            gr.update(choices=[], value=None, interactive=False),
            "",
        )

    pool = session.get("mcq_pool", [])
    if not pool:
        return (
            "No MCQ pool found. Please restart the adaptive quiz.",
            session,
            "<p>No question.</p>",
            gr.update(choices=[], value=None, interactive=False),
            "",
        )

    theta = float(session.get("theta", 0.0))
    topics_asked = dict(session.get("topics_asked", {}))
    asked_ids = set(session.get("asked_ids", []))
    idx = int(session.get("current_idx", 0))
    steps_done = int(session.get("steps_done", 0))
    max_steps = int(session.get("max_steps", 10))

    if steps_done >= max_steps:
        msg = f"Adaptive session already finished. Final θ = {theta:.2f}"
        return (
            msg,
            session,
            "<p>Session completed.</p>",
            gr.update(choices=[], value=None, interactive=False),
            f"{theta:.2f}",
        )

    # Current question
    q = pool[idx]
    topic = q.get("topic", "misc")
    diff = q.get("difficulty", "Medium")

    # Extract user's answer letter
    if selected_option:
        answer_letter = selected_option.split(":")[0].strip()
    else:
        answer_letter = None

    correct = bool(answer_letter and answer_letter == q.get("answer"))
    explanation = q.get("explanation", "No explanation provided.")

    # Update theta and bookkeeping
    theta = update_theta_simple(theta, correct)
    topics_asked[topic] = topics_asked.get(topic, 0) + 1
    steps_done += 1

    # Record history
    session_history = list(session.get("history", []))
    session_history.append(
        {
            "step": steps_done,
            "idx": idx,
            "topic": topic,
            "difficulty": diff,
            "stem": q.get("stem"),
            "student_answer": answer_letter,
            "correct_answer": q.get("answer"),
            "student_correct": correct,
            "theta_after": theta,
        }
    )

    # Update session object
    session["theta"] = theta
    session["topics_asked"] = topics_asked
    session["asked_ids"] = list(asked_ids)
    session["history"] = session_history
    session["steps_done"] = steps_done

    # If finished after this step
    if steps_done >= max_steps:
        msg = (
            f"{'✅ Correct' if correct else '❌ Incorrect'} | "
            f"θ = {theta:.2f} | Step {steps_done}/{max_steps}\n\n"
            f"Explanation: {explanation}\n\n"
            "Adaptive session complete."
        )
        # Show the same question with feedback
        q_html = format_single_question(q, steps_done, answer_letter, show_feedback=True)
        return (
            msg,
            session,
            q_html,
            gr.update(choices=[], value=selected_option, interactive=False),
            f"{theta:.2f}",
        )

    # Otherwise: pick next question
    asked_ids.add(idx)
    session["asked_ids"] = list(asked_ids)

    pick = select_next_mcq(pool, theta, topics_asked, asked_ids)
    new_idx = pick["index"]
    new_q = pick["mcq"]
    session["current_idx"] = new_idx

    next_step_num = steps_done + 1
    q_html = format_single_question(new_q, next_step_num, user_answer=None, show_feedback=False)
    options_list = [
        f"A: {new_q['options']['A']}",
        f"B: {new_q['options']['B']}",
        f"C: {new_q['options']['C']}",
        f"D: {new_q['options']['D']}",
    ]

    msg = (
        f"{'✅ Correct' if correct else '❌ Incorrect'} | "
        f"θ = {theta:.2f} | Step {steps_done}/{max_steps}\n\n"
        f"Explanation: {explanation}\n\n"
        f"Next question: Step {next_step_num}/{max_steps}."
    )

    return (
        msg,
        session,
        q_html,
        gr.update(choices=options_list, value=None, interactive=True),
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
        f"<div style='margin-bottom: 20px;'>"
        f"<strong>Page {page_num + 1} (Questions {start_idx + 1}-{end_idx} of {len(questions)})"
        f"</strong></div>"
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
    # Yield loading state
    loading_html = (
        "<div style='padding: 40px; text-align: center; background: #f5f5f5; border-radius: 10px;'>"
        "<h3>🔄 Generating MCQs...</h3>"
        "<p>Please wait while we retrieve content and generate questions.</p></div>"
    )
    yield (
        "Generating...",
        {},
        loading_html,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value="Generating..."),
        *([gr.update(visible=False)] * 10),
        *([gr.update(visible=False)] * 10)
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
            *([gr.update(visible=False)] * 10), *([gr.update(visible=False)] * 10)
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
    
    question_html_updates = []
    radio_updates = []
    
    header = f"<div style='margin-bottom: 20px;'><strong>Page 1 of {total_pages}</strong></div>"
    
    for i in range(10):
        if i < len(questions):
            q = questions[i]
            q_num = i + 1
            
            q_html = header if i == 0 else ""
            q_html += format_single_question(q, q_num, show_feedback=False)
            
            opts = [f"{k}: {v}" for k, v in q['options'].items()]
            
            question_html_updates.append(gr.update(value=q_html, visible=True))
            radio_updates.append(gr.update(
                choices=opts,
                label=f"Answer Q{q_num}",
                value=None,
                visible=True,
                interactive=True
            ))
        else:
            question_html_updates.append(gr.update(value="", visible=False))
            radio_updates.append(gr.update(visible=False))
            
    yield (
        msg,
        session,
        gr.update(visible=False), # Hide main text
        gr.update(visible=False), # Prev
        gr.update(visible=(total_pages > 1)), # Next
        gr.update(value=f"Page 1 of {total_pages}"),
        *question_html_updates,
        *radio_updates
    )

# ------------------------
# Pagination functions
# ------------------------
def next_page_ui(session):
    """Navigate to next page"""
    if session is None or session == {}:
        return (
            session,
            "<p>Click 'Generate MCQs' first.</p>",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10)
        )
    
    questions = session["questions"]
    questions_per_page = session.get("questions_per_page", 10)
    user_answers = session.get("user_answers", {})
    total_pages = (len(questions) + questions_per_page - 1) // questions_per_page
    current_page = session["current_page"]
    submitted = session.get("submitted", False)  # <<< NEW
    
    if current_page < total_pages - 1:
        session["current_page"] += 1
        new_page = session["current_page"]
        
        prev_btn_visible = True
        next_btn_visible = new_page < total_pages - 1
        
        # Create question HTMLs and radio updates
        start_idx = new_page * questions_per_page
        end_idx = min(start_idx + questions_per_page, len(questions))
        page_questions = questions[start_idx:end_idx]
        
        question_html_updates = []
        radio_updates = []
        page_header = (
            f"<div style='margin-bottom: 20px;'>"
            f"<strong>Page {new_page + 1} (Questions {start_idx + 1}-{end_idx} of {len(questions)})"
            f"</strong></div>"
        )
        
        for i in range(questions_per_page):
            if i < len(page_questions):
                q = page_questions[i]
                q_num = start_idx + i + 1
                q_id = f"q_{start_idx + i}"
                user_answer = user_answers.get(q_id, None)
                
                q_html = page_header if i == 0 else ""
                q_html += format_single_question(
                    q, q_num, user_answer, show_feedback=submitted
                )
                question_html_updates.append(gr.update(value=q_html, visible=True))
                
                options_list = [
                    f"A: {q['options']['A']}",
                    f"B: {q['options']['B']}",
                    f"C: {q['options']['C']}",
                    f"D: {q['options']['D']}"
                ]
                selected_value = None
                if user_answer:
                    for opt in options_list:
                        if opt.startswith(user_answer + ":"):
                            selected_value = opt
                            break
                
                radio_updates.append(
                    gr.update(
                        choices=options_list,
                        label=f"Select your answer for Question {q_num}",
                        value=selected_value,
                        visible=True,
                        interactive=not submitted  # lock after submit
                    )
                )
            else:
                question_html_updates.append(gr.update(value="", visible=False))
                radio_updates.append(
                    gr.update(choices=[], label="", visible=False, interactive=False)
                )
        
        return (
            session,
            gr.update(visible=False),
            gr.update(visible=prev_btn_visible),
            gr.update(visible=next_btn_visible),
            gr.update(value=f"Page {new_page + 1} of {total_pages}"),
            *question_html_updates,
            *radio_updates
        )
    
    return (
        session,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=f"Page {current_page + 1} of {total_pages}"),
        *([gr.update(visible=False)] * 10),
        *([gr.update(visible=False)] * 10)
    )


def prev_page_ui(session):
    """Navigate to previous page"""
    if session is None or session == {}:
        return (
            session,
            "<p>Click 'Generate MCQs' first.</p>",
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10)
        )
    
    questions = session["questions"]
    questions_per_page = session.get("questions_per_page", 10)
    user_answers = session.get("user_answers", {})
    total_pages = (len(questions) + questions_per_page - 1) // questions_per_page
    current_page = session["current_page"]
    submitted = session.get("submitted", False)  # <<< NEW
    
    if current_page > 0:
        session["current_page"] -= 1
        new_page = session["current_page"]
        
        prev_btn_visible = new_page > 0
        next_btn_visible = True
        
        # Create question HTMLs and radio updates
        start_idx = new_page * questions_per_page
        end_idx = min(start_idx + questions_per_page, len(questions))
        page_questions = questions[start_idx:end_idx]
        
        question_html_updates = []
        radio_updates = []
        page_header = (
            f"<div style='margin-bottom: 20px;'>"
            f"<strong>Page {new_page + 1} (Questions {start_idx + 1}-{end_idx} of {len(questions)})"
            f"</strong></div>"
        )
        
        for i in range(questions_per_page):
            if i < len(page_questions):
                q = page_questions[i]
                q_num = start_idx + i + 1
                q_id = f"q_{start_idx + i}"
                user_answer = user_answers.get(q_id, None)
                
                q_html = page_header if i == 0 else ""
                q_html += format_single_question(
                    q, q_num, user_answer, show_feedback=submitted
                )
                question_html_updates.append(gr.update(value=q_html, visible=True))
                
                options_list = [
                    f"A: {q['options']['A']}",
                    f"B: {q['options']['B']}",
                    f"C: {q['options']['C']}",
                    f"D: {q['options']['D']}"
                ]
                selected_value = None
                if user_answer:
                    for opt in options_list:
                        if opt.startswith(user_answer + ":"):
                            selected_value = opt
                            break
                
                radio_updates.append(
                    gr.update(
                        choices=options_list,
                        label=f"Select your answer for Question {q_num}",
                        value=selected_value,
                        visible=True,
                        interactive=not submitted  # lock after submit
                    )
                )
            else:
                question_html_updates.append(gr.update(value="", visible=False))
                radio_updates.append(
                    gr.update(choices=[], label="", visible=False, interactive=False)
                )
        
        return (
            session,
            gr.update(visible=False),
            gr.update(visible=prev_btn_visible),
            gr.update(visible=next_btn_visible),
            gr.update(value=f"Page {new_page + 1} of {total_pages}"),
            *question_html_updates,
            *radio_updates
        )
    
    return (
        session,
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(value=f"Page {current_page + 1} of {total_pages}"),
        *([gr.update(visible=False)] * 10),
        *([gr.update(visible=False)] * 10)
    )


# ------------------------
# Submit Quiz
# ------------------------
def submit_quiz_ui(session):  # <<< NEW
    """
    Submit quiz:
    - Ensure all questions are answered.
    - If yes: mark submitted, compute score, and show feedback.
    """
    if session is None or session == {}:
        # Same shape as start_quiz outputs: msg, session, questions_html, prev, next, page, 10 htmls, 10 radios
        return (
            "Please generate MCQs first.",
            session,
            gr.update(),  # questions_html unchanged
            gr.update(),
            gr.update(),
            gr.update(),
            *([gr.update()] * 10),
            *([gr.update()] * 10)
        )

    questions = session.get("questions", [])
    user_answers = session.get("user_answers", {})
    total_questions = len(questions)

    # Check if all questions are answered
    missing = [i + 1 for i in range(total_questions) if f"q_{i}" not in user_answers]
    if missing:
        missing_str = ", ".join(map(str, missing))
        msg = (
            "Please answer all questions before submitting. "
            f"Missing answers for Question(s): {missing_str}"
        )
        # Do not change layout, just update message
        return (
            msg,
            session,
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            *([gr.update()] * 10),
            *([gr.update()] * 10)
        )

    # All answered: mark submitted
    session["submitted"] = True

    # Compute score
    correct_count = 0
    for i, q in enumerate(questions):
        qid = f"q_{i}"
        if user_answers.get(qid) == q.get("answer"):
            correct_count += 1

    msg = f"✅ Quiz submitted! Your score: {correct_count} / {total_questions}"

    questions_per_page = session.get("questions_per_page", 10)
    current_page = session.get("current_page", 0)
    total_pages = (len(questions) + questions_per_page - 1) // questions_per_page

    start_idx = current_page * questions_per_page
    end_idx = min(start_idx + questions_per_page, len(questions))
    page_questions = questions[start_idx:end_idx]

    question_html_updates = []
    radio_updates = []

    page_header = (
        f"<div style='margin-bottom: 20px;'>"
        f"<strong>Page {current_page + 1} (Questions {start_idx + 1}-{end_idx} of {len(questions)})"
        f" — Submitted</strong></div>"
    )

    for i in range(questions_per_page):
        if i < len(page_questions):
            q = page_questions[i]
            q_num = start_idx + i + 1
            qid = f"q_{start_idx + i}"
            user_answer = user_answers.get(qid)

            q_html = page_header if i == 0 else ""
            q_html += format_single_question(
                q, q_num, user_answer, show_feedback=True
            )
            question_html_updates.append(gr.update(value=q_html, visible=True))

            options_list = [
                f"A: {q['options']['A']}",
                f"B: {q['options']['B']}",
                f"C: {q['options']['C']}",
                f"D: {q['options']['D']}"
            ]
            selected_value = None
            if user_answer:
                for opt in options_list:
                    if opt.startswith(user_answer + ":"):
                        selected_value = opt
                        break

            # Lock radios after submission
            radio_updates.append(
                gr.update(
                    choices=options_list,
                    label=f"Select your answer for Question {q_num}",
                    value=selected_value,
                    visible=True,
                    interactive=False
                )
            )
        else:
            question_html_updates.append(gr.update(value="", visible=False))
            radio_updates.append(
                gr.update(choices=[], label="", visible=False, interactive=False)
            )

    prev_btn_visible = current_page > 0
    next_btn_visible = current_page < (total_pages - 1)

    return (
        msg,
        session,
        gr.update(visible=False),  # questions_html stays hidden
        gr.update(visible=prev_btn_visible),
        gr.update(visible=next_btn_visible),
        gr.update(value=f"Page {current_page + 1} of {total_pages} (Submitted)"),
        *question_html_updates,
        *radio_updates
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
        
        # 10 Static Slots
        q_htmls = [gr.Markdown(visible=False) for _ in range(10)]
        q_radios = [gr.Radio(visible=False) for _ in range(10)]
        
        with gr.Row():
            btn_prev = gr.Button("◀ Prev", visible=False)
            btn_submit = gr.Button("Submit Quiz", visible=True)
            btn_next = gr.Button("Next ▶", visible=False)

        # --- EVENT HANDLERS ---
        
        gen_btn.click(
            start_quiz_ui,
            inputs=[mode_radio, subject_dropdown, topic_dropdown, num_q_input, diff_input],
            outputs=[msg_box, session_state, questions_container, btn_prev, btn_next, page_info] + q_htmls + q_radios
        )

        # Pagination & Submit (using your existing functions)
        btn_prev.click(
            prev_page_ui,
            inputs=[session_state],
            outputs=[session_state, questions_container, btn_prev, btn_next, page_info] + q_htmls + q_radios
        )
        btn_next.click(
            next_page_ui,
            inputs=[session_state],
            outputs=[session_state, questions_container, btn_prev, btn_next, page_info] + q_htmls + q_radios
        )
        btn_submit.click(
            submit_quiz_ui,
            inputs=[session_state],
            outputs=[msg_box, session_state, questions_container, btn_prev, btn_next, page_info] + q_htmls + q_radios
        )

        # Radio change logic (capturing answers)
        def make_radio_handler(idx):
            def handler(session, val):
                # Similar logic to your existing update_answer wrapper
                # Need to update specific HTML to show feedback if submitted, 
                # but basically just saves answer
                if not session: return session, *[gr.update() for _ in range(10)]
                
                # Update session
                qs_per_page = session.get("questions_per_page", 10)
                curr_page = session.get("current_page", 0)
                abs_idx = curr_page * qs_per_page + idx
                qid = f"q_{abs_idx}"
                
                if val:
                    ans = val.split(":")[0].strip()
                    if "user_answers" not in session: session["user_answers"] = {}
                    session["user_answers"][qid] = ans
                
                # If we want immediate feedback or just keep state, return session
                # We also need to return updates for all 10 HTML blocks to satisfy Gradio outputs
                # For simplicity, we just return the session and no visual updates until page change/submit
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
