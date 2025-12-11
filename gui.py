import gradio as gr
import json
from pathlib import Path
from mcq_generator import (
    generate_mcqs_for_topic,
    load_topic_blueprint
)
from metrics import generate_metrics_summary, plot_metrics

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
def generate_mcqs_for_selected_topic(selected_topic_str, num_questions):
    """Generate MCQs for a selected topic"""
    if not selected_topic_str:
        return "Please select a topic first.", []
    
    subject, topic = get_topic_from_selection(selected_topic_str)
    if not subject or not topic:
        return "Invalid topic selection.", []
    
    try:
        num_q = int(num_questions) if num_questions else 5
        print(f"[GUI] Generating {num_q} MCQs for {subject} - {topic}")
        
        questions = generate_mcqs_for_topic(subject, topic, n_questions=num_q)
        
        if not questions:
            print(f"[GUI] WARNING: Empty questions list returned for {subject} - {topic}")
            return f"No MCQs generated for {subject} - {topic}. Try a different topic or check the context.", []
        
        # Ensure questions is a list
        if not isinstance(questions, list):
            print(f"[GUI] WARNING: questions is not a list, type: {type(questions)}")
            return f"Error: Expected list of questions but got {type(questions)}", []
        
        # Validate questions structure
        valid_questions = []
        for i, q in enumerate(questions):
            if not isinstance(q, dict):
                print(f"[GUI] WARNING: Question {i} is not a dict, skipping")
                continue
            if "stem" not in q or "options" not in q:
                print(f"[GUI] WARNING: Question {i} missing required fields (stem or options), skipping")
                continue
            valid_questions.append(q)
        
        if not valid_questions:
            print(f"[GUI] WARNING: No valid questions after validation")
            return f"No valid MCQs generated for {subject} - {topic}. The LLM may have returned invalid format.", []
        
        # Save to file
        out_file = Path(MCQ_FOLDER) / "mcqs_generated.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(valid_questions, f, indent=2, ensure_ascii=False)
        
        print(f"[GUI] Successfully saved {len(valid_questions)} MCQs to {out_file}")
        return f"Generated {len(valid_questions)} MCQs for {subject} - {topic}", valid_questions
    except Exception as e:
        import traceback
        error_msg = f"Error generating MCQs: {str(e)}\n{traceback.format_exc()}"
        print(f"[GUI] Exception: {error_msg}")
        return error_msg, []


# ------------------------
# Format a single question for display
# ------------------------
def format_single_question(q, q_num, user_answer=None, show_feedback=False):  # <<< CHANGED
    """Format a single question with optional feedback"""
    feedback_html = ""
    if show_feedback and user_answer:  # <<< CHANGED: gate feedback on show_feedback
        correct = (user_answer == q.get('answer', ''))
        
        if correct:
            feedback_html = (
                "<p style='color: #4CAF50; font-weight: bold; margin-top: 10px;'>"
                "✅ Correct!</p>"
            )
        else:
            feedback_html = (
                "<p style='color: #f44336; font-weight: bold; margin-top: 10px;'>"
                f"❌ Incorrect! Correct: {q.get('answer', 'N/A')}</p>"
            )
        
        # Add explanation
        explanation = q.get('explanation', 'No explanation provided.')
        feedback_html += (
            "<p style='color: #666; font-size: 0.9em; margin-top: 5px;'>"
            f"<em>Explanation: {explanation}</em></p>"
        )
        
        # Add citations if available
        if q.get("citations"):
            citations = ', '.join(q['citations'])
            feedback_html += (
                "<p style='color: #666; font-size: 0.85em; margin-top: 5px;'>"
                f"<em>Citations: {citations}</em></p>"
            )
    
    html = f"""
    <div style='border: 1px solid #ddd; padding: 15px; margin-bottom: 10px; border-radius: 5px;'>
        <h4>Question {q_num}</h4>
        <p><strong>{q['stem']}</strong></p>
        <ul style='list-style-type: none; padding-left: 0;'>
            <li>A: {q['options']['A']}</li>
            <li>B: {q['options']['B']}</li>
            <li>C: {q['options']['C']}</li>
            <li>D: {q['options']['D']}</li>
        </ul>
        {feedback_html}
        <p style='color: #666; font-size: 0.9em; margin-top: 10px;'><em>Difficulty: {q.get('difficulty', 'Medium')}</em></p>
    </div>
    """
    return html


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
def start_quiz_ui(selected_topic, num_questions):
    # Show loading message immediately
    loading_msg = "🔄 Generating MCQs... This may take 30-60 seconds. Please wait..."
    loading_html = (
        "<div style='padding: 40px; text-align: center; background: #f5f5f5; "
        "border-radius: 10px;'><h3>🔄 Generating MCQs...</h3>"
        "<p style='font-size: 1.1em;'>Please wait while we generate your questions.</p>"
        "<p style='color: #666;'>This may take 30-60 seconds depending on the number of questions.</p></div>"
    )
    
    # Yield initial loading state to show immediately
    yield (
        loading_msg,
        {},
        loading_html,  # Show in main questions_html
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value="Generating..."),
        *([gr.update(visible=False)] * 10),  # Hide question HTMLs
        *([gr.update(visible=False)] * 10)   # Hide radio buttons
    )
    
    # Generate MCQs (this is the blocking operation)
    try:
        msg, questions = generate_mcqs_for_selected_topic(selected_topic, num_questions)
    except Exception as e:
        error_html = f"<div style='padding: 20px; color: red;'><h3>❌ Error</h3><p>{str(e)}</p></div>"
        yield (
            f"Error: {str(e)}",
            {},
            error_html,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value="Error"),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10)
        )
        return
    
    if not questions:
        error_html = "<div style='padding: 20px;'><p>No questions found.</p></div>"
        yield (
            msg,
            {},
            error_html,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=""),
            *([gr.update(visible=False)] * 10),
            *([gr.update(visible=False)] * 10)
        )
        return

    session = {
        "questions": questions,
        "current_page": 0,
        "history": [],
        "theta": 0.0,
        "questions_per_page": 10,
        "user_answers": {},
        "submitted": False  # <<< NEW
    }

    total_pages = (len(questions) + session["questions_per_page"] - 1) // session["questions_per_page"]
    
    prev_btn_visible = False  # First page, no previous
    next_btn_visible = total_pages > 1  # Show next if more than one page
    
    # Create question HTMLs and radio updates for first page
    start_idx = 0
    end_idx = min(session["questions_per_page"], len(questions))
    page_questions = questions[start_idx:end_idx]
    
    question_html_updates = []
    radio_updates = []
    
    page_header = (
        f"<div style='margin-bottom: 20px;'>"
        f"<strong>Page 1 (Questions {start_idx + 1}-{end_idx} of {len(questions)})"
        f"</strong></div>"
    )
    
    for i in range(session["questions_per_page"]):
        if i < len(page_questions):
            q = page_questions[i]
            q_num = start_idx + i + 1
            q_id = f"q_{start_idx + i}"
            user_answer = session["user_answers"].get(q_id, None)
            
            # No feedback yet (submitted=False)
            q_html = page_header if i == 0 else ""
            q_html += format_single_question(q, q_num, user_answer, show_feedback=False)
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
                    interactive=True
                )
            )
        else:
            question_html_updates.append(gr.update(value="", visible=False))
            radio_updates.append(
                gr.update(choices=[], label="", visible=False, interactive=False)
            )
    
    # Yield final result - hide main questions_html, show individual question blocks
    yield (
        msg,
        session,
        gr.update(visible=False),  # Hide main questions_html
        gr.update(visible=prev_btn_visible),
        gr.update(visible=next_btn_visible),
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
# NEW: Submit Quiz
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

with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Adaptive MCQ Quiz System (RAG + LLM)")

    with gr.Tab("Quiz"):
        gr.Markdown("### Select a topic and generate MCQs")
        
        with gr.Row():
            topic_dropdown = gr.Dropdown(
                choices=topic_options,
                label="Select Topic",
                value=None,
                interactive=True
            )
            num_q = gr.Number(value=5, label="Number of Questions", minimum=1, maximum=20)
        
        start_btn = gr.Button("Generate MCQs", variant="primary")

        msg_box = gr.Textbox(label="System Message", interactive=False)
        session_state = gr.State()

        # Page indicator
        page_indicator = gr.Textbox(label="Page Info", interactive=False)
        
        # Questions display - create question blocks with HTML and radio buttons grouped together
        question_htmls = []
        answer_radios = []
        
        # Keep a single HTML for loading/initial state
        questions_html = gr.HTML("### Questions will appear here")
        
        # Create question blocks where each question HTML is followed by its radio button
        for i in range(10):
            q_html = gr.HTML("", visible=False)
            question_htmls.append(q_html)
            
            q_radio = gr.Radio(
                choices=[],
                label="",
                visible=False,
                interactive=False
            )
            answer_radios.append(q_radio)
        
        # Pagination + Submit controls
        with gr.Row():
            prev_btn = gr.Button("◀ Previous Page", variant="secondary", visible=False)
            submit_btn = gr.Button("Submit Quiz", variant="primary", visible=True)  # <<< NEW
            next_btn = gr.Button("Next Page ▶", variant="secondary", visible=False)

        # --- Click handlers ---
        start_btn.click(
            start_quiz_ui,
            inputs=[topic_dropdown, num_q],
            outputs=[msg_box, session_state, questions_html, prev_btn, next_btn, page_indicator] + question_htmls + answer_radios
        )

        prev_btn.click(
            prev_page_ui,
            inputs=[session_state],
            outputs=[session_state, questions_html, prev_btn, next_btn, page_indicator] + question_htmls + answer_radios
        )

        next_btn.click(
            next_page_ui,
            inputs=[session_state],
            outputs=[session_state, questions_html, prev_btn, next_btn, page_indicator] + question_htmls + answer_radios
        )

        submit_btn.click(  # <<< NEW
            submit_quiz_ui,
            inputs=[session_state],
            outputs=[msg_box, session_state, questions_html, prev_btn, next_btn, page_indicator] + question_htmls + answer_radios
        )
        
        # Update session and display feedback when radio buttons change
        def make_radio_handler(idx):
            def handler(session, value):
                if not session or "questions" not in session:
                    return session, *([gr.update()] * 10)  # Return updates for all question HTMLs
                try:
                    questions_per_page = session.get("questions_per_page", 10)
                    current_page = session.get("current_page", 0)
                    submitted = session.get("submitted", False)
                    question_idx = f"q_{current_page * questions_per_page + idx}"
                    updated_session = update_answer(session, question_idx, value)
                    
                    # Update the specific question HTML
                    start_idx = current_page * questions_per_page
                    q = updated_session["questions"][start_idx + idx]
                    q_num = start_idx + idx + 1
                    user_answer = updated_session.get("user_answers", {}).get(question_idx, None)
                    
                    # Only show feedback after submission
                    page_header = (
                        f"<div style='margin-bottom: 20px;'>"
                        f"<strong>Page {current_page + 1}</strong></div>"
                    ) if idx == 0 else ""
                    updated_q_html = page_header + format_single_question(
                        q, q_num, user_answer, show_feedback=submitted
                    )
                    
                    html_updates = []
                    for i in range(10):
                        if i == idx:
                            html_updates.append(gr.update(value=updated_q_html, visible=True))
                        else:
                            html_updates.append(gr.update())
                    
                    return updated_session, *html_updates
                except Exception as e:
                    print(f"[GUI] Error in radio handler: {e}")
                    return session, *([gr.update()] * 10)
            return handler
        
        for i, radio in enumerate(answer_radios):
            radio.change(
                make_radio_handler(i),
                inputs=[session_state, radio],
                outputs=[session_state] + question_htmls
            )

    with gr.Tab("Metrics"):
        metrics_btn = gr.Button("Generate Metrics")
        out_text = gr.Textbox()
        out_json = gr.JSON()
        metrics_btn.click(generate_metrics_ui, outputs=[out_text, out_json])

demo.queue()  # Enable queueing for generators to work properly
demo.launch()
