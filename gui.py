import gradio as gr
import json
from pathlib import Path
from mcq_generator import (
    generate_mcqs_for_topic,
    load_topic_blueprint,
    iter_random_unique_topics
)
from metrics import generate_metrics_summary, plot_metrics

MCQ_FOLDER = "MCQ_final_output"
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]

Path(MCQ_FOLDER).mkdir(exist_ok=True)


# ------------------------
# Generate MCQs
# ------------------------
def generate_user_mcqs_ui(num_questions: int = 10, difficulty: str = "Medium"):
    df = load_topic_blueprint()
    all_questions = []
    questions_remaining = num_questions

    for subject, topic in iter_random_unique_topics(df):
        if questions_remaining <= 0:
            break
        n_for_topic = min(5, questions_remaining)
        topic_questions = generate_mcqs_for_topic(subject, topic, n_questions=n_for_topic)
        # Filter by difficulty
        topic_questions = [q for q in topic_questions if q.get("difficulty", "Medium") == difficulty]
        all_questions.extend(topic_questions)
        questions_remaining = num_questions - len(all_questions)
        if questions_remaining <= 0:
            break

    all_questions = all_questions[:num_questions]
    out_file = Path(MCQ_FOLDER) / "mcqs_generated.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_questions, f, indent=2, ensure_ascii=False)

    return f"Generated {len(all_questions)} MCQs", all_questions


# ------------------------
# Start quiz
# ------------------------
def start_quiz_ui(num_questions, difficulty):
    msg, questions = generate_user_mcqs_ui(num_questions, difficulty)
    if not questions:
        return msg, {}, "No questions found.", gr.update(choices=[], value=None)

    session = {
        "questions": questions,
        "current_idx": 0,
        "history": [],
        "theta": 0.0,
        "difficulty": difficulty
    }

    q = questions[0]
    stem = q["stem"]
    options = [f"A: {q['options']['A']}", f"B: {q['options']['B']}",
               f"C: {q['options']['C']}", f"D: {q['options']['D']}"]

    return msg, session, stem, gr.update(choices=options, value=None)


# ------------------------
# Answer + adaptive step
# ------------------------
def answer_question_ui(session, selected_option):
    if session is None or session == {}:
        return session, "Click 'Start Quiz' first.", gr.update(choices=[], value=None), False

    idx = session["current_idx"]
    q = session["questions"][idx]

    if selected_option is None:
        return session, "Please select an answer first.", gr.update(choices=[], value=None), False

    selected_letter = selected_option.split(":")[0]
    correct = (selected_letter == q["answer"])
    session["theta"] += 0.15 if correct else -0.15

    # adaptive difficulty
    current_level = DIFFICULTY_LEVELS.index(session["difficulty"])
    if correct and current_level < 2:
        session["difficulty"] = DIFFICULTY_LEVELS[current_level+1]
    elif not correct and current_level > 0:
        session["difficulty"] = DIFFICULTY_LEVELS[current_level-1]

    # record history
    session["history"].append({
        "step": idx+1,
        "stem": q["stem"],
        "student_correct": correct,
        "selected_option": selected_letter,
        "correct_option": q["answer"],
        "theta_after": session["theta"],
        "difficulty_after": session["difficulty"],
        "explanation": q.get("explanation", ""),
        "citations": q.get("citations", [])
    })

    session["current_idx"] += 1
    done = session["current_idx"] >= len(session["questions"])

    # Feedback with correct option and citation
    feedback_msg = f"✅ Correct!" if correct else f"❌ Incorrect! Correct: {q['answer']}\n"
    feedback_msg += f"Explanation: {q.get('explanation', 'No explanation provided.')}\n"
    if q.get("citations"):
        feedback_msg += f"Citations: {', '.join(q['citations'])}"

    if done:
        return session, "🎉 Quiz Finished!\n" + feedback_msg, gr.update(choices=[], value=None), True

    next_q = session["questions"][session["current_idx"]]
    stem = next_q["stem"]
    options = [f"A: {next_q['options']['A']}", f"B: {next_q['options']['B']}",
               f"C: {next_q['options']['C']}", f"D: {next_q['options']['D']}"]

    return session, stem, gr.update(choices=options, value=None), False


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
with gr.Blocks() as demo:
    gr.Markdown("## 🎯 Adaptive MCQ Quiz System (RAG + LLM)")

    with gr.Tab("Quiz"):
        with gr.Row():
            num_q = gr.Number(value=5, label="Number of Questions")
            diff = gr.Dropdown(DIFFICULTY_LEVELS, value="Medium", label="Start Difficulty")
            start_btn = gr.Button("Start Quiz", variant="primary")

        msg_box = gr.Textbox(label="System Message")
        session_state = gr.State()

        question_md = gr.Markdown("### Question will appear here")
        options_radio = gr.Radio(choices=[], label="Select an option")
        submit_btn = gr.Button("Submit Answer", variant="secondary")
        done_box = gr.Textbox(label="Status")

        start_btn.click(
            start_quiz_ui,
            inputs=[num_q, diff],
            outputs=[msg_box, session_state, question_md, options_radio]
        )

        submit_btn.click(
            answer_question_ui,
            inputs=[session_state, options_radio],
            outputs=[session_state, question_md, options_radio, done_box]
        )

    with gr.Tab("Metrics"):
        metrics_btn = gr.Button("Generate Metrics")
        out_text = gr.Textbox()
        out_json = gr.JSON()
        metrics_btn.click(generate_metrics_ui, outputs=[out_text, out_json])

demo.launch()
