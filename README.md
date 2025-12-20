# MCQ Generator: LLM + RAG-based Adaptive Question System

## Introduction
This project implements a high-quality, syllabus-aligned multiple-choice question (MCQ) generation and adaptive practice system using **Large Language Models (LLMs)** combined with **Retrieval-Augmented Generation (RAG)**. The system supports HSSC Pre-Engineering Mathematics, Physics, and Chemistry, as well as SAT-level English, providing:

- **Static and adaptive MCQ generation**  
- **Citation-backed explanations**  
- **STEM answer verification**  
- **Psychometric evaluation** (difficulty, discrimination, topic coverage)  
- **Adaptive practice based on learner ability (θ)**  
- **Web-based interactive interface**  

The system is fully reproducible, including corpus preprocessing, FAISS vector indexing, MCQ generation, and evaluation.

---


---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/smolblud/mcq-generator.git
cd mcq-generator


### 2. Create Virtual Environment & Install Dependencies

python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

### 3. Data
The mcq-generator/data/ folder contains subfolders for each subject:

math/ → Math textbooks 

physics/ → Physics textbooks

English/ → SAT textbook


### 4. Preprocess Corpus & Build Index
python preprocess_and_index.py

### 5. Run the GUI
python gui.py


### 6. Generate MCQs (Optional CLI)
python generate_mcqs.py --config config.yaml

### 7. Run Evaluation Metrics (Optional CLI)
python evaluation_metrics.py --config config.yaml

