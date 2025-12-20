import json
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np

# Import the module to be tested
# (Ensure mcq_generator.py is in the same folder or python path)
import mcq_generator

# -------------------------------------------------------------------
# Fixtures: Mock Data and Objects
# -------------------------------------------------------------------

@pytest.fixture
def mock_topic_blueprint():
    """Mock DataFrame for topic blueprint."""
    data = {
        "subject": ["Physics", "Physics", "Chemistry"],
        "unit": ["Unit 1", "Unit 2", "Unit 1"],
        "topic": ["Kinematics", "Dynamics", "Atoms"],
        "start_page": [1, 10, 20],
        "end_page": [9, 19, 30],
        "source_file": ["phys.pdf", "phys.pdf", "chem.pdf"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_mcq_json_response():
    """Mock valid JSON response from LLM."""
    return json.dumps([
        {
            "stem": "What is speed?",
            "options": {"A": "Scalar", "B": "Vector", "C": "Nothing", "D": "All"},
            "answer": "A",
            "explanation": "Speed has magnitude only.",
            "citations": ["phys_Kinematics"],
            "difficulty": "Medium"
        },
        {
            "stem": "What is velocity?",
            "options": {"A": "Scalar", "B": "Vector", "C": "Nothing", "D": "All"},
            "answer": "B",
            "explanation": "Velocity has direction.",
            "citations": ["phys_Kinematics"],
            "difficulty": "Medium"
        }
    ])

# -------------------------------------------------------------------
# Tests for Helper Functions
# -------------------------------------------------------------------

def test_parse_llm_json_clean():
    """Test parsing clean JSON string."""
    raw = '[{"key": "value"}]'
    result = mcq_generator.parse_llm_json(raw)
    assert isinstance(result, list)
    assert result[0]["key"] == "value"

def test_parse_llm_json_markdown_fences():
    """Test parsing JSON string wrapped in markdown code blocks."""
    raw = '```json\n[{"key": "value"}]\n```'
    result = mcq_generator.parse_llm_json(raw)
    assert isinstance(result, list)
    assert result[0]["key"] == "value"

def test_get_topics_for_subject(mock_topic_blueprint):
    """Test filtering topics by subject."""
    with patch('mcq_generator.load_topic_blueprint', return_value=mock_topic_blueprint):
        topics = mcq_generator.get_topics_for_subject("Physics")
        assert len(topics) == 2
        assert "Kinematics" in topics
        assert "Dynamics" in topics
        
        # Test case insensitivity
        topics_lower = mcq_generator.get_topics_for_subject("physics")
        assert len(topics_lower) == 2

# -------------------------------------------------------------------
# Tests for Logic with Mocks (RAG + LLM)
# -------------------------------------------------------------------

@patch('mcq_generator.index')  # Mock FAISS index
@patch('mcq_generator.emb_model')  # Mock Embedding Model
@patch('mcq_generator.chunks', [{"text": "Sample text", "source_file": "doc", "topic": "test"}]) # Mock Metadata
def test_retrieve_context(mock_emb_model, mock_index):
    """Test context retrieval logic."""
    # Setup mock returns
    mock_emb_model.encode.return_value = np.array([[0.1, 0.2]], dtype="float32")
    # Search returns (distances, indices)
    mock_index.search.return_value = (np.array([[0.5]]), np.array([[0]])) 
    
    context = mcq_generator.retrieve_context("Physics", "Kinematics")
    
    # Assertions
    assert isinstance(context, str)
    assert "[doc_test] Sample text" in context
    mock_emb_model.encode.assert_called_once()
    mock_index.search.assert_called_once()


@patch('mcq_generator.retrieve_context')
@patch('mcq_generator.llm')
def test_generate_mcqs_for_topic_success(mock_llm, mock_retrieve, sample_mcq_json_response):
    """Test successful MCQ generation for a single topic."""
    # Setup
    mock_retrieve.return_value = "Retrieved Context Data"
    mock_llm.invoke.return_value.content = sample_mcq_json_response
    
    questions = mcq_generator.generate_mcqs_for_topic("Physics", "Kinematics", n_questions=2)
    
    # Assertions
    assert len(questions) == 2
    assert questions[0]["stem"] == "What is speed?"
    assert questions[0]["difficulty"] == "Medium"
    
    # Verify LLM was called
    mock_llm.invoke.assert_called_once()


@patch('mcq_generator.retrieve_context')
def test_generate_mcqs_for_topic_no_context(mock_retrieve):
    """Test empty return when no context is found."""
    mock_retrieve.return_value = "" # Empty context
    
    questions = mcq_generator.generate_mcqs_for_topic("Physics", "Void", n_questions=2)
    
    assert questions == []

@patch('mcq_generator.retrieve_context')
@patch('mcq_generator.llm')
def test_generate_mcqs_for_topic_bad_json(mock_llm, mock_retrieve):
    """Test handling of invalid JSON from LLM."""
    mock_retrieve.return_value = "Context"
    mock_llm.invoke.return_value.content = "I cannot generate JSON"
    
    questions = mcq_generator.generate_mcqs_for_topic("Physics", "Kinematics", 2)
    assert questions == [] # Should handle error gracefully

# -------------------------------------------------------------------
# Tests for Subject-Wise Generation
# -------------------------------------------------------------------

@patch('mcq_generator.get_topics_for_subject')
@patch('mcq_generator.generate_mcqs_for_topic')
def test_generate_mcqs_for_subject(mock_gen_topic, mock_get_topics):
    """Test logic for aggregating questions across topics."""
    # Setup
    mock_get_topics.return_value = ["Topic A", "Topic B"]
    
    # Mock return values for Topic A and T  opic B calls
    # Assume we ask for 5 total, it splits roughly 3 and 2
    mock_gen_topic.side_effect = [
        [{"stem": "Q1"}, {"stem": "Q2"}, {"stem": "Q3"}], # Result for Topic A
        [{"stem": "Q4"}, {"stem": "Q5"}]                  # Result for Topic B
    ]
    
    result = mcq_generator.generate_mcqs_for_subject("Physics", n_questions=5)
    
    assert len(result) == 5
    assert mock_gen_topic.call_count >= 1

# -------------------------------------------------------------------
# Integration-style Test (Mocking Blueprint Iteration)
# -------------------------------------------------------------------

@patch('mcq_generator.load_topic_blueprint')
@patch('mcq_generator.generate_mcqs_for_topic')
def test_generate_random_mcqs_from_blueprint(mock_gen_topic, mock_load_bp, mock_topic_blueprint):
    """Test the main loop that generates 30 random MCQs."""
    mock_load_bp.return_value = mock_topic_blueprint
    
    # Mock generator to return 1 question every time it's called
    mock_gen_topic.return_value = [{"stem": "Sample Q", "options": {}, "answer": "A"}]
    
    # Set global TOTAL_QUESTIONS temporarily lower for speed
    original_total = mcq_generator.TOTAL_QUESTIONS
    mcq_generator.TOTAL_QUESTIONS = 3
    
    try:
        results = mcq_generator.generate_random_mcqs_from_blueprint(difficulty="Easy")
        
        # We requested 3 total questions (via global override)
        assert len(results) == 3
        # Check integrity
        assert results[0]["stem"] == "Sample Q"
        
    finally:
        # Restore global variable
        mcq_generator.TOTAL_QUESTIONS = original_total