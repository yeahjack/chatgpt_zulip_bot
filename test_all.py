"""Tests for the ChatGPT Zulip Bot."""

import pytest
import os
from unittest.mock import patch, MagicMock

from chatgpt import (
    ChatBot,
    get_model_specs,
    get_encoding,
    load_course_materials,
    extract_text_content,
    get_course_context,
    MODEL_SPECS,
)

# =============================================================================
# Model Configuration Tests
# =============================================================================

def test_get_model_specs_exact_match():
    """Test exact model name matching."""
    context, max_output, encoding = get_model_specs("gpt-4o")
    assert context == 128_000
    assert max_output == 16_384
    assert encoding == "o200k_base"


def test_get_model_specs_prefix_match():
    """Test prefix matching for versioned models."""
    context, max_output, encoding = get_model_specs("gpt-4o-2024-12-01")
    assert context == 128_000
    assert encoding == "o200k_base"


def test_get_model_specs_family_fallback():
    """Test fallback based on model family."""
    # GPT-5 family
    context, max_output, _ = get_model_specs("gpt-5.3-preview")
    assert context == 400_000
    
    # GPT-4.1 family
    context, max_output, _ = get_model_specs("gpt-4.1-turbo")
    assert context == 1_047_576
    
    # o-series
    context, max_output, _ = get_model_specs("o3-large")
    assert context == 200_000


def test_get_model_specs_unknown():
    """Test fallback for completely unknown models."""
    context, max_output, encoding = get_model_specs("unknown-model-xyz")
    assert context == 128_000  # default
    assert max_output == 16_384  # default


def test_get_encoding():
    """Test encoding retrieval for different models."""
    enc = get_encoding("gpt-4o")
    assert enc.name == "o200k_base"
    
    enc = get_encoding("gpt-4-turbo")
    assert enc.name == "cl100k_base"


def test_all_known_models_have_valid_specs():
    """Verify all models in MODEL_SPECS have valid configurations."""
    for model, (context, max_output, encoding) in MODEL_SPECS.items():
        assert context > 0, f"{model} has invalid context window"
        assert max_output > 0, f"{model} has invalid max output"
        assert encoding in ["cl100k_base", "o200k_base"], f"{model} has invalid encoding"
        assert max_output <= context, f"{model} max_output exceeds context window"


# =============================================================================
# Course Materials Tests
# =============================================================================

def test_load_course_materials():
    """Test course materials loading."""
    materials = load_course_materials()
    assert isinstance(materials, dict)


def test_load_course_materials_missing_dir():
    """Test loading from non-existent directory."""
    materials = load_course_materials("/nonexistent/path")
    assert materials == {}


def test_extract_text_content():
    """Test text extraction from typst content."""
    typst_sample = """#import "@preview/touying:0.6.1": *
#let globalvars = state("t", 0)

= Introduction
== The Simplest Computer

A *finite automaton* is the simplest possible computer.

- States: Red, Green, Yellow
- Alphabet: {tick}
"""
    result = extract_text_content(typst_sample)
    assert "Introduction" in result
    assert "finite automaton" in result
    assert "#import" not in result
    assert "#let" not in result


def test_get_course_context():
    """Test course context generation."""
    mock_materials = {
        "week1": {
            "week1/1.intro.typ": "= Week 1\nIntroduction to DFA"
        },
        "week2": {
            "week2/2.intro.typ": "= Week 2\nNFA and conversions"
        }
    }
    
    # Test specific week
    context = get_course_context(mock_materials, week="week1")
    assert "week1" in context.lower()
    
    # Test all materials
    context_all = get_course_context(mock_materials)
    assert isinstance(context_all, str)


def test_get_course_context_max_chars():
    """Test context truncation."""
    mock_materials = {
        "week1": {
            "week1/1.intro.typ": "= Week 1\n" + "A" * 10000
        }
    }
    context = get_course_context(mock_materials, week="week1", max_chars=100)
    assert len(context) <= 100


# =============================================================================
# ChatBot Tests
# =============================================================================

def test_chatbot_commands_with_mock():
    """Test bot commands using mock (no API key required)."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        MockClient.return_value = MagicMock()
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key-for-testing")
        
        # Test /help command
        result = bot.get_response("user1", "/help")
        assert "DSAA3071" in result
        assert "Theory of Computation" in result
        
        # Test /end command
        result = bot.get_response("user1", "/end")
        assert "cleared" in result.lower()
        
        # Test /week command
        result = bot.get_response("user1", "/week 1")
        assert "week" in result.lower()
        
        # Test invalid week
        result = bot.get_response("user1", "/week 99")
        assert "not found" in result.lower() or "available" in result.lower()


def test_chatbot_model_detection():
    """Test that model specs are correctly detected."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        MockClient.return_value = MagicMock()
        
        # GPT-4o
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        assert bot.context_window == 128_000
        assert bot.max_output_tokens == 16_384
        
        # GPT-5.2
        bot = ChatBot(model="gpt-5.2", api_key="fake-key")
        assert bot.context_window == 400_000
        assert bot.max_output_tokens == 32_768
        
        # Custom max_output_tokens override
        bot = ChatBot(model="gpt-4o", api_key="fake-key", max_output_tokens=8000)
        assert bot.max_output_tokens == 8000


def test_chatbot_token_counting():
    """Test token counting functionality."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        MockClient.return_value = MagicMock()
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        
        tokens = bot.count_tokens("Hello, world!")
        assert tokens > 0
        assert isinstance(tokens, int)


def test_chatbot_responses_api_call():
    """Test that the bot calls the Responses API correctly."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        
        # Mock the responses.create method
        mock_response = MagicMock()
        mock_response.id = "resp_123"
        mock_response.output_text = "This is a test response about DFA."
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.usage.total_tokens = 150
        mock_client.responses.create.return_value = mock_response
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        result = bot.get_response("user1", "What is a DFA?")
        
        # Verify the response
        assert "test response" in result
        assert "Tokens:" in result
        
        # Verify the API was called with correct params
        mock_client.responses.create.assert_called_once()
        call_kwargs = mock_client.responses.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["input"] == "What is a DFA?"
        assert "instructions" in call_kwargs
        assert call_kwargs["store"] == True


def test_chatbot_conversation_continuity():
    """Test that conversation IDs are tracked for multi-turn conversations."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        
        # Mock responses
        mock_response1 = MagicMock()
        mock_response1.id = "resp_123"
        mock_response1.output_text = "First response"
        mock_response1.usage.input_tokens = 100
        mock_response1.usage.output_tokens = 50
        mock_response1.usage.total_tokens = 150
        
        mock_response2 = MagicMock()
        mock_response2.id = "resp_456"
        mock_response2.output_text = "Second response"
        mock_response2.usage.input_tokens = 150
        mock_response2.usage.output_tokens = 60
        mock_response2.usage.total_tokens = 210
        
        mock_client.responses.create.side_effect = [mock_response1, mock_response2]
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        
        # First message
        bot.get_response("user1", "Hello")
        assert bot.user_response_ids["user1"] == "resp_123"
        
        # Second message should include previous_response_id
        bot.get_response("user1", "Follow up question")
        second_call = mock_client.responses.create.call_args_list[1]
        assert second_call.kwargs["previous_response_id"] == "resp_123"


def test_chatbot_fallback_to_chat_completions():
    """Test fallback to Chat Completions when Responses API fails."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        
        # Make responses.create fail
        mock_client.responses.create.side_effect = Exception("Responses API not available")
        
        # Mock chat completions fallback
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Fallback response"
        mock_completion.usage.prompt_tokens = 100
        mock_completion.usage.completion_tokens = 50
        mock_completion.usage.total_tokens = 150
        mock_client.chat.completions.create.return_value = mock_completion
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        result = bot.get_response("user1", "Hello")
        
        assert "Fallback response" in result
        mock_client.chat.completions.create.assert_called_once()


# =============================================================================
# Integration Tests (require API key)
# =============================================================================

@pytest.fixture
def chatbot_with_api():
    """Create ChatBot with real API key (skips if not available)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    
    model = os.environ.get("MODEL", "gpt-4o")
    return ChatBot(model=model, api_key=api_key)


def test_real_api_response(chatbot_with_api):
    """Test actual API response (requires API key)."""
    response = chatbot_with_api.get_response("test_user", "What is a DFA? Answer in one sentence.")
    assert isinstance(response, str)
    assert len(response) > 0
    # Clean up
    chatbot_with_api.user_response_ids.pop("test_user", None)
