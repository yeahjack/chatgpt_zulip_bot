"""Tests for the ChatGPT Zulip Bot."""

import pytest
import re
import os
from unittest.mock import patch, MagicMock

import tiktoken

from chatgpt import (
    ChatBot,
    get_model_specs,
    load_course_materials,
    extract_text_content,
    get_week_context,
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
    _, _, encoding_name = get_model_specs("gpt-4o")
    enc = tiktoken.get_encoding(encoding_name)
    assert enc.name == "o200k_base"
    
    _, _, encoding_name = get_model_specs("gpt-4-turbo")
    enc = tiktoken.get_encoding(encoding_name)
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


def test_get_week_context():
    """Test week context generation."""
    mock_materials = {
        "week1": {
            "1.intro.typ": "= Week 1\nIntroduction to DFA"
        },
        "week2": {
            "2.intro.typ": "= Week 2\nNFA and conversions"
        }
    }
    
    # Test specific week
    context = get_week_context(mock_materials, week_num=1)
    assert "WEEK 1" in context
    assert "DFA" in context
    
    # Test non-existent week
    context_empty = get_week_context(mock_materials, week_num=99)
    assert context_empty == ""


def test_get_week_context_max_chars():
    """Test context truncation."""
    mock_materials = {
        "week1": {
            "1.intro.typ": "= Week 1\n" + "A" * 10000
        }
    }
    context = get_week_context(mock_materials, week_num=1, max_chars=100)
    assert len(context) <= 100


def test_get_week_context_with_file_patterns():
    """Test context filtering by file patterns."""
    mock_materials = {
        "week1": {
            "1.learning-sheet.typ": "Learning sheet content",
            "1.validation.typ": "Validation content",
            "1.test.typ": "Test content (should be filtered)",
        }
    }
    
    # With patterns
    context = get_week_context(mock_materials, week_num=1, file_patterns=["*learning-sheet*"])
    assert "Learning sheet" in context
    assert "Validation" not in context
    assert "Test content" not in context


# =============================================================================
# ChatBot Tests
# =============================================================================

def test_chatbot_commands_with_mock():
    """Test bot commands using mock (no API key required)."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        MockClient.return_value = MagicMock()
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key-for-testing")
        
        # Test /reset command
        result = bot.get_dm_response("user1", "/reset")
        assert "cleared" in result.lower()
        
        # Test /week command (no materials loaded, so need to check response)
        result = bot.get_dm_response("user1", "/week 99")
        assert "not found" in result.lower() or "available" in result.lower()


def test_chatbot_model_detection():
    """Test that model specs are correctly detected."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        MockClient.return_value = MagicMock()
        
        # GPT-4o
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        assert bot.max_output_tokens == 16_384
        assert bot.encoding.name == "o200k_base"
        
        # GPT-5.2
        bot = ChatBot(model="gpt-5.2", api_key="fake-key")
        assert bot.max_output_tokens == 32_768
        
        # Custom max_output_tokens override
        bot = ChatBot(model="gpt-4o", api_key="fake-key", max_output_tokens=8000)
        assert bot.max_output_tokens == 8000


def test_chatbot_token_counting():
    """Test token counting via encoding attribute."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        MockClient.return_value = MagicMock()
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        
        # Token counting is done via the encoding attribute
        tokens = len(bot.encoding.encode("Hello, world!"))
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
        # Set a week first (required for DM mode)
        bot.user_weeks["user1"] = 1
        bot.course_materials = {"week1": {"test.typ": "test content"}}
        
        result = bot.get_dm_response("user1", "What is a DFA?")
        
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
        # Set a week first (required for DM mode)
        bot.user_weeks["user1"] = 1
        bot.course_materials = {"week1": {"test.typ": "test content"}}
        
        # First message
        bot.get_dm_response("user1", "Hello")
        assert bot.user_response_ids["user1"] == "resp_123"
        
        # Second message should include previous_response_id
        bot.get_dm_response("user1", "Follow up question")
        second_call = mock_client.responses.create.call_args_list[1]
        assert second_call.kwargs["previous_response_id"] == "resp_123"


def test_chatbot_error_handling():
    """Test error handling when Responses API fails."""
    with patch('chatgpt.OpenAIClient') as MockClient:
        mock_client = MagicMock()
        MockClient.return_value = mock_client
        
        # Make responses.create fail
        mock_client.responses.create.side_effect = Exception("API error")
        
        bot = ChatBot(model="gpt-4o", api_key="fake-key")
        # Set a week first (required for DM mode)
        bot.user_weeks["user1"] = 1
        bot.course_materials = {"week1": {"test.typ": "test content"}}
        
        result = bot.get_dm_response("user1", "Hello")
        
        # Should return error message, not crash
        assert "Error" in result or "error" in result


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


# =============================================================================
# Zulip Mention Pattern Tests
# =============================================================================

class TestMentionPatterns:
    """Test Zulip mention pattern matching."""
    
    @pytest.fixture
    def mention_pattern(self):
        """Create mention pattern for bot named 'ChatGPT'."""
        bot_name = "ChatGPT"
        return rf"@_?\*\*{re.escape(bot_name)}(\|\d+)?\*\*"
    
    def test_regular_mention(self, mention_pattern):
        """Test regular @**BotName** mention."""
        message = "@**ChatGPT** What is a DFA?"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == "What is a DFA?"
    
    def test_quote_reply_mention(self, mention_pattern):
        """Test quote reply @_**BotName|ID** mention."""
        message = "@_**ChatGPT|132** Can you explain more?"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == "Can you explain more?"
    
    def test_mention_with_id_no_underscore(self, mention_pattern):
        """Test @**BotName|ID** mention (without underscore)."""
        message = "@**ChatGPT|132** Another question"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == "Another question"
    
    def test_mention_not_at_start(self, mention_pattern):
        """Test that mention not at start doesn't match."""
        message = "Hello @**ChatGPT** how are you?"
        assert not re.match(mention_pattern, message)
    
    def test_mention_with_multiline(self, mention_pattern):
        """Test mention with multiline message."""
        message = "@**ChatGPT** First line\nSecond line\nThird line"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == "First line\nSecond line\nThird line"
    
    def test_mention_only(self, mention_pattern):
        """Test message that is only a mention."""
        message = "@**ChatGPT**"
        assert re.match(mention_pattern, message)
        cleaned = re.sub(mention_pattern, "", message).strip()
        assert cleaned == ""
    
    def test_wrong_bot_name(self, mention_pattern):
        """Test that wrong bot name doesn't match."""
        message = "@**OtherBot** Hello"
        assert not re.match(mention_pattern, message)
    
    def test_quote_reply_with_quote_block(self, mention_pattern):
        """Test stripping quote blocks from quote-replies."""
        message = '''@_**ChatGPT|132** [said](https://example.com/link):
````quote
> original question

Previous bot response here...
------
Tokens: 1000 (input) + 100 (output) = 1100
````

What is an NFA?'''
        
        # Strip mention
        prompt = re.sub(mention_pattern, "", message).strip()
        
        # Strip quote block
        quote_pattern = r'\[said\]\([^)]+\):\s*(`{3,})quote\s.*?\1'
        prompt = re.sub(quote_pattern, "", prompt, flags=re.DOTALL).strip()
        
        assert prompt == "What is an NFA?"
    
    def test_nested_quote_reply(self, mention_pattern):
        """Test stripping nested quote blocks."""
        message = '''@_**ChatGPT|132** [said](https://example.com/link):
````quote
@_**ChatGPT|132** [said](https://example.com/other):
```quote
> first question
First response
```
Second question
````

Third question here.'''
        
        # Strip ONLY FIRST mention (count=1 to preserve inner mentions)
        original_message = re.sub(mention_pattern, "", message, count=1).strip()
        
        # Inner ChatGPT mention should be preserved
        assert "@_**ChatGPT|132** [said]" in original_message
        
        # Strip quote block for prompt
        quote_pattern = r'\[said\]\([^)]+\):\s*(`{3,})quote\s.*?\1'
        prompt = re.sub(quote_pattern, "", original_message, flags=re.DOTALL).strip()
        
        assert prompt == "Third question here."
    
    def test_preserve_inner_bot_mentions(self, mention_pattern):
        """Test that bot mentions inside quotes are preserved."""
        message = '''@**ChatGPT** @_**jinguoliu|8** [said](https://example.com):
````quote
@_**ChatGPT|132** [said](https://example.com/prev):
```quote
> original question
Bot response
```
Follow up
````

New question'''
        
        # Strip only first bot mention
        original_message = re.sub(mention_pattern, "", message, count=1).strip()
        
        # Outer user attribution preserved
        assert "@_**jinguoliu|8** [said]" in original_message
        # Inner bot mention preserved
        assert "@_**ChatGPT|132** [said]" in original_message
    
    def test_user_quote_with_bot_mention(self, mention_pattern):
        """Test when user quotes another user's message that contains bot mention."""
        # User quotes their own previous message that mentioned the bot
        message = '''@**ChatGPT** @_**jinguoliu|8** [said](https://zulip.hkust-gz.edu.cn/#narrow/channel/128-Teaching-DSAA3071-2026-Spring/topic/GPT.20test/near/95980):
```quote
@**ChatGPT** You are so good.
```

What is a DFA?'''
        
        # Strip only the first bot mention (the one addressing the bot)
        prompt = re.sub(mention_pattern, "", message, count=1).strip()
        
        # The user attribution and inner bot mention should be preserved
        assert "@_**jinguoliu|8** [said]" in prompt
        
        # Strip quote block to get the actual question
        quote_pattern = r'@_\*\*[^*]+\*\*\s*\[said\]\([^)]+\):\s*(`{3,})quote\s.*?\1'
        prompt = re.sub(quote_pattern, "", prompt, flags=re.DOTALL).strip()
        
        assert prompt == "What is a DFA?"


# =============================================================================
# Response Formatting Tests
# =============================================================================

class TestResponseFormatting:
    """Test response formatting with quotes and mentions."""
    
    def test_quote_message_simple(self):
        """Test quoting a simple message."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            result = bot._quote_message("Hello world")
            assert result == "````quote\nHello world\n````"
    
    def test_quote_message_with_triple_backticks(self):
        """Test quoting a message containing triple backticks."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            message = "```quote\ninner content\n```"
            result = bot._quote_message(message)
            # Should use 4 backticks since content has 3
            assert result.startswith("````quote\n")
            assert result.endswith("\n````")
            assert "```quote" in result
    
    def test_quote_message_with_four_backticks(self):
        """Test quoting a message containing four backticks."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            message = "````quote\ninner content\n````"
            result = bot._quote_message(message)
            # Should use 5 backticks since content has 4
            assert result.startswith("`````quote\n")
            assert result.endswith("\n`````")
    
    def test_quote_message_nested(self):
        """Test quoting a nested quote message."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            message = '''@_**user|123** [said](url):
````quote
@_**bot|456** [said](url2):
```quote
original
```
reply
````

new question'''
            result = bot._quote_message(message)
            # Should use 5 backticks since content has 4
            assert result.startswith("`````quote\n")
            assert result.endswith("\n`````")
            # Content preserved
            assert "@_**user|123**" in result
            assert "@_**bot|456**" in result
    
    def test_format_response_with_mention(self):
        """Test response formatting with user mention."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            
            # Mock usage object
            usage = MagicMock()
            usage.input_tokens = 100
            usage.output_tokens = 50
            usage.total_tokens = 150
            
            result = bot._format_response(
                question="What is DFA?",
                reply="A DFA is a deterministic finite automaton.",
                usage=usage,
                sender_name="jinguoliu",
                sender_id=8,
                message_url="#narrow/channel/128/topic/test/near/123"
            )
            
            # Check mention format
            assert "@_**jinguoliu|8** [said](#narrow/channel/128/topic/test/near/123):" in result
            # Check quote block
            assert "````quote" in result
            assert "What is DFA?" in result
            # Check reply
            assert "A DFA is a deterministic finite automaton." in result
            # Check tokens
            assert "Tokens: 100 (input) + 50 (output) = 150" in result
    
    def test_format_response_without_url(self):
        """Test response formatting without message URL."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            
            usage = MagicMock()
            usage.input_tokens = 100
            usage.output_tokens = 50
            usage.total_tokens = 150
            
            result = bot._format_response(
                question="Hello",
                reply="Hi there!",
                usage=usage,
                sender_name="user",
                sender_id=123
            )
            
            # Should have mention without [said](url)
            assert "@_**user|123**:" in result
            assert "[said]" not in result
    
    def test_format_response_without_sender(self):
        """Test response formatting without sender info."""
        from chatgpt import ChatBot
        with patch('chatgpt.OpenAIClient'):
            bot = ChatBot(model="gpt-4o", api_key="fake")
            
            usage = MagicMock()
            usage.input_tokens = 100
            usage.output_tokens = 50
            usage.total_tokens = 150
            
            result = bot._format_response(
                question="Hello",
                reply="Hi there!",
                usage=usage
            )
            
            # Should not have mention
            assert "@_**" not in result
            # Should still have quote and reply
            assert "````quote" in result
            assert "Hello" in result
            assert "Hi there!" in result
