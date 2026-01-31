# chatgpt.py
"""
OpenAI-powered course assistant for DSAA3071 Theory of Computation.
Uses the modern Responses API (recommended by OpenAI for new projects).

Modes:
- Stream: RAG mode (vector store search, no chaining)
- DM: Weekly mode (user specifies week, with chaining)
"""

from openai import OpenAI as OpenAIClient
import tiktoken
import logging
import os
import glob
import fnmatch
import re

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_SPECS = {
    # GPT-5.x series
    "gpt-5.2": (400_000, 32_768, "cl100k_base"),
    "gpt-5.1": (400_000, 32_768, "cl100k_base"),
    "gpt-5": (400_000, 32_768, "cl100k_base"),
    # GPT-4.1 series  
    "gpt-4.1": (1_047_576, 32_768, "cl100k_base"),
    "gpt-4.1-mini": (1_047_576, 32_768, "cl100k_base"),
    # GPT-4o series
    "gpt-4o": (128_000, 16_384, "o200k_base"),
    "gpt-4o-mini": (128_000, 16_384, "o200k_base"),
    "gpt-4o-2024-11-20": (128_000, 16_384, "o200k_base"),
    "gpt-4o-2024-08-06": (128_000, 16_384, "o200k_base"),
    "gpt-4o-2024-05-13": (128_000, 4_096, "o200k_base"),
    # GPT-4 Turbo
    "gpt-4-turbo": (128_000, 4_096, "cl100k_base"),
    "gpt-4-turbo-preview": (128_000, 4_096, "cl100k_base"),
    # GPT-4 base
    "gpt-4": (8_192, 4_096, "cl100k_base"),
    "gpt-4-32k": (32_768, 4_096, "cl100k_base"),
    # GPT-3.5 Turbo
    "gpt-3.5-turbo": (16_385, 4_096, "cl100k_base"),
    "gpt-3.5-turbo-16k": (16_385, 4_096, "cl100k_base"),
    # o-series (reasoning models)
    "o1": (200_000, 100_000, "o200k_base"),
    "o1-mini": (128_000, 65_536, "o200k_base"),
    "o1-preview": (128_000, 32_768, "o200k_base"),
    "o3": (200_000, 100_000, "o200k_base"),
    "o3-mini": (200_000, 100_000, "o200k_base"),
    "o4-mini": (200_000, 100_000, "o200k_base"),
}

DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_MAX_OUTPUT = 16_384
DEFAULT_ENCODING = "cl100k_base"


def get_model_specs(model: str) -> tuple[int, int, str]:
    """Get model specifications (context_window, max_output_tokens, encoding)."""
    model_lower = model.lower()
    
    if model_lower in MODEL_SPECS:
        return MODEL_SPECS[model_lower]
    
    for known_model in sorted(MODEL_SPECS.keys(), key=len, reverse=True):
        if model_lower.startswith(known_model):
            return MODEL_SPECS[known_model]
    
    if "gpt-5" in model_lower:
        return (400_000, 32_768, "cl100k_base")
    elif "gpt-4.1" in model_lower:
        return (1_047_576, 32_768, "cl100k_base")
    elif "gpt-4o" in model_lower:
        return (128_000, 16_384, "o200k_base")
    elif "gpt-4" in model_lower:
        return (128_000, 4_096, "cl100k_base")
    elif "gpt-3.5" in model_lower:
        return (16_385, 4_096, "cl100k_base")
    elif model_lower.startswith(("o1", "o3", "o4")):
        return (200_000, 100_000, "o200k_base")
    
    logging.warning(f"Unknown model '{model}', using defaults")
    return (DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_OUTPUT, DEFAULT_ENCODING)


# =============================================================================
# Course Materials
# =============================================================================

DEFAULT_COURSE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DSAA3071TheoryOfComputation")

# Command patterns
WEEK_PATTERN = re.compile(r'^/week\s*(\d+)\b', re.IGNORECASE)
RESET_PATTERN = re.compile(r'^/reset\b', re.IGNORECASE)


def load_course_materials(course_dir: str = None) -> dict:
    """Load all typst files from the course directory, organized by week."""
    if course_dir is None:
        course_dir = DEFAULT_COURSE_DIR
    
    materials = {}
    
    if not os.path.exists(course_dir):
        logging.warning(f"Course directory not found: {course_dir}")
        return materials
    
    typ_files = glob.glob(os.path.join(course_dir, "**/*.typ"), recursive=True)
    
    for filepath in typ_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            rel_path = os.path.relpath(filepath, course_dir)
            parts = rel_path.split(os.sep)
            category = parts[0] if parts[0].startswith('week') else 'other'
            
            if category not in materials:
                materials[category] = {}
            materials[category][rel_path] = content
                
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
    
    return materials


def extract_text_content(typst_content: str) -> str:
    """Extract readable text from typst content, removing heavy formatting."""
    skip_patterns = ['#import', '#let ', '#show', '#align', '#box(', '#rect(', '#table(', 
                     '#canvas(', '#diagram(', '#grid(', '#place(', 'gradient.', 'rgb(']
    
    lines = []
    for line in typst_content.split('\n'):
        stripped = line.strip()
        if not stripped or stripped.startswith('//'):
            continue
        if any(stripped.startswith(p) for p in skip_patterns):
            continue
        if stripped.startswith('=') or stripped.startswith('-') or stripped.startswith('*'):
            lines.append(stripped)
        elif stripped.startswith('[') and stripped.endswith(']'):
            lines.append(stripped[1:-1])
        elif not stripped.startswith('#') and not stripped.startswith(')'):
            lines.append(stripped)
    
    return '\n'.join(lines)


def match_file_pattern(filename: str, patterns: list) -> bool:
    """Check if filename matches any of the patterns."""
    for pattern in patterns:
        if any(c in pattern for c in ['*', '?', '[']):
            if fnmatch.fnmatch(filename, pattern):
                return True
        else:
            if pattern in filename:
                return True
    return False


def get_week_context(materials: dict, week_num: int, file_patterns: list = None, max_chars: int = 300_000) -> str:
    """Get course context for a specific week."""
    week_key = f"week{week_num}"
    if week_key not in materials:
        return ""
    
    context_parts = [f"=== WEEK {week_num} COURSE MATERIALS ===\n"]
    
    for filename, content in materials[week_key].items():
        if file_patterns and not match_file_pattern(filename, file_patterns):
            continue
        
        clean_content = extract_text_content(content)
        if clean_content.strip():
            context_parts.append(f"--- {filename} ---\n{clean_content}")
    
    full_context = "\n".join(context_parts)
    return full_context[:max_chars]


# =============================================================================
# ChatBot Class
# =============================================================================

class ChatBot:
    """
    OpenAI-powered chatbot using the Responses API.
    
    Modes:
    - Stream messages: RAG mode (vector store search, no chaining)
    - DM messages: Weekly mode (user specifies week, with chaining)
    """
    
    def __init__(self, model: str, api_key: str, course_dir: str = None, 
                 file_patterns: list = None, vector_store_id: str = None,
                 max_output_tokens: int = None):
        """
        Initialize the chatbot.
        
        Args:
            model: OpenAI model name
            api_key: OpenAI API key
            course_dir: Path to course materials directory
            file_patterns: Patterns to filter course files
            vector_store_id: OpenAI Vector Store ID for RAG
            max_output_tokens: Override for max output tokens
        """
        self.model = model
        self.client = OpenAIClient(api_key=api_key)
        
        _, default_max_output, encoding_name = get_model_specs(model)
        self.max_output_tokens = max_output_tokens or default_max_output
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # User state for DM mode
        self.user_response_ids = {}  # user_id -> last response ID
        self.user_weeks = {}         # user_id -> week number
        
        # RAG settings
        self.vector_store_id = vector_store_id
        
        # Course materials
        self.file_patterns = file_patterns or []
        self.course_materials = load_course_materials(course_dir)
        
        # Available weeks
        self.available_weeks = sorted([
            int(k.replace('week', '')) 
            for k in self.course_materials.keys() 
            if k.startswith('week')
        ])
        
        logging.info(f"ChatBot initialized: model={model}")
        logging.info(f"Available weeks: {self.available_weeks}")
        if vector_store_id:
            logging.info(f"RAG enabled: {vector_store_id}")
    
    def _get_base_instructions(self) -> str:
        """Base instructions for AI assistant."""
        return """You are an expert teaching assistant for DSAA3071 Theory of Computation at HKUST(GZ).

Your role is to:
- Help students understand concepts in automata theory, formal languages, and computability
- Explain DFA, NFA, regular expressions, context-free grammars, pushdown automata, and Turing machines
- Guide students through proofs and problem-solving techniques
- Be concise but thorough in explanations

=== CRITICAL FORMATTING RULE FOR ZULIP ===
You MUST format ALL math with spaces around $$ delimiters. This is NON-NEGOTIABLE.
Do NOT use backticks around math. Only use $$ with spaces.

CORRECT format (note the spaces before and after $$):
- "The formula is $$x^2$$ here" - spaces around $$
- "where $$\\delta: Q \\times \\Sigma \\to Q$$ defines" - spaces around $$
- "The set $$Q$$ is finite." - spaces around $$

WRONG format (will break rendering):
- "The formula is$$x^2$$here" - no spaces
- "$$Q$$:" - no space after

Pattern: always write " $$content$$ " with a space before the opening $$ and after the closing $$.
=== END CRITICAL FORMATTING RULE ==="""
    
    def _quote_message(self, text: str) -> str:
        """Quote a message using Zulip markdown.
        
        Uses quote block syntax with enough backticks to handle nested code blocks.
        """
        # Find the longest sequence of backticks in the text
        max_backticks = 0
        current_count = 0
        for char in text:
            if char == '`':
                current_count += 1
                max_backticks = max(max_backticks, current_count)
            else:
                current_count = 0
        
        # Use one more backtick than the max found (minimum 4 for quote blocks)
        fence = '`' * max(4, max_backticks + 1)
        return f"{fence}quote\n{text}\n{fence}"
    
    def _format_response(self, question: str, reply: str, usage, sender_name: str = None, sender_id: int = None, message_url: str = None) -> str:
        """Format response with user mention, quoted question, and usage info."""
        quoted_question = self._quote_message(question)
        
        # Add user mention in Zulip quote format: @_**Name|ID** [said](url):
        if sender_name and sender_id and message_url:
            mention = f"@_**{sender_name}|{sender_id}** [said]({message_url}):\n"
        elif sender_name and sender_id:
            mention = f"@_**{sender_name}|{sender_id}**:\n"
        else:
            mention = ""
        
        return (
            f"{mention}{quoted_question}\n\n"
            f"{reply}\n"
            f"------\n"
            f"Tokens: {usage.input_tokens:,} (input) + {usage.output_tokens:,} (output) "
            f"= {usage.total_tokens:,}"
        )
    
    def get_stream_response(self, user_id: str, prompt: str, original_message: str = None, sender_name: str = None, sender_id: int = None, message_url: str = None) -> str:
        """
        Handle stream message: RAG mode, no chaining.
        
        Uses vector store to search relevant content per query.
        
        Args:
            user_id: User identifier (email)
            prompt: The actual question to process (quote blocks stripped)
            original_message: Full message for quoting in response (optional, defaults to prompt)
            sender_name: Sender's display name for @mention in response
            sender_id: Sender's numeric ID for Zulip mention format
            message_url: URL to the original message for [said](url) format
        """
        logging.info(f"Stream message from {user_id}")
        
        # Use original_message for quoting if provided, otherwise use prompt
        quote_text = original_message if original_message else prompt
        
        if not self.vector_store_id:
            return (
                self._quote_message(quote_text) + "\n\n"
                f"**Error:** RAG not configured for stream messages.\n\n"
                f"Please set `VECTOR_STORE_ID` in config.ini (run `make upload` first)."
            )
        
        try:
            instructions = self._get_base_instructions() + """

You have access to course materials through the file_search tool. 
Search for relevant content to answer the question accurately."""
            
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                instructions=instructions,
                max_output_tokens=self.max_output_tokens,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [self.vector_store_id]
                }],
                truncation="auto",
            )
            
            reply = response.output_text or "I couldn't generate a response."
            return self._format_response(quote_text, reply, response.usage, sender_name, sender_id, message_url)
            
        except Exception as e:
            logging.error(f"Stream response error: {e}")
            if sender_name and sender_id and message_url:
                mention = f"@_**{sender_name}|{sender_id}** [said]({message_url}):\n"
            elif sender_name and sender_id:
                mention = f"@_**{sender_name}|{sender_id}**:\n"
            else:
                mention = ""
            return mention + self._quote_message(quote_text) + f"\n\nError: {str(e)}"
    
    def get_dm_response(self, user_id: str, prompt: str) -> str:
        """
        Handle DM message: Weekly mode with chaining.
        
        User must specify which week to study. Follow-ups are supported.
        """
        logging.info(f"DM message from {user_id}")
        
        prompt_stripped = prompt.strip()
        
        # Check for /reset command - clears conversation history only, keeps week
        if RESET_PATTERN.match(prompt_stripped):
            self.user_response_ids.pop(user_id, None)  # Clear conversation only
            week_num = self.user_weeks.get(user_id)
            if week_num:
                return (
                    f"Conversation cleared. Still studying **Week {week_num}**.\n\n"
                    f"Commands: `/week N` to switch, `/reset` to clear conversation"
                )
            else:
                weeks_list = ", ".join(map(str, self.available_weeks))
                return (
                    f"Conversation cleared.\n\n"
                    f"Use `/week N` to start studying.\n\n"
                    f"**Available weeks:** {weeks_list}"
                )
        
        # Check if user is specifying a week with /week command
        week_match = WEEK_PATTERN.match(prompt_stripped)
        if week_match:
            week_num = int(week_match.group(1))
            if week_num in self.available_weeks:
                self.user_weeks[user_id] = week_num
                # Note: Does NOT clear conversation - /reset does that
                return (
                    f"Now studying **Week {week_num}**.\n\n"
                    f"Commands: `/week N` to switch, `/reset` to clear conversation"
                )
            else:
                return (
                    f"Week {week_num} not found.\n\n"
                    f"**Available weeks:** {', '.join(map(str, self.available_weeks))}"
                )
        
        # Check if user has a week set
        if user_id not in self.user_weeks:
            if not self.available_weeks:
                return (
                    f"> {prompt}\n\n"
                    f"No course materials found. Please check COURSE_DIR configuration."
                )
            weeks_list = ", ".join(map(str, self.available_weeks))
            return (
                f"> {prompt}\n\n"
                f"Please specify which week you'd like to study.\n\n"
                f"**Command:** `/week N` (e.g., `/week 3`)\n\n"
                f"**Available weeks:** {weeks_list}"
            )
        
        week_num = self.user_weeks[user_id]
        
        try:
            # Load week context
            week_context = get_week_context(
                self.course_materials,
                week_num,
                self.file_patterns
            )
            
            if not week_context.strip():
                week_context = "(No materials found for this week matching the file patterns.)"
            
            instructions = self._get_base_instructions() + f"""

Currently studying: **Week {week_num}**

The student may ask follow-up questions. Use the course materials below to answer accurately.

{week_context}"""
            
            params = {
                "model": self.model,
                "input": prompt,
                "instructions": instructions,
                "max_output_tokens": self.max_output_tokens,
                "store": True,
                "truncation": "auto",
            }
            
            # Chain with previous response if available
            previous_id = self.user_response_ids.get(user_id)
            if previous_id:
                params["previous_response_id"] = previous_id
            
            response = self.client.responses.create(**params)
            
            # Store for conversation continuity
            self.user_response_ids[user_id] = response.id
            
            reply = response.output_text or "I couldn't generate a response."
            return self._format_response(prompt, reply, response.usage)
            
        except Exception as e:
            logging.error(f"DM response error: {e}")
            return f"> {prompt}\n\nError: {str(e)}"
    
    def clear_user_session(self, user_id: str):
        """Clear a user's session (week selection and conversation history)."""
        self.user_weeks.pop(user_id, None)
        self.user_response_ids.pop(user_id, None)


# Alias for backward compatibility
OpenAI = ChatBot
