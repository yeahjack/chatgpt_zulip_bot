# chatgpt.py
"""
OpenAI-powered course assistant for DSAA3071 Theory of Computation.
Uses the modern Responses API (recommended by OpenAI for new projects).

Context management: Uses truncation="auto" which lets OpenAI automatically
drop items from the beginning of the conversation if context exceeds limits.
"""

from openai import OpenAI as OpenAIClient
import tiktoken
import logging
import os
import glob

# =============================================================================
# Model Configuration
# =============================================================================

# Model specifications: (context_window, max_output_tokens, encoding)
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
    
    # Exact match
    if model_lower in MODEL_SPECS:
        return MODEL_SPECS[model_lower]
    
    # Prefix matching for versioned models
    for known_model in sorted(MODEL_SPECS.keys(), key=len, reverse=True):
        if model_lower.startswith(known_model):
            return MODEL_SPECS[known_model]
    
    # Fallback based on model family
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


def get_encoding(model: str):
    """Get tiktoken encoding for the model."""
    _, _, encoding_name = get_model_specs(model)
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception:
        return tiktoken.get_encoding(DEFAULT_ENCODING)


# =============================================================================
# Course Materials
# =============================================================================

DEFAULT_COURSE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DSAA3071TheoryOfComputation")

HELP_MESSAGE = """# DSAA3071 Theory of Computation - Course Assistant

This bot is a specialized assistant for the DSAA3071 Theory of Computation course at HKUST(GZ).
It has access to all course materials including lecture notes, learning sheets, and validation exercises.

## Commands
* `/help`: Print this usage information.
* `/end`: End the current conversation and clear context.
* `/week N`: Focus on a specific week's content (e.g., `/week 3`).

## Topics Covered
Finite Automata (DFA/NFA), Regular Languages, Context-Free Grammars, 
Pushdown Automata, Turing Machines, Decidability, and Complexity Theory.

Feel free to ask questions about any topic from the course!
"""


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
            category = parts[0] if parts[0].startswith('week') else (parts[0] if len(parts) > 1 else 'root')
            
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


def get_course_context(materials: dict, week: str = None, max_chars: int = 500_000) -> str:
    """Generate a context string from course materials (learning-sheet.typ files only)."""
    context_parts = []
    
    # Determine which weeks to include
    if week and week in materials:
        weeks_to_load = [week]
    else:
        weeks_to_load = sorted([k for k in materials.keys() if k.startswith('week')])
    
    for wk in weeks_to_load:
        week_content = []
        for filename, content in materials[wk].items():
            # Only load learning-sheet files
            if 'learning-sheet' in filename:
                clean_content = extract_text_content(content)
                if clean_content.strip():
                    week_content.append(f"--- {filename} ---\n{clean_content}")
        
        if week_content:
            context_parts.append(f"\n=== {wk.upper()} ===")
            context_parts.extend(week_content)
    
    full_context = "\n".join(context_parts)
    return full_context[:max_chars]


# =============================================================================
# Main ChatBot Class (using Responses API)
# =============================================================================

class ChatBot:
    """
    OpenAI-powered chatbot using the modern Responses API.
    
    The Responses API is recommended by OpenAI for new projects (as of 2025).
    Benefits: better performance, lower costs, cleaner API, stateful conversations.
    """
    
    def __init__(self, model: str, api_key: str, course_dir: str = None, max_output_tokens: int = None):
        """
        Initialize the chatbot.
        
        Args:
            model: OpenAI model name (e.g., "gpt-4o", "gpt-5.2")
            api_key: OpenAI API key
            course_dir: Path to course materials directory (optional)
            max_output_tokens: Override for max output tokens (auto-detected if None)
        """
        self.model = model
        self.client = OpenAIClient(api_key=api_key)
        
        # Get model specifications (encoding and default max output)
        _, default_max_output, encoding_name = get_model_specs(model)
        self.max_output_tokens = max_output_tokens or default_max_output
        self.encoding = tiktoken.get_encoding(encoding_name)
        
        # User state: store response IDs for conversation continuity
        self.user_response_ids = {}  # user_id -> last response ID
        self.week_focus = {}         # user_id -> week string or None
        
        # Load course materials
        self.course_materials = load_course_materials(course_dir)
        
        logging.info(f"ChatBot initialized: model={model}, max_output={self.max_output_tokens:,}")
        logging.info(f"Loaded course materials: {sorted(self.course_materials.keys())}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def _handle_command(self, user_id: str, command: str) -> str | None:
        """Handle bot commands. Returns response string or None if not a command."""
        cmd = command.lower().strip()
        
        # End conversation
        if cmd in ["停止会话", "end the conversation", "/end"]:
            self.user_response_ids.pop(user_id, None)
            self.week_focus.pop(user_id, None)
            return "The conversation has been ended and the context has been cleared."
        
        # Help
        if cmd == "/help":
            return HELP_MESSAGE
        
        # Week focus
        if cmd.startswith("/week"):
            parts = cmd.split()
            if len(parts) >= 2:
                week_key = f"week{parts[1]}"
                if week_key in self.course_materials:
                    self.week_focus[user_id] = week_key
                    return f"Now focusing on **{week_key}** materials. Ask me anything about this week's content!"
                else:
                    available = sorted([k for k in self.course_materials.keys() if k.startswith('week')])
                    return f"Week {parts[1]} not found. Available: {', '.join(available)}"
            else:
                self.week_focus.pop(user_id, None)
                return "Week focus cleared. Now using all course materials."
        
        return None
    
    def _get_instructions(self, user_id: str) -> str:
        """Build system instructions with course context.
        
        Note: We don't manually truncate - OpenAI's truncation="auto" handles
        context overflow by dropping items from the beginning of the conversation.
        """
        week = self.week_focus.get(user_id)
        
        base_instructions = f"""You are an expert teaching assistant for DSAA3071 Theory of Computation at HKUST(GZ).
You have access to all course materials including lecture notes, learning sheets, and exercises.

IMPORTANT: You are responding in Zulip chat. For math equations:
- Use $$ ... $$ for LaTeX math (both inline and block)
- MUST have spaces around the delimiters: ` $$x^2$$ ` not `$$x^2$$`
- Example: The transition function is ` $$\\delta: Q \\times \\Sigma \\to Q$$ `

Your role is to:
- Help students understand concepts in automata theory, formal languages, and computability
- Explain DFA, NFA, regular expressions, context-free grammars, pushdown automata, and Turing machines
- Guide students through proofs and problem-solving techniques
- Reference specific course materials when relevant

{f"Currently focusing on: {week}" if week else "Covering all course materials."}

=== COURSE MATERIALS ===
"""
        
        # Load generous amount of course context - OpenAI handles overflow via truncation="auto"
        # Use a reasonable limit to avoid excessive API costs
        max_chars = 100_000  # ~25k tokens
        
        course_context = get_course_context(self.course_materials, week=week, max_chars=max_chars)
        
        return base_instructions + course_context
    
    def get_response(self, user_id: str, prompt: str) -> str:
        """
        Get a response for the user's prompt using the Responses API.
        
        Args:
            user_id: Unique identifier for the user
            prompt: User's message
            
        Returns:
            Response string
        """
        # Initialize user state
        if user_id not in self.week_focus:
            self.week_focus[user_id] = None
        
        # Check for commands
        command_response = self._handle_command(user_id, prompt)
        if command_response:
            return command_response
        
        logging.info(f"API call from user: {user_id}")
        
        try:
            # Build request parameters
            params = {
                "model": self.model,
                "input": prompt,
                "instructions": self._get_instructions(user_id),
                "max_output_tokens": self.max_output_tokens,
                "store": True,  # Enable stateful conversations
                "truncation": "auto",  # Let OpenAI handle context overflow
            }
            
            # Continue from previous response if available
            previous_id = self.user_response_ids.get(user_id)
            if previous_id:
                params["previous_response_id"] = previous_id
            
            # Call the Responses API
            response = self.client.responses.create(**params)
            
            # Store response ID for conversation continuity
            self.user_response_ids[user_id] = response.id
            
            # Extract text using the convenient output_text helper
            reply = response.output_text
            
            # Format response with usage info
            usage = response.usage
            return (
                f"{reply}\n"
                f"------\n"
                f"Tokens: {usage.input_tokens:,} (input) + {usage.output_tokens:,} (output) "
                f"= {usage.total_tokens:,}"
            )
            
        except Exception as e:
            # If Responses API fails, fall back to Chat Completions
            logging.warning(f"Responses API failed, falling back to Chat Completions: {e}")
            return self._fallback_chat_completions(user_id, prompt)
    
    def _fallback_chat_completions(self, user_id: str, prompt: str) -> str:
        """Fallback to Chat Completions API if Responses API is unavailable."""
        try:
            system_content = self._get_instructions(user_id)
            
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt}
            ]
            
            # Log token usage for debugging
            prompt_tokens = self.count_tokens(prompt)
            total_input = self.count_tokens(system_content) + prompt_tokens
            logging.info(f"Input tokens: {total_input:,} (system: {self.count_tokens(system_content):,}, user: {prompt_tokens:,})")
            
            # GPT-5.x and newer models use max_completion_tokens instead of max_tokens
            model_lower = self.model.lower()
            if any(x in model_lower for x in ["gpt-5", "gpt-4.1", "o1", "o3", "o4"]):
                token_param = {"max_completion_tokens": self.max_output_tokens}
            else:
                token_param = {"max_tokens": self.max_output_tokens}
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                **token_param,
            )
            
            if not response.choices:
                return "Sorry, I couldn't generate a response."
            
            reply = response.choices[0].message.content.strip()
            usage = response.usage
            
            return (
                f"{reply}\n"
                f"------\n"
                f"Tokens: {usage.prompt_tokens:,} (prompt) + {usage.completion_tokens:,} (response) "
                f"= {usage.total_tokens:,}"
            )
            
        except Exception as e:
            logging.error(f"Error: {e}")
            return "An error occurred while processing your request."


# Alias for backward compatibility
OpenAI = ChatBot
