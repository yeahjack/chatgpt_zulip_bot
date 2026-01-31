# ChatGPT Zulip Bot

![GitHub License](https://img.shields.io/github/license/yeahjack/chatgpt_zulip_bot) [![Python Tests](https://github.com/yeahjack/chatgpt_zulip_bot/actions/workflows/ci.yml/badge.svg)](https://github.com/yeahjack/chatgpt_zulip_bot/actions/workflows/ci.yml)

An AI-powered Zulip bot for course assistance with different modes for streams and DMs.

## Features

- **Stream Mode**: RAG-powered Q&A (vector store search, stateless)
- **DM Mode**: Weekly study sessions (user selects week, with follow-up support)
- **Access Control**: Restrict bot to specific Zulip streams and their members
- **Question Quoting**: Responses include quoted original question for context

---

## How It Works

| Message Type | Mode | Behavior |
|--------------|------|----------|
| **Stream** | RAG | Searches vector store per query, no conversation memory |
| **DM** | Weekly | User specifies week, then can ask follow-up questions |

### Stream Messages

When mentioned in a stream (`@ChatGPT what is a DFA?`):
- Uses RAG to search relevant course materials
- Each question is independent (stateless)
- Best for quick Q&A visible to everyone

### Direct Messages

When messaged directly:
1. User must first specify a week with `/week N` command
2. Bot loads that week's course materials
3. User can ask follow-up questions (conversation is chained)
4. To switch weeks, use `/week N` again

---

## File Structure

```
chatgpt_zulip_bot/
├── chatgpt.py              # ChatBot class with stream/DM modes
├── chatgpt_zulip_bot.py    # Zulip bot server
├── upload_to_openai.py     # Upload course materials to vector store
├── config.ini              # Your configuration (git-ignored)
├── config.ini.example      # Configuration template
├── zuliprc                 # Zulip credentials (git-ignored)
└── DSAA3071.../            # Course materials (organized by week)
```

---

## Installation

### 1. Clone and Install

```bash
git clone https://github.com/yeahjack/chatgpt_zulip_bot.git
cd chatgpt_zulip_bot

# Using uv (recommended)
uv sync

# Or pip
pip install -r requirements.txt
```

### 2. Set Up Zulip Bot

1. Go to Zulip **Settings → Your bots**
2. Create a generic bot
3. Download `zuliprc` and place in this directory

### 3. Configure

```bash
cp config.ini.example config.ini
# Edit config.ini with your settings
```

### 4. Upload Course Materials (for RAG)

```bash
make upload
# Copy the VECTOR_STORE_ID to config.ini
```

### 5. Run

```bash
make run
```

---

## Configuration

```ini
[settings]
# OpenAI
OPENAI_API_KEY = sk-...
MODEL = gpt-4o

# Zulip
ZULIP_CONFIG = zuliprc
USER_ID = 123
BOT_ID = 456
BOT_NAME = ChatGPT
ALLOWED_STREAMS = DSAA3071-2026-Spring

# Course materials
COURSE_DIR = DSAA3071TheoryOfComputation

# RAG (for stream mode) - run 'make upload' first
VECTOR_STORE_ID = vs_xxxxxxxxxxxx

# File patterns (for DM weekly mode)
FILE_PATTERNS = *learning-sheet*, *validation*
```

| Setting | Required | Description |
|---------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `MODEL` | Yes | Model name (gpt-4o, gpt-5.2, etc.) |
| `ZULIP_CONFIG` | Yes | Path to zuliprc file |
| `USER_ID` | Yes | Admin's Zulip user ID |
| `BOT_ID` | Yes | Bot's Zulip user ID |
| `BOT_NAME` | Yes | Bot's display name |
| `ALLOWED_STREAMS` | No | Comma-separated stream names |
| `COURSE_DIR` | Yes | Path to course materials |
| `VECTOR_STORE_ID` | Yes* | OpenAI vector store ID (*required for streams) |
| `FILE_PATTERNS` | No | Patterns for DM weekly mode |

---

## Usage Examples

### In Streams

```
@ChatGPT What is a DFA?
@ChatGPT Explain the pumping lemma for regular languages
@ChatGPT How do I convert NFA to DFA?
```

### In DMs

```
User: /week 3
Bot:  Now studying Week 3. Ask any questions about this week's content!
      Commands: /week N to switch weeks

User: What topics are covered?
Bot:  > What topics are covered?
      Week 3 covers... [answer based on week 3 materials]

User: Can you explain more about NFAs?
Bot:  > Can you explain more about NFAs?
      [follow-up answer, still using week 3 context]

User: /week 5
Bot:  Now studying Week 5. Ask any questions about this week's content!
```

### Commands

| Command | Context | Description |
|---------|---------|-------------|
| `/week N` | DM | Switch to week N (keeps conversation history) |
| `/reset` | DM | Clear conversation history (keeps current week) |
| `/refresh` | Any | Admin only: reload subscriber list |

---

## Response Format

All responses include the quoted question:

```
> What is a DFA?

A DFA (Deterministic Finite Automaton) is...

------
Tokens: 1,234 (input) + 567 (output) = 1,801
```

---

## Contributing

Key files:
- `chatgpt.py` - Core AI logic, stream/DM handlers
- `chatgpt_zulip_bot.py` - Zulip integration
- `upload_to_openai.py` - Vector store management

## License

MIT License
