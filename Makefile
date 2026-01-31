# ChatGPT Zulip Bot - Server Management
# Usage: make [target]

.PHONY: start stop restart status logs logs-err tail test install sync clean help venv upload list-files

# Python command (use venv if available)
PYTHON = $(shell [ -f .venv/bin/python ] && echo ".venv/bin/python" || echo "python3")
UV = uv

# Default target
help:
	@echo "ChatGPT Zulip Bot - Server Management"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Server Control:"
	@echo "  start      - Start the bot"
	@echo "  stop       - Stop the bot"
	@echo "  restart    - Restart the bot"
	@echo "  status     - Check bot status"
	@echo ""
	@echo "Logs:"
	@echo "  logs       - Show recent stdout logs"
	@echo "  logs-err   - Show recent stderr logs"
	@echo "  tail       - Follow logs in real-time (Ctrl+C to exit)"
	@echo ""
	@echo "Development:"
	@echo "  test       - Run tests"
	@echo "  tokens     - Count course material tokens"
	@echo "  check      - Check config and dependencies"
	@echo ""
	@echo "Setup:"
	@echo "  venv       - Create virtual environment with uv"
	@echo "  sync       - Sync dependencies with uv"
	@echo "  install    - Install dependencies (legacy pip)"
	@echo "  clean      - Remove cache files"
	@echo ""
	@echo "RAG (Retrieval-Augmented Generation):"
	@echo "  upload     - Upload course files to OpenAI vector store"
	@echo "  list-files - List files in the vector store"

# Server Control
start:
	supervisorctl start chatgpt_zulip_bot_autorun

stop:
	supervisorctl stop chatgpt_zulip_bot_autorun

restart:
	supervisorctl restart chatgpt_zulip_bot_autorun
	@sleep 2
	@supervisorctl status chatgpt_zulip_bot_autorun

status:
	supervisorctl status chatgpt_zulip_bot_autorun

# Logs
logs:
	supervisorctl tail chatgpt_zulip_bot_autorun stdout

logs-err:
	supervisorctl tail chatgpt_zulip_bot_autorun stderr

tail:
	supervisorctl tail -f chatgpt_zulip_bot_autorun stdout

# Development
test:
	$(UV) run pytest test_all.py -v

tokens:
	@$(UV) run python -c "\
import tiktoken, configparser; \
from chatgpt import load_course_materials, get_course_context; \
cfg = configparser.ConfigParser(); cfg.read('config.ini'); \
s = cfg['settings']; \
patterns = [p.strip() for p in s.get('FILE_PATTERNS', '').split(',') if p.strip()]; \
m = load_course_materials(s.get('COURSE_DIR')); \
enc = tiktoken.get_encoding('cl100k_base'); \
ctx = get_course_context(m, file_patterns=patterns); \
print(f'Patterns: {patterns or \"all\"}'); \
print(f'Files: {sum(len(v) for v in m.values())}'); \
print(f'Context tokens: {len(enc.encode(ctx)):,}'); \
print(f'Context chars: {len(ctx):,}')"

check:
	@echo "=== Config ===" 
	@test -f config.ini && echo "config.ini: OK" || echo "config.ini: MISSING"
	@test -f zuliprc && echo "zuliprc: OK" || echo "zuliprc: MISSING"
	@test -d DSAA3071TheoryOfComputation && echo "Course dir: OK" || echo "Course dir: MISSING"
	@echo ""
	@echo "=== Environment ==="
	@test -d .venv && echo ".venv: OK" || echo ".venv: MISSING (run 'make venv')"
	@which uv > /dev/null && echo "uv: $$(uv --version)" || echo "uv: NOT INSTALLED"
	@echo ""
	@echo "=== Dependencies ==="
	@$(UV) run python -c "import openai; print(f'openai: {openai.__version__}')"
	@$(UV) run python -c "import tiktoken; print('tiktoken: OK')"
	@$(UV) run python -c "import zulip; print('zulip: OK')"
	@echo ""
	@echo "=== Model Config ==="
	@grep "^MODEL" config.ini 2>/dev/null || echo "MODEL not set"

# Setup with uv
venv:
	$(UV) venv
	$(UV) sync

sync:
	$(UV) sync

# Legacy pip install
install:
	pip install -r requirements.txt

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Cache cleaned"

# RAG: Upload course files to OpenAI and create vector store
upload:
	@$(UV) run python upload_to_openai.py

# RAG: List files in the vector store
list-files:
	@$(UV) run python -c "\
import configparser; \
from openai import OpenAI; \
cfg = configparser.ConfigParser(); cfg.read('config.ini'); \
s = cfg['settings']; \
client = OpenAI(api_key=s['OPENAI_API_KEY']); \
vs_id = s.get('VECTOR_STORE_ID'); \
if not vs_id: \
    print('VECTOR_STORE_ID not set in config.ini'); \
    exit(1); \
vs = client.vector_stores.retrieve(vs_id); \
print(f'Vector Store: {vs.name} ({vs.id})'); \
print(f'Status: {vs.status}, Files: {vs.file_counts.completed}'); \
files = client.vector_stores.files.list(vs_id); \
for f in files.data: \
    info = client.files.retrieve(f.id); \
    print(f'  - {info.filename}')"
