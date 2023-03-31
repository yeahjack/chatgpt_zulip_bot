# chatgpt.py
import openai
from configparser import ConfigParser
import tiktoken

config = ConfigParser()
config.read("config.ini")
OPENAI_API_KEY = config["settings"]["OPENAI_API_KEY"]
OPENAI_API_VERSION = config["settings"]["API_VERSION"]
openai.api_key = OPENAI_API_KEY

user_conversations = {}  # Maintain a dictionary to store conversation history per user
MAX_CONTENT_LENGTH = 4097 - 300


def trim_conversation_history(history, max_tokens):
    # Determine the appropriate encoding based on the API version.
    # Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    if "gpt-3.5-turbo" in OPENAI_API_VERSION:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    elif "gpt-4" in OPENAI_API_VERSION:
        encoding = tiktoken.encoding_for_model("gpt-4")
    elif 'text-embedding-ada' in OPENAI_API_VERSION:
        encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
    elif "text-davinci-002" in OPENAI_API_KEY:
        encoding = tiktoken.encoding_for_model("text-davinci-002")
    elif 'text-davinci-003' in OPENAI_API_KEY:
        encoding = tiktoken.encoding_for_model("text-davinci-003")
    else:
        return "OpenAI API Version Wrong!"

    tokens = 0
    trimmed_history = []
    for message in reversed(history):
        # Get the number of tokens for the current message
        message_tokens = len(encoding.encode(message))
        if tokens + message_tokens <= max_tokens:
            trimmed_history.insert(0, message)
            tokens += message_tokens
        else:
            break
    if tokens > 0:
        return trimmed_history
    else:
        return "Cannot Trim Conversation History! Please start a new conversation."


def prompt_manager(message):
    # Academic prompts which might be helpful.
    # Credits to https://github.com/binary-husky/chatgpt_academic/blob/b1e33b0f7aa9e69061d813262eb36ac297d49d0d/functional.py
    if message.startswith('/polish_en '):
        return "Below is a paragraph from an academic paper. Polish the writing to meet the academic style, \
improve the spelling, grammar, clarity, concision and overall readability. When neccessary, rewrite the whole sentence. \
Furthermore, list all modification and explain the reasons to do so.\n\n" + message

    elif message.startswith('/polish_zh '):
        return "作为一名中文学术论文写作改进助理，你的任务是改进所提供文本的拼写、语法、清晰、简洁和整体可读性，同时分解长句，减少重复，并提供改进建议。请只提供文本的更正版本，避免包括解释。请编辑以下文本：\n\n" + message

    elif message.startswith('/find_grammar_mistakes '):
        return "Below is a paragraph from an academic paper. Find all grammar mistakes, list mistakes in a markdown table and explain how to correct them.\n\n" + message

    elif message.startswith('/zh-en '):
        return "As an English-Chinese translator, your task is to accurately translate text between the two languages. \
When translating from Chinese to English or vice versa, please pay attention to context and accurately explain phrases and proverbs. \
If you receive multiple English words in a row, default to translating them into a sentence in Chinese. \
Below is the text you need to translate: \n\n" + message

    elif message.startswith('/en_ac '):
        return "Please translate following sentence to English with academic writing, and provide some related authoritative examples: \n\n" + message

    elif message.startswith('/ex_code_zh ') or message.startswith('/ex_code_zh\n'):
        return "请解释以下代码：\n```\n" + message

    else:
        return message


def get_chatgpt_response(user_id, prompt):
    global user_conversations

    if user_id not in user_conversations:
        user_conversations[
            user_id
        ] = []  # Create a new conversation history for a new user

    # Check if user input is "停止会话" or "end the conversation"
    if prompt == "停止会话" or prompt.lower() == "end the conversation" or prompt.lower() == "/end":
        # Clear the conversation history for the user
        user_conversations[user_id] = []
        return "The conversation has been ended and the context has been cleared."
    elif prompt == "/help":
        return """# Usage

To use the bot, mention it in any Zulip stream or send it a private message. You can mention the bot by typing `@bot_name` where the name is what you gave to the bot when you created it.
Except normal texts, the bot also accepts the following commands

## Commands
* `/help`: print this usage information.
* `/end`: end the current / start a new conversation. Bot will answer questions based on the context of the conversation. If see a rate limit exceed error after approximately 3500 token limit is reached in a single conversation, then you must restart the conversation with `/end`.
* `/polish_en`: polish the writing to meet the academic style, improve the spelling, grammar, clarity, concision and overall readability, and list all modification and explainations.
* `/polish_zh`: 使用中文改进所提供文本的拼写、语法、清晰、简洁和整体可读性，同时分解长句，减少重复。
* `/find_grammar_mistakes`: find all grammar mistakes, list mistakes in a table and explain how to correct them.
* `/zh-en`: translate text between the Chinese and English. 中英互译。
* `/en_ac`: translate sentence to English with academic writing, and provide related authoritative examples. 翻译至学术英语，并提供相关权威样例。
* `/ex_code_zh`: 用中文解释代码。

"""
    else:
        prompt = prompt_manager(prompt)
        # If use academic prompts, then context will not be recorded.
        if not prompt.startswith("/"):
            conversation_history = user_conversations[user_id]
            conversation_history.append(
                f"User: {prompt}"
            )  # Add user input to conversation history

    while True:
        messages = [
            {
                "role": "system",
                "content": "You are an AI language model trained to assist with a variety of tasks.",
            }
        ]  # System message for context

        for message in conversation_history:
            role, content = message.split(": ", 1)
            messages.append({"role": role.lower(), "content": content})

        try:
            response = openai.ChatCompletion.create(
                model=OPENAI_API_VERSION,
                messages=messages,
                max_tokens=1200,
                temperature=0.5,
            )

            if response.choices:
                role = response["choices"][0]["message"]["role"]
                reply = (
                    response["choices"][0]["message"]["content"].strip().replace(
                        "", "")
                )
                conversation_history.append(
                    f"{role}: {reply}"
                )  # Add AI response to conversation history
                return (
                    reply
                    + "\n------\nPrompt tokens used: "
                    + str(response.usage.prompt_tokens)
                    + "\nAnswer tokens used: "
                    + str(response.usage.completion_tokens)
                    + "\nTotal tokens used: "
                    + str(response.usage.total_tokens)
                )
            else:
                return "Sorry, I couldn't generate a response."

        except openai.error.RateLimitError:
            print("ERROR: OpenAI API rate limit exceeded. Please retry.")
            return "ERROR: OpenAI API rate limit exceeded. Please retry."

        except openai.error.OpenAIError as e:
            if "Please reduce the length" in str(e):
                conversation_history = trim_conversation_history(
                    conversation_history, MAX_CONTENT_LENGTH
                )
            else:
                print(f"Error: {e}")
                return "Sorry, there was an error generating a response."

        except Exception as e:
            print(f"Error: {e}")
            return "Sorry, there was an error generating a response."
