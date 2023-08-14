# chatgpt.py
import openai
import tiktoken
import logging

prompt_table = {
    "/polish_en": "Below is a paragraph from an academic paper. Polish the writing to meet the academic style, \
improve the spelling, grammar, clarity, concision and overall readability. When neccessary, rewrite the whole sentence. \
Furthermore, list all modification and explain the reasons to do so.",
    "/polish_zh": "作为一名中文学术论文写作改进助理，你的任务是改进所提供文本的拼写、语法、清晰、简洁和整体可读性，同时分解长句，减少重复，并提供改进建议。请只提供文本的更正版本，避免包括解释。请编辑以下文本：",
    "/find_grammar_mistakes": "Below is a paragraph from an academic paper. Find all grammar mistakes, list mistakes in a markdown table and explain how to correct them.",
    "/zh-en": "As a translator, your task is to accurately translate text from Chinese to English. \
Please pay attention to context and accurately explain phrases and proverbs. \
Below is the text you need to translate: ",
    "/en-zh": "As a translator, your task is to accurately translate text from English to Chinese. \
Please pay attention to context and accurately explain phrases and proverbs. \
Below is the text you need to translate: ",
    "/en_ac": "Please translate following sentence to English with academic writing, and provide some related authoritative examples: ",
    "/ex_code_zh": "请解释以下代码："
}

help_str = """# Usage

To use the bot, mention it in any Zulip stream or send it a private message. You can mention the bot by typing `@bot_name` where the name is what you gave to the bot when you created it.
Except normal texts, the bot also accepts the following commands

## Commands
* `/help`: print this usage information.
* `/end`: end the current and start a new conversation. Bot will answer questions based on the context of the conversation. If see a rate limit exceed error after approximately 3500 token limit is reached in a single conversation, then you must restart the conversation with `/end`.
""" + "\n".join(["* `%s`: (Prompt) %s" % (k, v) for k, v in prompt_table.items()])


class OpenAI(object):
    def __init__(self, api_version, api_key, max_content_length):
        # setup API version and API keys
        self.api_version = api_version
        self.api_key = api_key
        openai.api_key = api_key
        # Maintain a dictionary to store conversation history per user
        self.user_conversations = {}
        # maximum number of tokens in a single conversation
        self.max_content_length = max_content_length

    def trim_conversation_history(self, history, max_tokens):
        # Determine the appropriate encoding based on the API version.
        # Ref: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        if "gpt-3.5-turbo" in self.api_version:
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif "gpt-4" in self.api_version:
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif 'text-embedding-ada' in self.api_version:
            encoding = tiktoken.encoding_for_model('text-embedding-ada-002')
        elif "text-davinci-002" in self.api_key:
            encoding = tiktoken.encoding_for_model("text-davinci-002")
        elif 'text-davinci-003' in self.api_key:
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

    def get_chatgpt_response(self, user_id, prompt):
        global help_str
        if user_id not in self.user_conversations:
            self.user_conversations[
                user_id
            ] = []  # Create a new conversation history for a new user

        # Check if user input is "停止会话" or "end the conversation"
        if prompt == "停止会话" or prompt.lower() == "end the conversation" or prompt.lower() == "/end":
            # Clear the conversation history for the user
            self.user_conversations[user_id] = []
            return "The conversation has been ended and the context has been cleared."
        elif prompt == "/help":
            return help_str
        else:
            prompt, ret_code = prompt_manager(prompt)
            conversation_history = self.user_conversations[user_id]
            if ret_code == 1: 
                # Contexual conversation
                conversation_history.append(
                    f"User: {prompt}"
                )  # Add user input to conversation history
            elif ret_code == 2:  # command not found
                return prompt
        logging.info(f"API call from user: {user_id}")
        while True:
            messages = [
                {
                    "role": "system",
                    "content": "You are an AI language model trained to assist with a variety of tasks.",
                }
            ]  # System message for context
            if ret_code == 0:
                messages.append({'role': 'user', 'content': prompt})
            else:
                for message in conversation_history:
                    role, content = message.split(": ", 1)
                    messages.append({"role": role.lower(), "content": content})
            try:
                response = openai.ChatCompletion.create(
                    model=self.api_version,
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
                        + "\n------\nTokens used: "
                        + str(response.usage.prompt_tokens) + " (prompt) + "
                        + str(response.usage.completion_tokens) + " (answer)"
                        + " = " + str(response.usage.total_tokens) + "/" + str(self.max_content_length)
                    )
                else:
                    err_msg = "Sorry, I couldn't generate a response."
                    logging.error(err_msg)
                    return err_msg

            except openai.error.RateLimitError:
                err_msg = "Sorry, You have exceeded the content length set by OpenAI. History has been cleared and please retry your request."
                self.user_conversations[user_id] = []
                logging.error(err_msg)
                return err_msg

            except Exception as e:
                logging.error(f"An error occurred on line {e.__traceback__.tb_lineno}: {e}")
                return f"An error occurred."

# take a message as input, returns a prompt message and a return code
# return code 0: command message: no context
# return code 1: normal contextual message
# return code 2: command not found
def prompt_manager(message):
    # Academic prompts which might be helpful.
    # Credits to https://github.com/binary-husky/chatgpt_academic/blob/b1e33b0f7aa9e69061d813262eb36ac297d49d0d/functional.py
    if message.startswith("/"):
        # get command name and message
        result = message.split(" ", 1)
        if len(result) > 1:  # has space
            name, msg = result
        else:
            name, msg = result[0], ""
        # get prompt and return code from name
        if name in prompt_table:
            prompt = prompt_table[name]
            logging.info(f"Command hit: {name}")
            return prompt + "\n\n" + msg, 0
        else:
            msg = f"Sorry, command not found: `{name}`, type `/help` to get the list of commands."
            logging.error(msg)
            return msg, 2
    else:
        return message, 1
