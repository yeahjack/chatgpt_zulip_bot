# chat_gpt.py
import openai
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
OPENAI_API_KEY = config['settings']['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY

user_conversations = {}  # Maintain a dictionary to store conversation history per user


def get_chatgpt_response(user_id, prompt):
    global user_conversations

    if user_id not in user_conversations:
        user_conversations[user_id] = []  # Create a new conversation history for a new user

    # Check if user input is "停止会话" or "end the conversation"
    if prompt == "停止会话" or prompt.lower() == "end the conversation":
        user_conversations[user_id] = []  # Clear the conversation history for the user
        return "The conversation has been ended and the context has been cleared."

    conversation_history = user_conversations[user_id]
    conversation_history.append(f"User: {prompt}")  # Add user input to conversation history
    context = "\n".join(conversation_history)  # Concatenate the conversation history

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"{context}\nAI:",
            temperature=0.5,
            max_tokens=3500,
            top_p=1,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["\n\n\n"]
        )
        if response.choices:
            reply = response.choices[0]['text'].strip().replace('', '')
            conversation_history.append(f"AI: {reply}")  # Add AI response to conversation history
            return reply
        else:
            return "Sorry, I couldn't generate a response."

    except openai.error.RateLimitError:
        print('ERROR: OpenAI API rate limit exceeded. Please retry.')
        return "ERROR: OpenAI API rate limit exceeded. Please retry."

    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, there was an error generating a response."
