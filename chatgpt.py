# chat_gpt.py
import openai

OPENAI_API_KEY = "your_openai_api_key_here"
openai.api_key = OPENAI_API_KEY

def get_chatgpt_response(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=3072,
            n=1,
            stop=None,
            temperature=0.5,
        )

        if response.choices:
            return response.choices[0].text.strip()
        else:
            return "Sorry, I couldn't generate a response."

    except Exception as e:
        if "Incorrect API key provided" in e:
            print('ERROR: Please set your OpenAI API key correctly.')
        else:
            print(f"Error: {e}")
            return "Sorry, there was an error generating a response."
