# chat_gpt.py
import openai
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')
OPENAI_API_KEY = config['settings']['OPENAI_API_KEY']
openai.api_key = OPENAI_API_KEY


def get_chatgpt_response(prompt):
    try:
        response = openai.Completion.create(
            model="text-davinci-003",  # 对话模型的名称
            prompt=prompt,
            temperature=0.5,  # 值在[0,1]之间，越大表示回复越具有不确定性
            max_tokens=1200,  # 回复最大的字符数
            top_p=1,
            frequency_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            presence_penalty=0.0,  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            stop=["\n\n\n"]
        )
        if response.choices:
            return response.choices[0]['text'].strip().replace('<|endoftext|>', '')
        else:
            return "Sorry, I couldn't generate a response."
        
    except openai.error.RateLimitError:
        print('ERROR: OpenAI API rate limit exceeded. Please rety.')
        return "ERROR: OpenAI API rate limit exceeded. Please rety."

    except Exception as e:
        print(f"Error: {e}")
        return "Sorry, there was an error generating a response."
