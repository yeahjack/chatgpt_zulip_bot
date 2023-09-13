import requests, json
import error
api_key = ""
class ChatCompletion(object):
    @staticmethod
    def create(model, messages, max_tokens, temperature):
        url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            #"max_tokens": max_tokens
        }
        return requests.post(url, headers=headers, data=json.dumps(data))

if __name__ == '__main__':
    from configparser import ConfigParser
    configfile = "config.ini"
    config = ConfigParser()
    config.read(configfile)
    api_key = config["settings"]["OPENAI_API_KEY"]

    messages = [{"role": "user", "content": "Please ask me here"}]
    result = ChatCompletion.create('gpt-4', messages, max_tokens=1200, temperature=0.5)
    print(result)
    import pdb
    pdb.set_trace()
