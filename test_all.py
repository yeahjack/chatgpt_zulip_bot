from chatgpt import prompt_manager, help_str, prompt_table, OpenAI
from chatgpt_zulip_bot import ChatGPTZulipBot
from configparser import ConfigParser
import os

def test_prompt_manager():
    for command, prompt in prompt_table.items():
        result = prompt_manager("%s This is a test!"%command)
        assert isinstance(result, str)
    
    result = prompt_manager("/en-zh This is a test!")
    assert result == "As a translator, your task is to accurately translate text from English to Chinese. \
Please pay attention to context and accurately explain phrases and proverbs. \
Below is the text you need to translate: \n\nThis is a test!"


def test_bot():
    configfile = os.path.join("config-sample", "config.ini")
    config = ConfigParser()
    config.read(configfile)

    OPENAI_API_KEY = config["settings"]["OPENAI_API_KEY"]
    OPENAI_API_VERSION = config["settings"]["API_VERSION"]

    ai = OpenAI(OPENAI_API_VERSION, OPENAI_API_KEY)
    assert isinstance(ai, OpenAI)
    helpstr = ai.get_chatgpt_response(3, "/help")
    global help_str
    assert helpstr == help_str
    assert isinstance(helpstr, str)

    #ZULIP_CONFIG = config["settings"]["ZULIP_CONFIG"]
    #USER_ID = int(config["settings"]["USER_ID"])
    #BOT_ID = int(config["settings"]["BOT_ID"])
    #bot = ChatGPTZulipBot(ZULIP_CONFIG, USER_ID, BOT_ID, ai)
    #assert isinstance(bot, ChatGPTZulipBot)
