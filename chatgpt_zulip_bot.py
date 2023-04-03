# chat_gpt_zulip_bot.py

import zulip
import re
from chatgpt import OpenAI
import atexit
from configparser import ConfigParser
import logging


class ChatGPTZulipBot(zulip.Client):
    def __init__(self, config_file, user_id, bot_id, ai):
        super().__init__(config_file=config_file)
        self.user_id = user_id
        self.bot_id = bot_id
        # the OpenAI instance
        self.ai = ai

    def send_notification(self, message):
        self.send_message(
            {
                "type": "private",
                "to": [self.user_id],
                "content": message,
            }
        )

    def process_message(self, msg):
        sender_email = msg["sender_email"]
        message_content = msg["content"]
        message_type = msg["type"]
        if msg["sender_id"] != self.bot_id:
            if message_content.startswith("@**ChatGPT**"):
                stream_id = msg.get("stream_id", None)
                topic = msg.get("subject", None)
                prompt = re.sub("@\*\*ChatGPT\*\*", "",
                                message_content).strip()
                response = self.ai.get_chatgpt_response(
                    msg["sender_email"], prompt)
                self.send_message(
                    {
                        "type": "stream",
                        "to": stream_id,
                        "subject": topic,
                        "content": response,
                    }
                )

            if message_type == "private":
                prompt = message_content
                response = self.ai.get_chatgpt_response(
                    msg["sender_email"], prompt)
                self.send_message(
                    {
                        "type": "private",
                        "to": sender_email,
                        "content": response,
                    }
                )


def on_exit(bot):
    bot.send_notification("NOTICE: The ChatGPT bot is now offline.")


def serve(configfile):
    config = ConfigParser()
    config.read(configfile)

    OPENAI_API_KEY = config["settings"]["OPENAI_API_KEY"]
    OPENAI_API_VERSION = config["settings"]["API_VERSION"]

    ZULIP_CONFIG = config["settings"]["ZULIP_CONFIG"]
    USER_ID = int(config["settings"]["USER_ID"])
    BOT_ID = int(config["settings"]["BOT_ID"])

    ai = OpenAI(OPENAI_API_VERSION, OPENAI_API_KEY)
    bot = ChatGPTZulipBot(ZULIP_CONFIG, USER_ID, BOT_ID, ai)
    bot.send_notification("NOTICE: The ChatGPT bot is now online.")
    print("Successfully started the ChatGPT bot.")

    atexit.register(on_exit, bot)

    bot.call_on_each_message(bot.process_message)


if __name__ == "__main__":
    logging.basicConfig(filename="bot.log")
    serve("config.ini")
