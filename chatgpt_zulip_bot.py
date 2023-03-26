# chat_gpt_zulip_bot.py

import zulip
import re
from chatgpt import get_chatgpt_response
import atexit
from configparser import ConfigParser

config = ConfigParser()
config.read('config.ini')

ZULIP_CONFIG = config['settings']['ZULIP_CONFIG']
USER_ID = int(config['settings']['USER_ID'])
BOT_ID = int(config['settings']['BOT_ID'])

class ChatGPTZulipBot(zulip.Client):
    def __init__(self, config_file):
        super().__init__(config_file=config_file)

    def set_status(self, status):
        request = {
            "status_text": "Offline" if status else "Online",
            "away": status,
            # "emoji_name": "car",
            # "emoji_code": "1f697",
            "reaction_type": "unicode_emoji",
        }
        self.call_endpoint(url="/users/me/status",
                           method="POST", request=request)

    def send_notification(self, message):
        self.send_message({
            "type": "private",
            "to": [USER_ID],
            "content": message,
        })

    def process_message(self, msg):
        sender_email = msg['sender_email']
        message_content = msg['content']
        message_type = msg['type']
        if msg['sender_id'] != BOT_ID:
            if message_content.startswith('@**ChatGPT**'):
                stream_id = msg.get('stream_id', None)
                topic = msg.get('subject', None)
                prompt = re.sub('@\*\*ChatGPT\*\*', '', message_content).strip()
                response = get_chatgpt_response(msg['sender_email'], prompt)
                self.send_message({
                    "type": "stream",
                    "to": stream_id,
                    "subject": topic,
                    "content": response,
                })

            if message_type == 'private':
                prompt = message_content
                response = get_chatgpt_response(msg['sender_email'], prompt)
                self.send_message({
                    "type": "private",
                    "to": sender_email,
                    "content": response,
                })


def on_exit(bot):
    bot.send_notification("NOTICE: The ChatGPT bot is now offline.")
    bot.set_status(True)


if __name__ == "__main__":
    bot = ChatGPTZulipBot(ZULIP_CONFIG)
    bot.send_notification("NOTICE: The ChatGPT bot is now online.")
    bot.set_status(False)
    print("Successfully started the ChatGPT bot.")

    atexit.register(on_exit, bot)

    bot.call_on_each_message(bot.process_message)
