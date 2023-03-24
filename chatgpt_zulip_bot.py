# chat_gpt_zulip_bot.py
import zulip
import re
from chatgpt import get_chatgpt_response
import atexit

ZULIP_CONFIG = "zuliprc"
USER_ID = 47

class ChatGPTZulipBot(zulip.Client):
    def __init__(self, config_file):
        super().__init__(config_file=config_file)

    def set_status(self, status):
        presence = 'idle' if status else 'active'
        self.update_presence({
            "status_text": "",
            "away": presence,
        })
        
    def send_notification(self, message, recipient_email):
        self.send_message({
            "type": "private",
            "to": [USER_ID],
            "content": message,
        })

    def process_message(self, msg):
        sender_email = msg['sender_email']
        message_content = msg['content']
        message_type = msg['type']

        # Check if the message is a private message or a mention in a stream
        if message_type == 'private' or message_content.startswith('@**ChatGPT**'):
            if message_content.startswith('@**ChatGPT**'):
                prompt = re.sub('@\*\*ChatGPT\*\*', '', message_content).strip()
            else:
                prompt = message_content

            response = get_chatgpt_response(prompt)

            self.send_message({
                "type": "private",
                "to": sender_email,
                "content": response,
            })

def on_exit(bot, recipient_email):
    bot.send_notification("The ChatGPT bot is now offline.", recipient_email)
    bot.set_status(True)

if __name__ == "__main__":
    bot = ChatGPTZulipBot(ZULIP_CONFIG)
    bot.send_notification("The ChatGPT bot is now online.", USER_ID)
    bot.set_status(False)
    print("Successfully started the ChatGPT bot.")

    bot.call_on_each_message(bot.process_message)
    atexit.register(on_exit, bot, USER_ID)
