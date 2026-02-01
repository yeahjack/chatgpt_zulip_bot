# chatgpt_zulip_bot.py
"""
Zulip bot that integrates with OpenAI for course assistance.
Restricted to specific streams and their members.
"""

import zulip
import re
import atexit
import logging
from configparser import ConfigParser

from chatgpt import ChatBot


class ChatGPTZulipBot(zulip.Client):
    """Zulip bot that responds to messages using OpenAI."""
    
    def __init__(
        self, 
        config_file: str, 
        user_id: int, 
        bot_id: int, 
        bot_name: str, 
        chatbot: ChatBot,
        allowed_streams: list = None
    ):
        super().__init__(config_file=config_file)
        self.user_id = user_id
        self.bot_id = bot_id
        self.bot_name = bot_name
        self.chatbot = chatbot
        self.allowed_streams = allowed_streams or []
        
        # Cache for allowed users (stream subscribers) and stream IDs
        self.allowed_users = set()
        self.allowed_stream_ids = set()
        
        # Load allowed users if stream restriction is enabled
        if self.allowed_streams:
            self._load_stream_subscribers()
    
    def _load_stream_subscribers(self):
        """Load the list of subscribers from all allowed streams."""
        if not self.allowed_streams:
            return
        
        self.allowed_users = set()
        self.allowed_stream_ids = set()
        
        for stream_name in self.allowed_streams:
            # Get stream ID
            result = self.get_stream_id(stream_name)
            if result["result"] != "success":
                logging.error(f"Failed to get stream ID for '{stream_name}': {result}")
                continue
            
            stream_id = result["stream_id"]
            self.allowed_stream_ids.add(stream_id)
            logging.info(f"Allowed stream: {stream_name} (ID: {stream_id})")
            
            # Get subscribers
            result = self.get_subscribers(stream=stream_name)
            if result["result"] == "success":
                subscribers = set(result["subscribers"])
                self.allowed_users.update(subscribers)
                logging.info(f"Loaded {len(subscribers)} subscribers from '{stream_name}'")
            else:
                logging.error(f"Failed to get subscribers for '{stream_name}': {result}")
        
        logging.info(f"Total allowed users: {len(self.allowed_users)}")
    
    def refresh_subscribers(self):
        """Refresh the list of allowed users from all streams."""
        self._load_stream_subscribers()
    
    def is_user_allowed(self, user_id: int) -> bool:
        """Check if a user is allowed to use the bot."""
        # If no stream restriction, allow everyone
        if not self.allowed_streams:
            return True
        
        # Admin is always allowed
        if user_id == self.user_id:
            return True
        
        # Check if user is in any allowed stream
        return user_id in self.allowed_users
    
    def is_stream_allowed(self, stream_name: str = None, stream_id: int = None) -> bool:
        """Check if a stream is allowed."""
        if not self.allowed_streams:
            return True
        
        if stream_name:
            return stream_name in self.allowed_streams
        if stream_id:
            return stream_id in self.allowed_stream_ids
        return False

    def send_notification(self, message: str):
        """Send a private notification to the admin user."""
        self.send_message({
            "type": "private",
            "to": [self.user_id],
            "content": message,
        })

    def process_message(self, msg: dict):
        """Process incoming messages and respond."""
        sender_id = msg["sender_id"]
        sender_email = msg["sender_email"]
        sender_name = msg.get("sender_full_name", "")
        message_id = msg.get("id")
        message_content = msg["content"]
        message_type = msg["type"]
        
        # Ignore messages from self
        if sender_id == self.bot_id:
            return
        
        # Handle mentions in streams (regular @**Name** or quote reply @_**Name|ID**)
        # Use search() to detect mentions anywhere in the message, not just at the start
        mention_pattern = rf"@_?\*\*{re.escape(self.bot_name)}(\|\d+)?\*\*"
        if re.search(mention_pattern, message_content):
            stream_id = msg.get("stream_id")
            stream_name = msg.get("display_recipient")  # Stream name for stream messages
            topic = msg.get("subject")
            
            # Check if stream is allowed
            if not self.is_stream_allowed(stream_name=stream_name, stream_id=stream_id):
                logging.info(f"Ignored message from unauthorized stream: {stream_name}")
                return
            
            # Strip the triggering bot mention (and [said](url): if quote-reply)
            # This removes: "@**ChatGPT** " or "@_**ChatGPT|132** [said](url):"
            trigger_pattern = rf"@_?\*\*{re.escape(self.bot_name)}(\|\d+)?\*\*(\s*\[said\]\([^)]+\):)?\s*"
            original_message = re.sub(trigger_pattern, "", message_content, count=1).strip()
            
            # For AI processing: also strip quote blocks
            quote_pattern = r'(`{3,})quote\s.*?\1'
            prompt = re.sub(quote_pattern, "", original_message, flags=re.DOTALL).strip()
            
            # Handle admin commands
            if sender_id == self.user_id and prompt.lower() == "/refresh":
                self.refresh_subscribers()
                response = f"Refreshed subscriber list. Now tracking {len(self.allowed_users)} users."
            else:
                # Stream: RAG mode, no chaining
                # Construct message URL for [said](url) format
                # URL encode the topic for the link
                import urllib.parse
                encoded_topic = urllib.parse.quote(topic, safe='')
                message_url = f"#narrow/channel/{stream_id}-{stream_name}/topic/{encoded_topic}/near/{message_id}"
                
                response = self.chatbot.get_stream_response(
                    sender_email, prompt, original_message, 
                    sender_name, sender_id, message_url
                )
            
            self.send_message({
                "type": "stream",
                "to": stream_id,
                "subject": topic,
                "content": response,
            })
        
        # Handle private messages (DM)
        elif message_type == "private":
            # Check if user is allowed (member of the allowed stream)
            if not self.is_user_allowed(sender_id):
                logging.info(f"Ignored private message from unauthorized user: {sender_email}")
                self.send_message({
                    "type": "private",
                    "to": sender_email,
                    "content": f"Sorry, this bot is only available to members of: **{', '.join(self.allowed_streams)}**",
                })
                return
            
            # Handle admin commands
            if sender_id == self.user_id and message_content.lower().strip() == "/refresh":
                self.refresh_subscribers()
                response = f"Refreshed subscriber list. Now tracking {len(self.allowed_users)} users."
            else:
                # DM: Weekly mode with chaining, user must specify week
                response = self.chatbot.get_dm_response(sender_email, message_content)
            
            self.send_message({
                "type": "private",
                "to": sender_email,
                "content": response,
            })


def on_exit(bot: ChatGPTZulipBot):
    """Send offline notification when bot exits."""
    bot.send_notification("NOTICE: The ChatGPT bot is now offline.")


def serve(config_file: str = "config.ini"):
    """Start the Zulip bot server."""
    config = ConfigParser()
    config.read(config_file)
    settings = config["settings"]
    
    # OpenAI settings
    api_key = settings["OPENAI_API_KEY"]
    model = settings.get("MODEL") or settings.get("API_VERSION", "gpt-4o")
    course_dir = settings.get("COURSE_DIR")
    
    # File patterns to filter course materials (comma-separated)
    file_patterns_str = settings.get("FILE_PATTERNS", "")
    file_patterns = [p.strip() for p in file_patterns_str.split(",") if p.strip()]
    
    # Vector store ID for RAG (both modes)
    vector_store_id = settings.get("VECTOR_STORE_ID")
    
    # Optional: override auto-detected max tokens
    max_output_tokens = None
    if "MAXIMUM_CONTENT_LENGTH" in settings:
        max_output_tokens = int(settings["MAXIMUM_CONTENT_LENGTH"])
    
    # Optional: log Q&A pairs (default: true)
    log_qa = settings.get("LOG_QA", "true").lower() in ("true", "1", "yes")
    
    # Zulip settings
    zulip_config = settings["ZULIP_CONFIG"]
    user_id = int(settings["USER_ID"])
    bot_id = int(settings["BOT_ID"])
    bot_name = settings["BOT_NAME"]
    
    # Access control: restrict to specific streams (comma-separated)
    allowed_streams_str = settings.get("ALLOWED_STREAMS") or settings.get("ALLOWED_STREAM")
    allowed_streams = []
    if allowed_streams_str:
        allowed_streams = [s.strip() for s in allowed_streams_str.split(",") if s.strip()]
    
    # Initialize chatbot
    chatbot = ChatBot(
        model=model,
        api_key=api_key,
        course_dir=course_dir,
        file_patterns=file_patterns,
        vector_store_id=vector_store_id,
        max_output_tokens=max_output_tokens,
        log_qa=log_qa,
    )
    
    # Initialize Zulip bot
    bot = ChatGPTZulipBot(
        zulip_config, user_id, bot_id, bot_name, chatbot,
        allowed_streams=allowed_streams
    )
    
    # Print startup info
    print(f"ChatGPT bot starting (model: {model})")
    print(f"  Stream mode: Responses API + RAG")
    print(f"  DM mode: Responses API + Conversations")
    print(f"  Vector store: {vector_store_id or 'NOT SET'}")
    print(f"  Q&A logging: {'enabled' if log_qa else 'disabled'}")
    
    if allowed_streams:
        print(f"  Access restricted to: {', '.join(allowed_streams)}")
        print(f"  Authorized users: {len(bot.allowed_users)}")
    
    if not vector_store_id:
        print("\n⚠️  WARNING: VECTOR_STORE_ID not set. RAG/file search will not work.")
        print("   Run 'make upload' to create a vector store.\n")
    
    bot.send_notification("NOTICE: The ChatGPT bot is now online.")
    print("Bot is now online and listening for messages...")
    
    atexit.register(on_exit, bot)
    bot.call_on_each_message(bot.process_message)


if __name__ == "__main__":
    logging.basicConfig(
        filename="bot.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    serve("config.ini")
