# ChatGPT Zulip Bot

The ChatGPT Zulip Bot is a bot that responds to users by using the ChatGPT language model. It can be used in any Zulip chat stream or private message.

## Updates
### March 27, 2023
- Upgraded the model from `text-davinci-003` to `gpt-turbo-3.5`. It can be configured in `chatgpt.py`.
- Added contextual support, allowing the bot to answer questions based on the context of the conversation. Type `停止会话` or `end the conversation` to end the current conversation.
- Implemented conversation history trimming to ensure it stays within OpenAI's maximum token limit.
- Fixed a bug causing the bot to crash during long conversations.
- Resolved an issue where the bot would reply privately when mentioned in a group conversation.

## Current Bugs
- The bot's status cannot be set to "away" when it's offline or not running.

# Installation

1. Clone the repository:

```bash
git clone https://github.com/yeahjack/chatgpt_zulip_bot.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your Zulip bot:

- Go to your Zulip organization settings, and navigate to the "Your bots" section.
- Click the "Add a new bot" button and follow the prompts to create a new bot.
- Download the configuartion file and move it to this folder.
- Rename `config.ini.example` to `config.ini` and fill in the values, note that you do not have to add `" "` around the values.


4. Start the bot:

```bash
python chatgpt_zulip_bot.py
```

5. Errors

When having errors, the bot will send a message to the admin and set the status to `away` (**Seems the later does not work**).

## Usage

To use the bot, mention it in any Zulip stream or send it a private message. You can mention the bot by typing `@bot_name` where the name is what you gave to the bot when you created it.

The bot will respond to your message by generating text using the ChatGPT language model. You can customize the bot's behavior by modifying the `chatgpt_zulip_bot.py` file.

## Contact

Feel free to contact me or leave an issue if you have any questions or suggestions.
