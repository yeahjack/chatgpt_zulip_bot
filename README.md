# ChatGPT Zulip Bot

The ChatGPT Zulip Bot is a bot that responds to users by using the ChatGPT language model. It can be used in any Zulip chat stream or private message.

## Installation

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
- Download the configuartion file and move it to this folder. Rename it to `zuliprc`.
- Define `USER_ID` to the user id of the admin, and "BOT_NAME" where the name is what you gave to the bot when you created it.

4. Start the bot:

```bash
python chatgpt_zulip_bot.py
```

5. Errors

When having errors, the bot will send a message to the admin and set the status to `away`.

## Usage

To use the bot, mention it in any Zulip stream or send it a private message. You can mention the bot by typing `@bot_name` where the name is what you gave to the bot when you created it.

The bot will respond to your message by generating text using the ChatGPT language model. You can customize the bot's behavior by modifying the `chatgpt_zulip_bot.py` file.

## Contact

Feel free to contact me or leave an issue if you have any questions or suggestions.
