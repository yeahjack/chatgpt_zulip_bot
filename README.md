# ChatGPT Zulip Bot

![GitHub License](https://img.shields.io/github/license/yeahjack/chatgpt_zulip_bot) [![Python Tests](https://github.com/yeahjack/chatgpt_zulip_bot/actions/workflows/ci.yml/badge.svg)](https://github.com/yeahjack/chatgpt_zulip_bot/actions/workflows/ci.yml) ![visitors](https://visitor-badge.glitch.me/badge?page_id=yeahjack.chatgpt_zulip_bot&left_color=green&right_color=blue)

The ChatGPT Zulip Bot is a bot that responds to users by using the ChatGPT language model. It can be used in any Zulip chat streams or private messages.

# Installation

1. Clone the repository:

```bash
git clone https://github.com/yeahjack/chatgpt_zulip_bot.git
```

2. Install the required dependencies (You might need to create a virtual env if you like):

```bash
pip install -r requirements.txt
```

3. Set up your Zulip bot:

- Go to your Zulip organization settings, and navigate to the "Your bots" section.
- Click the "Add a new bot" button and follow the prompts to create a new bot.
- Download the configuartion file and move it to this folder.
- Rename `config.ini.example` to `config.ini` and fill in the values, note that you do not have to add commas (`" "`) around the values.
Here are the explanations of parameters in `config.ini`.

| Parameters | Details |
| --- | --- |
| OPENAI_API_KEY | Your OpenAI API Key. Could be set via [this](https://platform.openai.com/account/api-keys).
| ZULIP_CONFIG | The filename of your zulip bot configuration file.
| USER_ID | The ID of the admin. Used for notifications.
| BOT_ID | The ID of the zulip bot. Used for positioning.
| API_VERSION | The version of OpenAI models. e.g. `gpt-3.5-turbo` |

4. Start the bot:

```bash
python chatgpt_zulip_bot.py
```

## Proxies

If you need to set a proxy for the OpenAI API, the normal method would be
```python
import os
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
```
Remember to replace both strings to your proxy address and port.

# Usage

To use the bot, mention it in any Zulip stream or send it a private message. You can mention the bot by typing `@bot_name` where the name is what you gave to the bot when you created it.
Except normal texts, the bot also accepts the following commands

## Commands
* `/help`: print this usage information.
* `/end`: end the current conversation. Bot will answer questions based on the context of the conversation. If a conversation reaches its 3000 token limit (you will see a message: "ERROR: OpenAI API rate limit exceeded. Please retry."), then you must restart the conversation with `/end`.

# Testing

If you forked this repo and did some changes, you can run the tests to make sure everything is working fine. Each time you push a commit, the tests will be automatically run by GitHub Actions. Note that you need to set `API_VERSION` and `OPENAI_API_KEY` in GitHub secrets.

# Contributing and bug reports

Feel free to leave an [issue](https://github.com/yeahjack/chatgpt_zulip_bot/issues) if you have any questions or suggestions.
Pull requests are also welcomed. If you are interested in contributing this project, a good place to start is the `chatgpt_zulip_bot.py` and `chatgpt.py` files. You can customize the bot's behavior by modifying it.
