# Updates
## March 28, 2023
- Add `/end` to end the current conversation, which is shorter and fits Zulip syntexes.

## March 27, 2023
- Upgraded the model from `text-davinci-003` to `gpt-3.5-turbo` by default. It can be configured in `config.ini`.
- Added contextual support, allowing the bot to answer questions based on the context of the conversation. Type `停止会话` or `end the conversation` to end the current conversation.
- The output will also show numbers of tokens used in the conversation.
- Implemented conversation history trimming to ensure it stays within OpenAI's maximum token limit.
- Fixed a bug causing the bot to crash during long conversations.
- Resolved an issue where the bot would reply privately when mentioned in a group conversation.


