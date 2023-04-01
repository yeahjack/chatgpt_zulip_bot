# Updates
## April 2, 2023
- Fixed bugs during the refactoring.
- Improve the pytest support. Note that the pytest file should ONLY be runned locally!

## April 1, 2023
- Refactored the codes by [@GiggleLiu](https://github.com/GiggleLiu) to obey the DRY principle. Thanks for his contribution!
- Initial pytest support added.

## March 31, 2023
- Improved academic prompts added to the bot, type `/help` to access them.
- Context feature now disabled when using academic prompts.
- Fixed issue with conversation trimming when single prompt exceeding token limit.

## March 28, 2023
- Add `/end` to end the current conversation, which is shorter and fits Zulip syntexes.

## March 27, 2023
- Upgraded the model from `text-davinci-003` to `gpt-3.5-turbo` by default. It can be configured in `config.ini`.
- Added contextual support, allowing the bot to answer questions based on the context of the conversation. Type `停止会话` or `end the conversation` to end the current conversation.
- The output will also show numbers of tokens used in the conversation.
- Implemented conversation history trimming to ensure it stays within OpenAI's maximum token limit.
- Fixed a bug causing the bot to crash during long conversations.
- Resolved an issue where the bot would reply privately when mentioned in a group conversation.


