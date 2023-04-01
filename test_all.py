from chatgpt import prompt_manager, help_str, prompt_table

def test_prompt_manager():
    for command, prompt in prompt_table.items():
        result = prompt_manager("%s This is a test!"%command)
        assert isinstance(result, str)
    
    result = prompt_manager("/en-zh This is a test!")
    assert result == "As a translator, your task is to accurately translate text from English to Chinese. \
Please pay attention to context and accurately explain phrases and proverbs. \
Below is the text you need to translate: \n\nThis is a test!"