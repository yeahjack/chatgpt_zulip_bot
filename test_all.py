from chatgpt import prompt_manager

def test_prompt_manager():
    result = prompt_manager("/polish_en This is a test!")
    assert isinstance(result, str)