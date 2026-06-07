"""测试 prompt_converter 共享消息转换逻辑。"""

from asc.api.prompt_converter import convert_messages_to_prompt


class TestConvertMessagesToPrompt:
    """convert_messages_to_prompt 核心转换逻辑。"""

    def test_single_user_message(self):
        messages = [{"role": "user", "content": "Hello"}]
        prompt = convert_messages_to_prompt(messages)
        assert "[User]: Hello" in prompt
        assert prompt.endswith("[Assistant]:")

    def test_system_and_user(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        prompt = convert_messages_to_prompt(messages)
        assert "[System]: You are helpful." in prompt
        assert "[User]: Hi" in prompt

    def test_conversation(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        prompt = convert_messages_to_prompt(messages)
        assert "[User]: Hello" in prompt
        assert "[Assistant]: Hi there!" in prompt
        assert "[User]: How are you?" in prompt

    def test_multimodal_content(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "image_url", "url": "http://example.com/img.png"},
                    {"type": "text", "text": "in detail"},
                ],
            }
        ]
        prompt = convert_messages_to_prompt(messages)
        assert "Describe this in detail" in prompt

    def test_top_level_system(self):
        messages = [{"role": "user", "content": "Hi"}]
        prompt = convert_messages_to_prompt(messages, system="Be concise.")
        assert "[System]: Be concise." in prompt
        assert "[User]: Hi" in prompt

    def test_top_level_system_with_system_in_messages(self):
        messages = [
            {"role": "system", "content": "Inner system"},
            {"role": "user", "content": "Hi"},
        ]
        prompt = convert_messages_to_prompt(messages, system="Outer system")
        assert "[System]: Outer system" in prompt
        assert "[System]: Inner system" in prompt

    def test_unknown_role_defaults_to_user(self):
        messages = [{"role": "unknown", "content": "test"}]
        prompt = convert_messages_to_prompt(messages)
        assert "[User]: test" in prompt

    def test_empty_messages(self):
        prompt = convert_messages_to_prompt([])
        assert prompt == "[Assistant]:"

    def test_missing_content_defaults_to_empty(self):
        messages = [{"role": "user"}]
        prompt = convert_messages_to_prompt(messages)
        assert "[User]: " in prompt
