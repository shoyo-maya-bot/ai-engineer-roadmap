"""p06_template.py のテスト（06-3 チャットのテンプレート）."""

from roadmap_examples.llm import Message, rough_tokens
from roadmap_examples.p06_template import overhead_tokens, render_chat

MESSAGES = [Message("system", "短く答える"), Message("user", "首都は？")]


def test_render_chat_marks_roles_and_ends_with_assistant() -> None:
    text = render_chat(MESSAGES)
    assert text == "<|system|>\n短く答える<|end|>\n<|user|>\n首都は？<|end|>\n<|assistant|>\n"


def test_template_adds_tokens_beyond_the_contents() -> None:
    assert overhead_tokens(MESSAGES, rough_tokens) > 0
    assert overhead_tokens(MESSAGES, len) == len(render_chat(MESSAGES)) - len("短く答える") - len("首都は？")
