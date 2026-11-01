"""The turn end comes from the chat template, so a training example ends the way the model's own template ends an answer."""
from qwen_symbolic_thinking.train import turn_end


class QwenLike:
    """Renders the way Qwen3.5's template does: <|im_start|>role\\ntext<|im_end|>\\n per turn."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        out = ""
        for m in messages:
            text = "".join(c["text"] for c in m["content"] if c["type"] == "text")
            out += f"<|im_start|>{m['role']}\n{text}<|im_end|>\n"
        if add_generation_prompt:
            out += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        return out


class GemmaLike:
    """Renders the way Gemma 4's template does: <|turn>role\\ntext<turn|>\\n per turn, the empty thought block only in the
    generation prompt."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        out = "<bos>"
        for m in messages:
            role = "model" if m["role"] == "assistant" else m["role"]
            text = "".join(c["text"] for c in m["content"] if c["type"] == "text")
            out += f"<|turn>{role}\n{text}<turn|>\n"
        if add_generation_prompt:
            out += "<|turn>model\n<|channel>thought\n<channel|>"
        return out


class Bare:
    """A template that drops the assistant turn's text altogether: the trainer must refuse it."""

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        return "".join(f"<{m['role']}>" for m in messages)


def test_turn_end_qwen():
    assert turn_end(QwenLike()) == "<|im_end|>\n"


def test_turn_end_gemma():
    assert turn_end(GemmaLike()) == "<turn|>\n"


def test_turn_end_refuses_a_template_without_the_answer():
    try:
        turn_end(Bare())
    except AssertionError as e:
        assert "assistant turn" in str(e)
    else:
        raise AssertionError("a template that drops the answer must be refused")


def test_training_text_matches_the_generation_prompt():
    """A training example is the generation prompt + the answer + the turn end: for Gemma the empty thought block the model will
    see at inference is inside the prompt, so the answer follows it in training too."""
    p = GemmaLike()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "look"}]}]
    prefix = p.apply_chat_template(msgs, add_generation_prompt=True)
    text = prefix + "see: a lamp." + turn_end(p)
    assert text.endswith("<|turn>model\n<|channel>thought\n<channel|>see: a lamp.<turn|>\n")
