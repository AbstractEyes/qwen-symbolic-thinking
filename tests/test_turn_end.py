"""The turn end comes from the chat template, so a training example ends the way the model's own template ends an answer; the
stop ids at generation include that turn end beside the model's and the tokenizer's end ids."""
from qwen_symbolic_thinking.template import stop_ids, turn_end
from qwen_symbolic_thinking import train


class Tokens:
    """A tokenizer over a fixed vocabulary of special tokens: text is split into those tokens and newlines."""

    def __init__(self, vocab: dict, eos: str, pad: str):
        self.vocab, self.eos_token_id, self.pad_token_id = vocab, vocab[eos], vocab[pad]
        self.back = {v: k for k, v in vocab.items()}

    def __call__(self, text, add_special_tokens=False):
        ids, rest = [], text
        while rest:
            for tok in sorted(self.vocab, key=len, reverse=True):
                if rest.startswith(tok):
                    ids.append(self.vocab[tok]); rest = rest[len(tok):]; break
            else:
                raise ValueError(rest)
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(self.back[i] for i in ids)


class Gen:
    def __init__(self, eos):
        self.eos_token_id = eos


class Model:
    def __init__(self, gen_eos=None, config_eos=None):
        self.generation_config = Gen(gen_eos) if gen_eos is not None else None
        self.config = Gen(config_eos)


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


class QwenProc(QwenLike):
    tokenizer = Tokens({"<|endoftext|>": 248044, "<|im_start|>": 248045, "<|im_end|>": 248046, "\n": 198},
                       eos="<|im_end|>", pad="<|endoftext|>")


class GemmaProc(GemmaLike):
    tokenizer = Tokens({"<pad>": 0, "<eos>": 1, "<|turn>": 105, "<turn|>": 106, "\n": 107}, eos="<eos>", pad="<pad>")


class NewlineEnd(QwenLike):
    """A template that closes an answer with a bare newline: no vocabulary token to stop on, so only the end ids remain."""
    tokenizer = QwenProc.tokenizer

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
        return "".join(f"{m['role']}: " + "".join(c["text"] for c in m["content"]) + "\n" for m in messages)


def test_stop_ids_qwen_adds_the_turn_end_to_the_model_config_end():
    """Qwen3.5 ships no generation config and its model config names <|endoftext|>; the stop ids add <|im_end|> from the template."""
    assert stop_ids(QwenProc(), Model(config_eos=248044)) == [248044, 248046]
    assert train.turn_end is turn_end


def test_stop_ids_gemma_keeps_the_generation_config_list():
    assert stop_ids(GemmaProc(), Model(gen_eos=[1, 106, 50])) == [1, 50, 106]


def test_stop_ids_never_a_newline():
    assert stop_ids(NewlineEnd(), Model(gen_eos=248044)) == [248044, 248046]   # the model's and the tokenizer's, not "\n"
