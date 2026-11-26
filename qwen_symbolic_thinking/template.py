"""What a model's own chat template says about the end of an answer: the text the template writes after an assistant turn
(the training examples end with it), the token ids generation must stop on (the model's own, the tokenizer's, and the
template's turn end), and the id of the turn end's own token (a LoRA may train that row of its output head: train.run
token_rows)."""
from __future__ import annotations


def turn_end(processor) -> str:
    """What the chat template writes after an assistant turn's text ("<|im_end|>\\n" for Qwen3.5, "<turn|>\\n" for Gemma 4), read
    from the template itself: a two-turn conversation is rendered without a generation prompt and the text after the answer is
    taken. A training example is the generation prompt (thinking off) + the answer + this, so the model learns to write the row
    exactly where it will be asked to at inference."""
    marker = "ANSWERMARKER"
    try:
        text = processor.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": "q"}]},
                                              {"role": "assistant", "content": [{"type": "text", "text": marker}]}],
                                             add_generation_prompt=False, tokenize=False)
    except Exception as first:   # a template that concatenates the content as a string (Phi-4-reasoning-vision)
        try:
            text = processor.apply_chat_template([{"role": "user", "content": "q"}, {"role": "assistant", "content": marker}],
                                                 add_generation_prompt=False, tokenize=False)
        except Exception:
            raise first from None
    assert marker in text, "the chat template did not render the assistant turn's text"
    end = text[text.rindex(marker) + len(marker):]
    if not end.strip():
        # A template that writes nothing after an answer (GLM-4V opens the next turn instead): the answer ends with the tokenizer's
        # end-of-sequence token, which is among the ids generation stops on.
        eos = getattr(getattr(processor, "tokenizer", None), "eos_token", None)
        assert eos, f"the chat template writes nothing after an assistant turn and the tokenizer names no end token: {text[-80:]!r}"
        return eos
    return end


def turn_end_id(processor, model=None) -> int:
    """The id of the token that opens the chat template's turn end (<|im_end|> for Qwen3.5 and InternVL3.5, <turn|> for Gemma 4):
    the turn end (turn_end) without its trailing whitespace, which must be one token of the vocabulary; with the model, it must
    also be one of the ids generation stops on (stop_ids), so it is the token every read stops on. Raises ValueError otherwise
    (a turn end of several tokens, or none at all: a template that closes an answer with a bare newline and a tokenizer that
    names no end token)."""
    try:
        end = turn_end(processor).strip()
    except AssertionError as e:
        raise ValueError(f"the chat template has no turn end to name: {e}") from None
    ids = processor.tokenizer(end, add_special_tokens=False)["input_ids"] if end else []
    if len(ids) != 1:
        raise ValueError(f"the chat template's turn end {end!r} is {len(ids)} tokens of the vocabulary, not one: {ids}")
    if model is not None and int(ids[0]) not in stop_ids(processor, model):
        raise ValueError(f"the turn end {end!r} (id {ids[0]}) is not among the ids generation stops on {stop_ids(processor, model)}")
    return int(ids[0])


def _ids(x) -> list:
    if x is None:
        return []
    return [int(v) for v in (x if isinstance(x, (list, tuple)) else [x]) if v is not None]


def stop_ids(processor, model) -> list:
    """The ids generation stops on, in one sorted list: the model's own end ids (its generation config, else its model config), the
    tokenizer's end-of-sequence id, and the token that opens the chat template's turn end when it is one token of the vocabulary
    (<|im_end|> for Qwen3.5, <turn|> for Gemma 4). Qwen3.5 ships no generation config and its model config's end id is
    <|endoftext|>, not the <|im_end|> that closes a turn: a base model happens to write <|endoftext|> right after <|im_end|>, so it
    stops anyway; a fully fine-tuned one no longer does, and without the turn end among the stop ids it ran to the token cap on
    every row, repeating the row (the 0.8B full fine-tune, 2026-11-01)."""
    tok = processor.tokenizer
    ids = _ids(getattr(getattr(model, "generation_config", None), "eos_token_id", None))
    if not ids:
        ids = _ids(getattr(getattr(model, "config", None), "eos_token_id", None))
    ids += _ids(getattr(tok, "eos_token_id", None))
    try:
        end = turn_end(processor)
    except AssertionError:          # a template with no turn end to speak of: the end ids above are all there is
        end = ""
    first = tok(end, add_special_tokens=False)["input_ids"] if end else []
    if first and tok.decode(first[:1]) == end.strip():
        ids += _ids(first[0])
    assert ids, "no end-of-sequence id: neither the model nor the tokenizer names one"
    return sorted(set(ids))
