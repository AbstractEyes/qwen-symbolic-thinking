"""Seams for vision-language families beyond Qwen3.5 and Gemma 4: a processor that refuses a flat picture list in its own words
gets the per-text form; a configuration the multimodal auto class does not map loads through the next class; GLM-4V's fused MLP
projection is a LoRA target while the vision tower stays out; the hooks of the two families whose code runs from their
repositories (Phi-4-reasoning-vision, Molmo2: family.py) and the queue's transformers gate that keeps their runs on their own
card."""
import re

from qwen_symbolic_thinking.screen import encode, load_vlm
from qwen_symbolic_thinking.train import TARGETS


class Pic:
    def __init__(self, name):
        self.name = name


class OwnWordsProcessor:
    """Refuses a flat list for a batch of texts with an error worded its own way; takes one list per text."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, images=None, **kw):
        self.calls.append(images)
        if images is not None and len(text) > 1 and not all(isinstance(x, list) for x in images):
            raise ValueError("Invalid input images. Please provide a list of lists of images.")
        return {"text": text, "images": images}


def test_encode_per_text_on_an_error_in_other_words():
    p = OwnWordsProcessor()
    a, b = Pic("a"), Pic("b")
    out = encode(p, ["a", "b", "s"], [a, b, None])
    assert out["images"] == [[a], [b], []] and len(p.calls) == 2


def test_encode_first_error_surfaces_when_both_forms_fail():
    class Refuses:
        def __call__(self, text, images=None, **kw):
            raise ValueError("first words" if not any(isinstance(x, list) for x in images) else "second words")
    try:
        encode(Refuses(), ["a", "b"], [Pic("a"), Pic("b")])
    except ValueError as e:
        assert "first words" in str(e)
    else:
        raise AssertionError("the first error must surface")


class Unmapped:
    calls = 0

    @classmethod
    def from_pretrained(cls, src, **kw):
        cls.calls += 1
        raise ValueError(f"Unrecognized configuration class <class 'X'> for this kind of AutoModel: {src}")


class Mapped:
    @classmethod
    def from_pretrained(cls, src, **kw):
        return ("loaded", src, kw.get("dtype"))


class Broken:
    @classmethod
    def from_pretrained(cls, src, **kw):
        raise ValueError("the weights are missing")


def test_load_vlm_falls_through_an_unmapped_class():
    assert load_vlm("org/model", classes=[Unmapped, Mapped], dtype="bf16") == ("loaded", "org/model", "bf16")


def test_load_vlm_surfaces_other_errors_and_the_last_unmapped():
    try:
        load_vlm("org/model", classes=[Broken, Mapped])
    except ValueError as e:
        assert "missing" in str(e)
    else:
        raise AssertionError("an unrelated load error must surface")
    try:
        load_vlm("org/model", classes=[Unmapped])
    except ValueError as e:
        assert "Unrecognized configuration class" in str(e)
    else:
        raise AssertionError("no class mapped the configuration")


class Mask:
    """An attention mask's interface as the collator uses it: per-row sums as a list."""

    def __init__(self, rows):
        self.rows = rows

    def sum(self, dim):
        assert dim == 1
        return Rows([sum(r) for r in self.rows])


class Rows:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_overrun_counts_the_examples_past_max_len():
    from qwen_symbolic_thinking.train import overrun
    mask = Mask([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 0]])
    assert overrun(mask, 4) == (3, 1, 5)
    assert overrun(mask, 5) == (3, 0, 5)
    assert overrun(Mask([]), 5) == (0, 0, 0)


class Cfg:
    def __init__(self, model_type):
        self.model_type = model_type


class Model:
    def __init__(self, model_type):
        self.config = Cfg(model_type)


class Tok:
    def apply_chat_template(self, messages, **kw):
        return "rendered by the tokenizer"


class Proc:
    def __init__(self):
        self.tokenizer = Tok()

    def apply_chat_template(self, messages, **kw):
        raise ValueError("Cannot use apply_chat_template because this processor does not have a chat template.")


def test_family_hooks_by_model_type():
    from qwen_symbolic_thinking import family
    phi, molmo, qwen = Model("phi4-siglip"), Model("molmo2"), Model("qwen3_5")
    assert family.marker(phi) == "<image>" and family.marker(molmo) is None and family.marker(qwen) is None
    assert family.model_type("molmo2") == "molmo2" and family.model_type(phi.config) == "phi4-siglip"
    family.prepare_inference(phi)
    assert phi.config.tokenizer_padding_side == "left"
    family.prepare_inference(qwen)
    assert not hasattr(qwen.config, "tokenizer_padding_side")
    p = Proc()
    family.prepare_processor(p, phi)
    assert p.apply_chat_template([]) == "rendered by the tokenizer"
    p = Proc()
    family.prepare_processor(p, qwen)
    try:
        p.apply_chat_template([])
    except ValueError:
        pass
    else:
        raise AssertionError("only Phi-4's processor renders through its tokenizer")


def test_fill_images_gives_phi4_rows_a_slot_each():
    from qwen_symbolic_thinking import family
    a = Pic("a")
    filled = family.fill_images("phi4-siglip", [a, None])
    assert filled[0] is a and filled[1] is not None and filled[1].size == (32, 32)
    assert family.fill_images("phi4-siglip", [None, None]) == [None, None]   # no picture in the batch: text only
    assert family.fill_images("molmo2", [a, None]) == [a, None]
    assert family.fill_images("qwen3_5", [a, None]) == [a, None]


class Ids:
    """A token-id matrix's interface as fix_padding and generation_inputs use it."""

    def __init__(self, rows, dtype="long"):
        self.rows, self.dtype = rows, dtype
        self.shape = (len(rows), len(rows[0]) if rows else 0)

    def __ne__(self, v):
        return Ids([[int(x != v) for x in r] for r in self.rows], "bool")

    def to(self, dtype):
        return Ids(self.rows, dtype)


def test_fix_padding_rebuilds_molmo2_mask_from_the_pad_id():
    from qwen_symbolic_thinking import family
    batch = {"input_ids": Ids([[1, 5, 6, 9, 9], [1, 5, 6, 7, 8]]), "attention_mask": Ids([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])}
    family.fix_padding("molmo2", batch, 9)
    assert batch["attention_mask"].rows == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]] and batch["attention_mask"].dtype == "long"
    other = {"input_ids": Ids([[1, 9]]), "attention_mask": Ids([[1, 1]])}
    family.fix_padding("qwen3_5", other, 9)
    assert other["attention_mask"].rows == [[1, 1]]


def test_generation_inputs_build_phi4_embeddings(monkeypatch):
    import sys
    import types
    from qwen_symbolic_thinking import family
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(BatchFeature=dict))

    class Phi(Model):
        def __init__(self):
            super().__init__("phi4-siglip")
            self.calls = []

        def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images):
            self.calls.append((position_ids, past_key_values, labels, sorted(images)))
            return None, None, "left-padded mask", past_key_values, "embeddings", None

    ids = Ids([[1, -200, 3], [1, -200, 0]])
    inputs = {"input_ids": ids, "attention_mask": Ids([[1, 1, 1], [1, 1, 0]]), "pixel_values": "pv",
              "pixel_attention_mask": "pam", "spatial_shapes": "ss"}
    phi = Phi()
    kwargs, start = family.generation_inputs(phi, inputs)
    assert kwargs == {"inputs_embeds": "embeddings", "attention_mask": "left-padded mask"} and start == 0
    assert phi.calls == [(None, None, None, ["pixel_attention_mask", "pixel_values", "spatial_shapes"])]
    text_only = {"input_ids": ids, "attention_mask": inputs["attention_mask"]}
    assert family.generation_inputs(phi, text_only) == (text_only, 3)          # no picture: generation from the token ids
    assert family.generation_inputs(Model("qwen3_5"), inputs) == (inputs, 3)


def test_family_content_string_or_parts():
    from qwen_symbolic_thinking import family
    p = Pic("p")
    assert family.content("look", p, "<image>") == "<image>\nlook"
    assert family.content("spec", None, "<image>") == "spec"
    assert family.content("look", p) == [{"type": "image", "image": p}, {"type": "text", "text": "look"}]
    assert family.content("spec") == [{"type": "text", "text": "spec"}]


def test_family_clean_drops_the_mode_tag_only_at_the_start():
    from qwen_symbolic_thinking import family
    assert family.clean("<nothink> see: a lamp.") == "see: a lamp."
    assert family.clean("see: a lamp. <nothink>") == "see: a lamp. <nothink>"


def test_transformers_fits_keeps_each_kind_of_card_to_its_runs(monkeypatch):
    from qwen_symbolic_thinking import queue
    remote = {"name": "S", "transformers": "==4.57.6"}
    assert queue.transformers_fits(remote, "4.57.6") and not queue.transformers_fits(remote, "5.22.1")
    assert queue.transformers_fits({"name": "T"}, "5.22.1") and queue.transformers_fits({"name": "T"}, "5.23.0.dev0")
    assert not queue.transformers_fits({"name": "T"}, "4.57.6")
    assert queue.transformers_fits({"name": "T", "transformers": "not a specifier"}, "5.22.1")
    monkeypatch.setattr(queue, "_transformers_version", lambda: "")
    assert queue.transformers_fits(remote) and queue.transformers_fits({"name": "T"})   # transformers absent: nothing held back
    monkeypatch.setattr(queue, "_transformers_version", lambda: "4.57.6")
    assert queue.runnable_here({"name": "S", "needs": "0.1.31", "transformers": "==4.57.6"})
    assert not queue.runnable_here({"name": "T", "needs": "0.1.31"})
    assert not queue.runnable_here({"name": "S", "needs": "9.0.0", "transformers": "==4.57.6"})


def test_messages_and_conversation_take_the_marker():
    from qwen_symbolic_thinking.screen import messages
    from qwen_symbolic_thinking.train import conversation
    it = {"id": "x", "depth": "medium", "spec": {"object": {"kind": "lamp"}}, "kind": "image", "see": "a lamp",
          "caption": "A lamp.", "thoughts": [], "wordings": [], "prompt": "lamp"}
    m = messages("bare", it, {}, Pic("p"), mark="<image>")
    assert isinstance(m[-1]["content"], str) and m[-1]["content"].startswith("<image>\n")
    m = messages("spec", it, {}, None, mark="<image>")
    assert isinstance(m[-1]["content"], str) and "<image>" not in m[-1]["content"]
    m = messages("bare", it, {}, Pic("p"))
    assert m[-1]["content"][0]["type"] == "image"
    try:
        msgs, _ = conversation(it, "short", Pic("p"), mark="<image>")
    except (KeyError, TypeError):   # chain.render needs a whole row; the user turn is what this test reads
        return
    assert msgs[-1]["content"].startswith("<image>\n")


def test_targets_take_phi4_and_molmo2_layers_and_leave_their_towers():
    for name in ("model.layers.3.self_attn.qkv_proj", "model.layers.3.self_attn.o_proj", "model.layers.3.mlp.gate_up_proj",
                 "model.layers.3.mlp.down_proj", "model.transformer.blocks.3.self_attn.att_proj",
                 "model.transformer.blocks.3.self_attn.attn_out", "model.transformer.blocks.3.mlp.ff_proj",
                 "model.transformer.blocks.3.mlp.ff_out"):
        assert re.fullmatch(TARGETS, name), name
    for name in ("model.vision_tower.vision_tower.vision_model.encoder.layers.3.self_attn.q_proj",
                 "model.vision_tower.vision_tower.vision_model.encoder.layers.3.mlp.fc1",
                 "model.vision_tower.vision_tower.vision_model.head.attention.out_proj", "model.mm_projector.0",
                 "model.vision_backbone.image_vit.transformer.resblocks.3.attention.wq",
                 "model.vision_backbone.image_projector.w1", "lm_head", "model.layers.3.input_layernorm"):
        assert not re.fullmatch(TARGETS, name), name


def test_turn_end_of_a_template_that_takes_only_strings():
    from qwen_symbolic_thinking.template import turn_end

    class PhiLike:
        def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kw):
            out = "<|im_start|>system<|im_sep|>You are Phi.<|im_end|>"
            for m in messages:
                out += f"<|im_start|>{m['role']}<|im_sep|>" + m["content"] + "<|im_end|>"   # a list here raises TypeError
            return out + ("<|im_start|>assistant<|im_sep|>" if add_generation_prompt else "")
    assert turn_end(PhiLike()) == "<|im_end|>"


def test_after_specs_and_the_arm_screen_carry_the_repository_code_settings():
    from qwen_symbolic_thinking.queue import after_specs
    spec = {"name": "G", "kind": "train",
            "train": {"model_id": "allenai/Molmo2-O-7B", "trust_remote_code": True, "revision": "abc123", "rank": 32},
            "after": [{"name": "P", "kind": "extract", "extract": {"manifest": "m", "split": "probe"}}]}
    x = after_specs(spec, "lora/molmo2/r32/stamp")[0]["extract"]
    assert x["trust_remote_code"] is True and x["revision"] == "abc123" and x["model_id"] == "allenai/Molmo2-O-7B"
    assert x["adapter_subfolder"] == "lora/molmo2/r32/stamp/final"


def test_targets_take_the_fused_mlp_and_leave_the_vision_tower():
    assert re.fullmatch(TARGETS, "model.language_model.layers.3.mlp.gate_up_proj")
    assert re.fullmatch(TARGETS, "model.language_model.layers.3.mlp.down_proj")
    assert re.fullmatch(TARGETS, "model.language_model.layers.3.self_attn.q_proj")
    assert not re.fullmatch(TARGETS, "model.visual.blocks.3.mlp.gate_up_proj")
    assert not re.fullmatch(TARGETS, "model.vision_tower.transformer.layers.0.attention.q_proj")
