"""What differs by model family, read from the model's config (its model_type), for the families whose code or chat template
departs from the common path. Everything else (the auto classes, the chat template's own thinking switch and turn end, the
processor's picture handling) is shared.

- phi4-siglip (Phi-4-reasoning-vision): the processor carries no chat template (the tokenizer's renders the conversation); the
  template concatenates a message's content as a string, so a picture row's content is "<image>\\n" + the text (the processor
  puts the picture in the <image> slot); the model builds its picture embeddings inside forward and assigns the pictures of a
  batch to its rows in order, one per row (a row without a picture takes a slot it does not use), so a batch that has any
  picture gives every row one; its generation from token ids keeps the text attention mask after the first step, which is
  wrong for every padded row of a batch, so batched generation runs from embeddings built here, left-padded; its answers may
  open with a mode tag, <nothink>, which is not part of the row.
- molmo2 (Molmo2): the processor inserts the start token assuming left padding (generation pads left); under the right padding
  of a training batch it marks the padding as attended, so the training mask is rebuilt from the pad id; the model adds its
  picture features into the embedding output in place, which torch refuses on an output marked as requiring grad, so it trains
  without that mark (input_grads).
- mistral3 (Ministral 3): the processor config leaves do_pad null, which transformers 5 reads as no padding, so pictures of
  different sizes cannot share a batch ("stack expects each tensor to be equal size"); the image processor pads, its class
  default (the model crops each picture back by image_sizes). Its vision tower packs every picture of a batch into one sequence
  under a dense block mask; in a 64-picture batch on the card (bf16, about 250,000 patches) the pictures after the first few came
  out wrong, while one at a time they read right, so the vision tower takes a batch's pictures one at a time (prepare_model).
- internvl (InternVL3.5 through transformers' own InternVL class, no repository code): one 448-pixel tile a picture (256
  picture tokens, the processor's default); the chat template has no thinking switch and closes a turn with <|im_end|>; the
  processor pads on its own default side (left) whatever the tokenizer says, so it is made to follow the tokenizer
  (follow_padding_side). With the anima_eyes record (InternVL3.5-1B with Anima's own top layers put back: anima_eyes.py), the
  config carries the record "anima_eyes": Anima's slot
  (the language layers from first_base_layer up, and the final norm) is the Base's (Qwen3-0.6B-Base, Anima's text encoder):
  every load checks its fingerprint (prepare_model), the LoRA targets the layers below it only (lora_targets), and a full
  fine-tune keeps it frozen (frozen). InternVL as it is (no record) takes the common path.

The repository code of these families runs only when a spec asks for it (trust_remote_code), loaded from a local snapshot at a
pinned revision (the Phi-4 loader reads its own vision weights only from a local folder), on the transformers release it was
built on (4.56-4.57: Phi-4's image processor subclasses an image-processor module that transformers 5 rewrote), on a lane whose
queue carries only such runs (queue.transformers_fits)."""
import re

PHI4 = "phi4-siglip"
MOLMO2 = "molmo2"
MISTRAL3 = "mistral3"
INTERNVL = "internvl"
_MODE_TAG = re.compile(r"^\s*<nothink>\s*")


def model_type(model_or_config) -> str:
    """The family key: a model's or a config's model_type, or the key itself."""
    if isinstance(model_or_config, str):
        return model_or_config
    cfg = getattr(model_or_config, "config", model_or_config)
    return str(getattr(cfg, "model_type", "") or "")


def marker(model_or_config) -> str | None:
    """The picture marker for a family whose chat template takes the content as a string (None: content as a list of parts)."""
    return "<image>" if model_type(model_or_config) == PHI4 else None


def content(text: str, image=None, mark: str | None = None):
    """A user turn's content: the list of parts (a picture part, then the text) or, with a marker, one string."""
    if mark:
        return (mark + "\n" if image is not None else "") + text
    return ([{"type": "image", "image": image}] if image is not None else []) + [{"type": "text", "text": text}]


def input_grads(model_or_config) -> bool:
    """Whether training marks the input embeddings' output as requiring grad (the usual companion of gradient checkpointing over
    a frozen embedding table; the checkpointing here is non-reentrant, which does not need it). Not for Molmo2: its in-place
    picture add on that output fails ("a view of a leaf Variable that requires grad is being used in an in-place operation")."""
    return model_type(model_or_config) != MOLMO2


def prepare_model(model) -> str | None:
    """Patch a loaded model where its family's own code fails our batches; returns a one-line note of what changed (None when the
    family needs nothing or the model is patched already). Molmo2: its image batching per picture row (mixed_batches). Ministral 3:
    its vision tower one picture at a time (pictures_one_at_a_time). Anima's eyes: nothing patched; its slot's fingerprint is
    checked against its record (a model whose slot changed stops here), and its output head must not be tied to its word table
    (it was built untied: a transformers that tied them at load would have replaced the head)."""
    kind = model_type(model)
    if kind == MOLMO2:
        return mixed_batches(model)
    if kind == MISTRAL3:
        return pictures_one_at_a_time(model)
    rec = anima_eyes(model)
    if rec is not None:
        from .anima_eyes import check_slot
        fp = check_slot(model, int(rec["first_base_layer"]), rec.get("slot_fingerprint"))
        head, table = model.get_output_embeddings(), model.get_input_embeddings()
        if head is not None and table is not None and head.weight.data_ptr() == table.weight.data_ptr():
            raise ValueError(f"Anima's eyes ({rec.get('name')}): the output head (head {rec.get('head')!r}, built untied) is "
                             "tied to the word table after the load: this transformers ignored tie_word_embeddings false")
        return (f"Anima's eyes ({rec.get('name')}, head {rec.get('head')}): Anima's slot (language layers "
                f"{rec['first_base_layer']}-27, the final norm) is the Base's (fingerprint {fp[:16]}...)")
    return None


def anima_eyes(model_or_config) -> dict | None:
    """The record of InternVL3.5-1B with Anima's top layers put back (anima_eyes.py), from the config's "anima_eyes" key; None for
    every other model (InternVL as it is included)."""
    cfg = getattr(model_or_config, "config", model_or_config)
    rec = getattr(cfg, "anima_eyes", None)
    return dict(rec) if isinstance(rec, dict) else None


def lora_targets(model_or_config) -> str | None:
    """The LoRA targets a family needs in place of the common pattern (train.TARGETS), or None. Anima's eyes: the language layers
    below Anima's slot only."""
    rec = anima_eyes(model_or_config)
    if rec is None:
        return None
    from .anima_eyes import lora_targets as targets
    return targets(int(rec["first_base_layer"]))


def frozen(model_or_config, name: str) -> bool:
    """Whether a parameter a full fine-tune would train stays frozen in its family: Anima's slot in Anima's eyes."""
    rec = anima_eyes(model_or_config)
    if rec is None:
        return False
    from .anima_eyes import is_slot
    return is_slot(name, int(rec["first_base_layer"]))


def mixed_batches(model) -> str | None:
    """Molmo2's build_batched_images fails its own check on a batch that mixes picture rows with text-only rows: for an example
    without a picture it builds a one-item offset list where the example has no picture to offset. Training batches mix them (the
    specification rows carry no picture), so the call runs on the picture rows and the text-only rows get the padding the function
    uses for missing crops and patches (-1), which the vision backbone masks out."""
    inner = next((m for m in model.modules() if hasattr(type(m), "build_batched_images")), None)
    if inner is None or getattr(inner, "_mixed_batches", False):
        return None
    original, end_id = inner.build_batched_images, inner.config.image_end_token_id

    def build_batched_images(input_ids, pixel_values, image_token_pooling, image_grids, image_num_crops):
        has = (input_ids == end_id).sum(1) > 0
        if bool(has.all()):
            return original(input_ids=input_ids, pixel_values=pixel_values, image_token_pooling=image_token_pooling,
                            image_grids=image_grids, image_num_crops=image_num_crops)
        images, pooling = original(input_ids=input_ids[has], pixel_values=pixel_values, image_token_pooling=image_token_pooling,
                                   image_grids=image_grids, image_num_crops=image_num_crops)
        n = input_ids.shape[0]
        full_images = images.new_full((n, *images.shape[1:]), -1)
        full_pooling = pooling.new_full((n, *pooling.shape[1:]), -1)
        full_images[has] = images
        full_pooling[has] = pooling
        return full_images, full_pooling

    inner.build_batched_images = build_batched_images
    inner._mixed_batches = True
    return "batches that mix picture rows and text-only rows go through this family's own image batching per picture row"


def pictures_one_at_a_time(model) -> str | None:
    """Ministral 3's get_image_features runs the vision tower once over all of a batch's pictures, packed into one sequence whose
    dense block mask keeps each picture to itself. In a 64-picture batch on the card (bf16, about 250,000 patches) the pictures after
    the first few came out wrong, while one at a time they read right. The wrapped call runs the original once per picture and joins
    the outputs in order (the per-picture features, and the hidden states along the sequence, as the packed call lays them out);
    the batch stays whole for the language model. A call with one picture, or one that returns a tuple, goes through as it was."""
    inner = next((m for m in model.modules()
                  if "multi_modal_projector" in dict(m.named_children()) and hasattr(type(m), "get_image_features")), None)
    if inner is None or getattr(inner, "_pictures_one_at_a_time", False):
        return None
    original = inner.get_image_features

    def get_image_features(pixel_values, image_sizes, vision_feature_layer=None, **kwargs):
        n = 0 if pixel_values is None else int(pixel_values.shape[0])
        if n <= 1:
            return original(pixel_values=pixel_values, image_sizes=image_sizes, vision_feature_layer=vision_feature_layer, **kwargs)
        outs = [original(pixel_values=pixel_values[i:i + 1], image_sizes=image_sizes[i:i + 1],
                         vision_feature_layer=vision_feature_layer, **kwargs) for i in range(n)]
        if not all(hasattr(o, "pooler_output") for o in outs):
            return original(pixel_values=pixel_values, image_sizes=image_sizes, vision_feature_layer=vision_feature_layer, **kwargs)
        import torch
        out = outs[0]
        out.pooler_output = tuple(t for o in outs for t in o.pooler_output)
        if all(getattr(o, "last_hidden_state", None) is not None for o in outs):
            out.last_hidden_state = torch.cat([o.last_hidden_state for o in outs], dim=1)
        if all(getattr(o, "hidden_states", None) for o in outs):
            out.hidden_states = tuple(torch.cat(layer, dim=1) for layer in zip(*(o.hidden_states for o in outs)))
        return out

    inner.get_image_features = get_image_features
    inner._pictures_one_at_a_time = True
    return "the vision tower reads a batch's pictures one at a time (a packed batch of many pictures came out wrong on the card)"


def drop_input_grads(model) -> bool:
    """Remove the input-embedding grad hook (transformers' enable_input_require_grads) from a model that must train without it
    (input_grads False). Skipping our own call is not enough: PEFT installs the same hook when it wraps a model whose layers already
    checkpoint. Returns whether a hook was removed."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    hook = getattr(base, "_require_grads_hook", None)
    if hook is None:
        return False
    hook.remove()
    del base._require_grads_hook
    return True


def fix_padding(model_or_config, batch, pad_id) -> None:
    """A right-padded training batch whose processor marked the padding as attended (Molmo2): the mask is rebuilt from the pad
    id, which no text token takes."""
    if model_type(model_or_config) == MOLMO2 and pad_id is not None:
        batch["attention_mask"] = (batch["input_ids"] != pad_id).to(batch["attention_mask"].dtype)


def prepare_processor(processor, model_or_config) -> None:
    """A processor that renders the family's chat template and batches its pictures: Phi-4's processor has no chat template
    of its own (its tokenizer's renders); Ministral 3's image processor is told to pad (its config leaves do_pad null);
    InternVL's processor pads on its tokenizer's side (follow_padding_side)."""
    kind = model_type(model_or_config)
    if kind == PHI4:
        processor.apply_chat_template = processor.tokenizer.apply_chat_template
    images = getattr(processor, "image_processor", None)
    if kind == MISTRAL3 and images is not None and getattr(images, "do_pad", True) is None:
        images.do_pad = True
    if kind == INTERNVL:
        follow_padding_side(processor)


def follow_padding_side(processor) -> bool:
    """InternVL's processor pads on the left by its own default, whatever its tokenizer's padding_side says (transformers 5.22:
    setting tokenizer.padding_side = "right" still gave left-padded batches). The training collator masks each prompt counting
    from the left edge, so a left-padded short row would be taught its own prompt. The processor is made to pad, at each call,
    on the side its tokenizer names then (train.run sets right, the generation loaders left, the keep read switches between).
    Returns whether it was changed (False: already following)."""
    cls = type(processor)
    if getattr(cls, "_follows_padding_side", False):
        return False
    original = cls.__call__

    def __call__(self, *args, **kwargs):
        if kwargs.get("padding") and "padding_side" not in kwargs:
            kwargs["padding_side"] = self.tokenizer.padding_side
        return original(self, *args, **kwargs)

    processor.__class__ = type(cls.__name__, (cls,), {"__call__": __call__, "_follows_padding_side": True})
    return True


def apply_image_settings(processor, settings: dict | None) -> dict:
    """Settings a spec gives the processor's image processor (attributes it already has), returned as applied: a picture budget,
    e.g. {"max_num_patches": 1024} for Phi-4, whose pictures otherwise expand to up to 3,600 tokens. An unknown name is an error."""
    if not settings:
        return {}
    images = getattr(processor, "image_processor", None)
    for key, value in settings.items():
        assert images is not None and hasattr(images, key), f"the image processor has no setting {key!r}"
        setattr(images, key, value)
    return dict(settings)


def thinking_kwargs(processor, thinking: bool) -> dict:
    """The thinking switch for apply_chat_template when the template reads one (enable_thinking: Qwen3.5, Gemma 4, GLM-4V);
    nothing when the template text is readable and never names it (Ministral, North, Molmo2), so transformers does not warn on
    every row that an unused keyword reached the processor."""
    for owner in (processor, getattr(processor, "tokenizer", None)):
        template = getattr(owner, "chat_template", None)
        if template:
            text = template if isinstance(template, str) else " ".join(str(v) for v in dict(template).values())
            return {"enable_thinking": thinking} if "enable_thinking" in text else {}
    return {"enable_thinking": thinking}


def prepare_inference(model) -> None:
    """Settings a family needs for batched generation: Phi-4 re-pads its multimodal sequence on the side its config names."""
    if model_type(model) == PHI4:
        model.config.tokenizer_padding_side = "left"


def fill_images(model_or_config, images: list) -> list:
    """The pictures of a batch, one slot per row: for Phi-4, when any row has a picture, each row without one gets a blank
    picture, the slot its model takes for that row (the picture's features are not used: the row has no <image> marker)."""
    if model_type(model_or_config) != PHI4 or all(im is None for im in images):
        return images
    from PIL import Image
    return [im if im is not None else Image.new("RGB", (32, 32), (127, 127, 127)) for im in images]


def generation_inputs(model, inputs) -> tuple:
    """The keyword arguments of generate() and the number of prompt positions at the head of its output. For Phi-4 with
    pictures, the multimodal embeddings are built here through the model's own routine (left-padded, see prepare_inference)
    and generation runs from them, so the output holds only the new tokens."""
    if model_type(model) != PHI4 or "pixel_values" not in inputs:
        return dict(inputs), int(inputs["input_ids"].shape[1])
    from transformers import BatchFeature
    images = BatchFeature({k: inputs[k] for k in ("pixel_values", "pixel_attention_mask", "spatial_shapes") if k in inputs})
    _, _, mask, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(inputs["input_ids"], None, inputs["attention_mask"],
                                                                         None, None, images)
    return {"inputs_embeds": embeds, "attention_mask": mask}, 0


def clean(text: str) -> str:
    """A generated answer without a leading mode tag (Phi-4-reasoning-vision opens with <nothink> when it does not think)."""
    return _MODE_TAG.sub("", text)
