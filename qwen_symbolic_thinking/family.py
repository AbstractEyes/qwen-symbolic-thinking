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
  default (the model crops each picture back by image_sizes).

The repository code of these families runs only when a spec asks for it (trust_remote_code), loaded from a local snapshot at a
pinned revision (the Phi-4 loader reads its own vision weights only from a local folder), on the transformers release it was
built on (4.56-4.57: Phi-4's image processor subclasses an image-processor module that transformers 5 rewrote), on a lane whose
queue carries only such runs (queue.transformers_fits)."""
import re

PHI4 = "phi4-siglip"
MOLMO2 = "molmo2"
MISTRAL3 = "mistral3"
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


def fix_padding(model_or_config, batch, pad_id) -> None:
    """A right-padded training batch whose processor marked the padding as attended (Molmo2): the mask is rebuilt from the pad
    id, which no text token takes."""
    if model_type(model_or_config) == MOLMO2 and pad_id is not None:
        batch["attention_mask"] = (batch["input_ids"] != pad_id).to(batch["attention_mask"].dtype)


def prepare_processor(processor, model_or_config) -> None:
    """A processor that renders the family's chat template and batches its pictures: Phi-4's processor has no chat template
    of its own (its tokenizer's renders); Ministral 3's image processor is told to pad (its config leaves do_pad null)."""
    kind = model_type(model_or_config)
    if kind == PHI4:
        processor.apply_chat_template = processor.tokenizer.apply_chat_template
    images = getattr(processor, "image_processor", None)
    if kind == MISTRAL3 and images is not None and getattr(images, "do_pad", True) is None:
        images.do_pad = True


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
