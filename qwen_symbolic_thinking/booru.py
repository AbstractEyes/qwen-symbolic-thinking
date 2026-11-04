"""Booru taggers, run the way their authors run them, one picture or a batch at a time.

Two families:
- animetimm (deepghs, Danbooru v4 tags): the image on a white background, the test pipeline of the repository's preprocess.json
  (fit inside a square and pad it white, resize, center crop, normalize), a sigmoid, then each tag's own best threshold from
  selected_tags.csv (the reference: deepghs/imgutils, imgutils/generic/multilabel_timm.py).
- WD v3 (SmilingWolf): the image on a white background padded to a white square, the model's timm transform, the channels in BGR
  order, a sigmoid, then 0.35 for general tags and 0.75 for character tags (the reference: neggles/wdv3-timm).

Rating tags are dropped. Tags come out most confident first as (name, score, category); tag_name turns underscores into spaces
(kaomoji kept as they are). keep() is the filter a reviewer may set on top: a list of tags to leave out and a floor on each tag's
own expected precision (animetimm publishes it per tag), clothing and jewelry exempt from the floor."""
import csv
import json
import os
import re
import sys

KINDS = {"animetimm/": "animetimm", "SmilingWolf/": "wd"}
FILES = ["config.json", "model.safetensors", "selected_tags.csv", "thresholds.csv", "preprocess.json"]   # WD repos: the first three
WD_THRESHOLDS = {0: 0.35, 4: 0.75}           # general, character (the reference's defaults)
KEEP = (0, 4)                                 # general and character tags; 9 = rating, dropped
MAX_TAGS = 40
CLOTHES = re.compile(r"\b(shirt|t-shirt|top|blouse|sweater|hoodie|cardigan|jacket|coat|blazer|vest|dress|skirt|shorts|pants|jeans|"
                     r"leggings|trousers|footwear|shoes|sneakers|boots|sandals|heels|loafers|slippers|socks|hat|cap|beanie|scarf|"
                     r"gloves|belt|bag|jumpsuit|overalls|swimsuit|uniform|kimono|robe|gown|apron|necklace|bracelet|ring|earrings)\b")
US = {"grey": "gray", "colour": "color", "coloured": "colored", "colourful": "colorful", "centre": "center", "jewellery": "jewelry"}


def kind_of(repo: str) -> str:
    return next((k for prefix, k in KINDS.items() if repo.startswith(prefix)), "animetimm")


def tag_name(name: str) -> str:
    """A tag as text: underscores to spaces, except in kaomoji (no letters in them, such as ^_^)."""
    return name.replace("_", " ") if any(ch.isalpha() for ch in name) and len(name) > 3 else name


def us(text: str) -> str:
    """US spelling, for matching a leave-out list written either way."""
    return re.sub(r"\b(" + "|".join(US) + r")\b", lambda m: US[m.group(1)], text)


def keep(name: str, precision, omit=(), floor: float = 0.0, keep_clothes: bool = True) -> bool:
    """A tag kept: not on the leave-out list, and its own expected precision at least the floor (clothing and jewelry exempt when
    keep_clothes; a tag without a published precision is never dropped by the floor)."""
    t = us(tag_name(name).lower())
    if t in {us(str(x).strip().lower()) for x in omit}:
        return False
    if not floor or precision is None or (keep_clothes and CLOTHES.search(t)):
        return True
    return precision >= floor


def fetch(repo: str, cache_dir: str, token: str | None = None) -> str:
    """A tagger's files downloaded into cache_dir/<repo>/ (the animetimm repositories are gated: the token needs access)."""
    from huggingface_hub import hf_hub_download
    folder = os.path.join(cache_dir, *repo.split("/"))
    for name in FILES if kind_of(repo) == "animetimm" else FILES[:3]:
        if not os.path.isfile(os.path.join(folder, name)):
            hf_hub_download(repo, name, local_dir=folder, token=token)
    return folder


def _read_csv(path: str) -> list:
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _pad_to_size(image, size, background, interpolation):
    """Fit the picture inside size keeping its shape, then pad it to size, centred (imgutils.data.pad.pad_image_to_size)."""
    from PIL import Image
    tw, th = size
    ratio = min(tw / image.width, th / image.height)
    nw, nh = round(image.width * ratio), round(image.height * ratio)
    resized = image.resize((nw, nh), interpolation)
    canvas = Image.new("RGB", (tw, th), background)
    canvas.paste(resized, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def _animetimm_transform(steps: list):
    """The test pipeline of an animetimm preprocess.json as a callable, PIL image in, tensor out."""
    from PIL import Image
    from torchvision import transforms as T
    pil = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "nearest": Image.NEAREST, "lanczos": Image.LANCZOS}
    tv = {"bilinear": T.InterpolationMode.BILINEAR, "bicubic": T.InterpolationMode.BICUBIC, "nearest": T.InterpolationMode.NEAREST}
    ops = []
    for s in steps:
        kind = s["type"]
        if kind == "pad_to_size":
            size = tuple(s["size"]) if isinstance(s["size"], (list, tuple)) else (s["size"], s["size"])
            ops.append(lambda im, size=size, s=s: _pad_to_size(im, size, s.get("background_color", "white"),
                                                               pil[s.get("interpolation", "bilinear")]))
        elif kind == "resize":
            size = s["size"]
            ops.append(T.Resize(tuple(size) if isinstance(size, (list, tuple)) else size,
                                interpolation=tv[s.get("interpolation", "bilinear")], max_size=s.get("max_size"),
                                antialias=s.get("antialias", True)))
        elif kind == "center_crop":
            ops.append(T.CenterCrop(tuple(s["size"]) if isinstance(s["size"], (list, tuple)) else s["size"]))
        elif kind in ("maybe_to_tensor", "to_tensor"):
            ops.append(T.ToTensor())
        elif kind == "normalize":
            ops.append(T.Normalize(mean=s["mean"], std=s["std"]))
        else:
            raise ValueError(f"unknown preprocessing step {kind!r}")
    return T.Compose(ops)


def white_rgb(image):
    """The picture in RGB on a white background (a transparent picture's see-through parts turn white)."""
    from PIL import Image
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
    if image.mode == "RGBA":
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
    return image


def load(folder: str, kind: str = "animetimm", device: str = "cuda") -> dict:
    """A tagger from its downloaded folder: the timm model in float32 on device (its thresholds were set on float32 scores),
    its transform, its tag table, and each tag's published precision (animetimm)."""
    import timm
    import torch
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(folder, "config.json"), encoding="utf-8"))
    model = timm.create_model(cfg["architecture"], pretrained=False, num_classes=cfg["num_classes"], **(cfg.get("model_args") or {}))
    model.load_state_dict(load_file(os.path.join(folder, "model.safetensors")))
    model = model.to(device=device, dtype=torch.float32).eval()
    rows = _read_csv(os.path.join(folder, "selected_tags.csv"))
    names = [r["name"] for r in rows]
    cats = [int(r["category"]) for r in rows]
    assert len(names) == cfg["num_classes"], (len(names), cfg["num_classes"])
    if kind == "animetimm":
        transform = _animetimm_transform(json.load(open(os.path.join(folder, "preprocess.json"), encoding="utf-8"))["test"])
        cat_thr = {int(r["category"]): float(r["threshold"]) for r in _read_csv(os.path.join(folder, "thresholds.csv"))}
        thresholds = [float(r["best_threshold"]) if r.get("best_threshold") not in (None, "") else cat_thr.get(c, 0.4)
                      for r, c in zip(rows, cats)]
    else:
        from timm.data import create_transform, resolve_data_config
        transform = create_transform(**resolve_data_config(pretrained_cfg=cfg["pretrained_cfg"]))
        thresholds = [WD_THRESHOLDS.get(c, 1.0) for c in cats]
    precision = {r["name"]: float(r["best_precision"]) for r in rows if r.get("best_precision") not in (None, "")}
    return {"model": model, "transform": transform, "names": names, "cats": cats, "thresholds": thresholds, "kind": kind,
            "device": device, "precision": precision}


def _tensor(bundle: dict, image):
    image = white_rgb(image)
    if bundle["kind"] == "wd":
        from PIL import Image
        side = max(image.size)
        square = Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        return bundle["transform"](square)[[2, 1, 0]]          # RGB to BGR, as the WD taggers were trained
    return bundle["transform"](image)


def _pick(bundle: dict, probs: list, threshold: float | None, max_tags: int) -> list:
    out = [(name, p, c) for name, p, c, t in zip(bundle["names"], probs, bundle["cats"], bundle["thresholds"])
           if c in KEEP and p >= (t if threshold is None else threshold)]
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:max_tags]


def predict_batch(bundle: dict, images: list, threshold: float | None = None, max_tags: int = MAX_TAGS) -> list:
    """[(tag, score, category)] for each picture, most confident first: general and character tags at or above their thresholds
    (threshold, when given, for every tag instead), at most max_tags each."""
    import torch
    x = torch.stack([_tensor(bundle, im) for im in images])
    with torch.inference_mode():
        probs = torch.sigmoid(bundle["model"](x.to(bundle["device"], torch.float32)).float()).cpu().tolist()
    return [_pick(bundle, p, threshold, max_tags) for p in probs]


def predict(bundle: dict, image, threshold: float | None = None, max_tags: int = MAX_TAGS) -> list:
    return predict_batch(bundle, [image], threshold, max_tags)[0]
