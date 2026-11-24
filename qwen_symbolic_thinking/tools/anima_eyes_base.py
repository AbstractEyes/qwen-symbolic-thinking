"""The first run of an Anima's-eyes lane: build the hybrid (anima_eyes.build: InternVL3.5-1B with Anima's own top layers put
back, from the two pinned Hub sources, fingerprint-checked), load it once the way every later run will (transformers' own
InternVL class; the slot's fingerprint checked at the load), read one picture with it as a sanity check, and ship the folder to
the project repo under base/<name>/, where every later run loads it as the model id hub:base/<name>.

    from qwen_symbolic_thinking.tools.anima_eyes_base import run
    run(head="internvl", tag="base_ae1b")              (a queue tool spec: {"module": "anima_eyes_base", "args": {...}})

A folder already shipped with the same fingerprint is not shipped again; one with another fingerprint stops the run (the
lane's later runs would load a different model than the one their names promise)."""
from __future__ import annotations

import json
import os
import time

from .. import REPO, __version__, anima_eyes
from ..hub import ship, token


def shipped_record(repo: str, folder: str) -> dict | None:
    """The record of a hybrid already in the project repo (its anima_eyes.json), or None."""
    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(repo, f"{folder}/anima_eyes.json", repo_type="model", token=token())
    except Exception:
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def sanity_read(path: str, device: str | None = None) -> dict:
    """The hybrid loaded with transformers' own class (no repository code; the family's slot check runs in prepare_model) and
    asked to describe one plain picture (a red square on white: it should say red, square); greedy, 40 new tokens."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor

    from .. import family
    from ..screen import load_vlm
    from ..template import stop_ids
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(path)
    model = load_vlm(path, dtype=torch.bfloat16 if device == "cuda" else torch.float32, device_map=device)
    note = family.prepare_model(model)
    model.eval()
    im = Image.new("RGB", (448, 448), (255, 255, 255))
    im.paste((220, 20, 20), (112, 112, 336, 336))
    conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this picture in one sentence."}]}]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    enc = processor(text=[text], images=[im], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=40, do_sample=False, eos_token_id=stop_ids(processor, model),
                             pad_token_id=processor.tokenizer.pad_token_id)
    answer = processor.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"note": note, "answer": answer, "says_red": "red" in answer.lower()}


def run(head: str | None = None, out_dir: str = "/content/anima_eyes", folder: str | None = None, tag: str = "",
        repo: str = REPO, do_ship: bool = True, read: bool = True) -> dict:
    head = head or anima_eyes.HEAD
    folder = folder or anima_eyes.FOLDER
    t0 = time.time()
    path = anima_eyes.build(out_dir, head)
    with open(os.path.join(path, "anima_eyes.json"), encoding="utf-8") as fh:
        rec = json.load(fh)
    summary = {"package": __version__, "tag": tag, "folder": folder, "model_id": f"hub:{folder}", "record": rec,
               "build_s": round(time.time() - t0, 1)}
    if read:
        summary["sanity_read"] = sanity_read(path)
        print(f"  the hybrid read a red square: {summary['sanity_read']['answer']!r}", flush=True)
    if do_ship:
        there = shipped_record(repo, folder)
        if there is not None:
            assert there.get("fingerprint") == rec["fingerprint"], (
                f"{repo}/{folder} holds another hybrid (fingerprint {there.get('fingerprint')}, not {rec['fingerprint']})")
            summary["shipped"] = "already there, the same fingerprint"
        else:
            url = ship(path, folder, f"{anima_eyes.NAME}: InternVL3.5-1B + Anima's top layers (head {head}), "
                                     f"fingerprint {rec['fingerprint'][:16]}", repo)
            assert url is not None, f"the hybrid did not ship to {repo}/{folder}; its folder stays at {path}"
            summary["shipped"] = url
        print(f"  {folder}: {summary['shipped']}", flush=True)
    summary["seconds"] = round(time.time() - t0, 1)
    return summary
