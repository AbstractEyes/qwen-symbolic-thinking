"""The model registry: one entry per model of the version-1 list, holding what a v1.2 lane needs (the base, its pins, the recipe,
the picture budget, the smoke's limits), what its card and release need (the repos, the tags, the older versions) and its
looker's settings in the testbed Space. `lines.py` builds a lane (its queue and its notebook) from an entry.

    from qwen_symbolic_thinking import registry
    entry = registry.get("2b")               # by its run key, its base id, a release repo or a Space looker key
    registry.recipe(entry)                   # the train keys of its v1.2 training
    registry.smoke_stop(entry, "apache")     # the limits its smoke stops at
    registry.todo()                          # every value the records do not give yet, and what decides it

    python -m qwen_symbolic_thinking.registry [<key> ...]      (prints the entries and every open value)

Where each value comes from is written beside it: a record (a run's marker in the record repo, a release tag, a model's
config), a measurement (a v1.2 run's own numbers), or an estimate (the v1.2 / v1 step ratio times the model's own v1 pace,
labeled as such until its smoke measures it). A value no record gives is None, and the entry's `todo` names what decides it;
nothing is filled in by guess. `blockers` lists what stops a model's lane from running as generated.

An entry's fields:

| field | what it holds |
|---|---|
| key | the short key in its run names (V12_A1_gkl1_1ep_<key>_by, S_v12_smoke_gc1_<key>, v11a_<key>, ...) |
| name, line_name, family, model_type | its name on the cards, its named line (Lucia, Ulisse) or None, the model family, the config's model_type |
| base, base_revision | the base model's Hub id, and the revision its repository code is pinned to (None: not pinned) |
| trust_remote_code, transformers, install | whether its repository's code runs; the transformers release its specs ask for (None: 5 or later, the queue's default); the notebook's install line and whether it builds the linear-attention kernels |
| lines, scheduled | the v1.2 lines the catalogue gives it ("apache", "nc"); whether it is trained (False: training instructions only) |
| lane | its lane: the queue runs/queue_<lane>.json and the notebook <lane>.ipynb |
| lora, lr, batch, grad_accum, max_len, image_settings | its training settings (the rest of the recipe is V12_RECIPE, the same for every model) |
| read_batch, keep_batch | the batch of its extraction reads and general answers; of its preservation reads |
| general_answers | the project-repo folders of its base model's own answers (anchors/<name>/anchors.json, held.json) |
| template | what the package does differently for its family (read from the model; noted here for people) |
| pace, peak | per line: the seconds a training step takes and the peak GPU memory, measured or estimated, with the source |
| repos, tags, licenses | its public release repos per line (and the full fine-tune's), their tags, each line's license |
| versions | its trained versions: where each adapter lives (the record repo, or PROJECT: the project repo the lane runs on) |
| comparisons | the reads of the base and of older versions its v1.2 card compares against (None: the generator's default set); "earlier": the reads of the same that earlier lanes made, by kind, under the names they ran with (the reads command reads them; the generator does not make them again) |
| space | its looker in the testbed Space: the looker keys, gb (bf16 weights on the card), tps (new tokens a second) |
| todo, blockers | what is open and what decides it; what keeps its lane from running |
"""
from __future__ import annotations

import copy
import json

from . import anima_eyes

OWNER = "AbstractPhil"
# The public record repo: every version-1 adapter lives here (lora/<model>/r32/<stamp>_<run>/final), with its run's marker.
RECORD = "AbstractPhil/qwen-symbolic-thinking-image-training"
# An adapter that lives in the lane's own project repo, the repo its training ran in (the project repo a lane is generated for).
PROJECT = "project"
LINES = ("apache", "nc")

# The v1.2 recipe, the same for every model (the v1.2 training specs of the 4B, 9B and 2B lanes, key for key): one pass over the
# rows at 8 x 2 sequences a step, the keeping objective at weight 1 on the base model's own answers, gradient checkpointing, the
# training state saved for a resume, a ship every 1,000 steps, the pictures held on disk. The per-model settings (LoRA rank and
# alpha, learning rate, batch, max length, picture budget) are the entries' own.
V12_RECIPE = {"epochs": 1, "seed": 20261030, "seed_adapter": True, "anchor_batch": 4, "general_kl": 1.0,
              "grad_checkpointing": True, "resume": True, "ship_every": 1000, "lazy_pictures": True, "picture_workers": 64}
# The writers whose text the release's rows carry (the dataset card's "Rows by source and writer"), in the order the v1.2 lanes
# list them; the base model that answered the general rows is appended per model.
V12_WRITERS = ("gpt-6-luna", "deepseek-flash", "gemini-4-argon", "gemini-3.8-flash")
# The two training sets of the v1.2 line (release 5 of the public dataset, split by license), as paths in the project repo; the
# same rows for every model (pictures that fail to load or examples cut at max_len can move a model's counts slightly).
V12_SETS = {
    "apache": {"path": "sft/v12_rows_a_by.json", "items": 91_138, "examples": 162_183, "steps": 10_137},
    "nc": {"path": "sft/v12_rows_nc.json", "items": 184_562, "examples": 349_031, "steps": 21_815},
}
# The smoke that runs before each training: the set's first 2,000 items, 30 steps, nothing shipped, no screen, and the stop rule
# of the v1.2 lanes: its steady pace (the steps after the first ten) above 1.25 x the measured v1.2 pace, or above 1.5 x the
# estimate where no v1.2 run has measured one (an estimate carries more doubt than a measurement), or its peak memory above 85 GB,
# fails the smoke and holds its training back.
SMOKE = {"n_items": 2000, "max_steps": 30, "resume": False, "do_ship": False}
STOP_FACTOR, STOP_FACTOR_ESTIMATE, STOP_PEAK_GB, STOP_AFTER_STEP = 1.25, 1.5, 85, 10
KEEP_BATCH = 32          # the preservation reads' batch in the v1.2 lanes (tools.keep_read)

# The bars' yardstick (tools.bars; `python -m qwen_symbolic_thinking.reads <key> <line> bars`): per line, the plain arm of
# version 1.1 at 4B, trained on that version's rows of the line without the keeping objective; every v1.2 line's bars are read
# against its line's arm (across sizes, as registered). Its values as the bars print them (the yardstick column of every
# v1.2 bars table): the screen's hint said and backed, the probe, the drift per answer token at strength 1, the row form's
# excess over the base.
YARDSTICK = {
    "apache": {"run": "V11_A0_rep_2ep_4b", "tag": "v11_A0",
               "values": {"said": 0.209, "backed": 0.225, "probe": 143, "kl_picture": 1.597, "kl_text": 0.593, "kl_all": 1.154,
                          "form_excess": 0.949}},
    "nc": {"run": "V11NC_A0_rep_2ep_4b_r2", "tag": "v11nc_A0_r2",
           "values": {"said": 0.241, "backed": 0.244, "probe": 134, "kl_picture": 1.762, "kl_text": 0.714, "kl_all": 1.291,
                      "form_excess": 0.976}},
}
# The bars our v1.2 lines read (tools.bars, each line's training against its line's yardstick), as the tables print them,
# with the day they were read. TRAINING.md shows them as the numbers to compare against.
BARS = {
    ("4b", "apache"): {"run": "V12_A1_gkl1_1ep_4b_by", "read": "2026-11-20", "format": 1.0, "full_rows": 1.0, "said": 0.210,
                       "backed": 0.264, "probe": 150, "kl_picture": 0.071, "kl_text": 0.082, "kl_all": 0.084, "form": 0.0,
                       "ended": 0.904, "base_ended": 0.907, "pass": True},
    ("4b", "nc"): {"run": "V12NC_A1_gkl1_1ep_4b", "read": "2026-11-19", "format": 1.0, "full_rows": 1.0, "said": 0.239,
                   "backed": 0.244, "probe": 160, "kl_picture": 0.096, "kl_text": 0.124, "kl_all": 0.119, "form": 0.0,
                   "ended": 0.922, "base_ended": 0.910, "pass": True},
    ("2b", "apache"): {"run": "V12_A1_gkl1_1ep_2b_by", "read": "2026-11-20", "format": 1.0, "full_rows": 1.0, "said": 0.217,
                       "backed": 0.220, "probe": 154, "kl_picture": 0.066, "kl_text": 0.069, "kl_all": 0.085, "form": 0.0,
                       "ended": 0.883, "base_ended": 0.887, "pass": True},
    ("2b", "nc"): {"run": "V12NC_A1_gkl1_1ep_2b", "read": "2026-11-21", "format": 1.0, "full_rows": 1.0, "said": 0.231,
                   "backed": 0.246, "probe": 147, "kl_picture": 0.079, "kl_text": 0.102, "kl_all": 0.124, "form": 0.0,
                   "ended": 0.874, "base_ended": 0.887, "pass": True},
    ("9b", "apache"): {"run": "V12_A1_gkl1_1ep_9b_by", "read": "2026-11-21", "format": 1.0, "full_rows": 1.0, "said": 0.194,
                       "backed": 0.258, "probe": 165, "kl_picture": 0.054, "kl_text": 0.062, "kl_all": 0.066, "form": 0.0,
                       "ended": 0.930, "base_ended": 0.918, "pass": True},
    ("9b", "nc"): {"run": "V12NC_A1_gkl1_1ep_9b", "read": "2026-11-23", "format": 1.0, "full_rows": 1.0, "said": 0.234,
                   "backed": 0.244, "probe": 172, "kl_picture": 0.081, "kl_text": 0.104, "kl_all": 0.103, "form": 0.0,
                   "ended": 0.918, "base_ended": 0.918, "pass": True},
    ("08b", "apache"): {"run": "V12_A1_gkl1_1ep_08b_by", "read": "2026-11-21", "format": 1.0, "full_rows": 1.0, "said": 0.208,
                        "backed": 0.239, "probe": 136, "kl_picture": 0.077, "kl_text": 0.073, "kl_all": 0.105, "form": 0.0,
                        "ended": 0.887, "base_ended": 0.905, "pass": True},
    # K3 ended missed (0.863 against the line 0.885): released with that caveat
    ("08b", "nc"): {"run": "V12NC_A1_gkl1_1ep_08b", "read": "2026-11-22", "format": 1.0, "full_rows": 1.0, "said": 0.220,
                    "backed": 0.237, "probe": 155, "kl_picture": 0.077, "kl_text": 0.094, "kl_all": 0.105, "form": 0.0,
                    "ended": 0.863, "base_ended": 0.905, "pass": False},
    # read with its keep read on the held set with text requests (K_v12_A1_ge4b_by_t, the held text requests asked apart from
    # text/requests_qwen_v1.json): the registered K_v12_A1_ge4b_by holds no text request ("keep": reads --keep names it)
    ("ge4b", "apache"): {"run": "V12_A1_gkl1_1ep_ge4b_by", "read": "2026-11-22", "format": 1.0, "full_rows": 1.0, "said": 0.231,
                         "backed": 0.202, "probe": 145, "kl_picture": 0.063, "kl_text": 0.061, "kl_all": 0.072, "form": 0.0,
                         "ended": 0.933, "base_ended": 0.908, "pass": True, "keep": "K_v12_A1_ge4b_by_t"},
    ("ge4b", "nc"): {"run": "V12NC_A1_gkl1_1ep_ge4b", "read": "2026-11-23", "format": 1.0, "full_rows": 1.0, "said": 0.219,
                     "backed": 0.239, "probe": 136, "kl_picture": 0.094, "kl_text": 0.060, "kl_all": 0.098, "form": 0.0,
                     "ended": 0.910, "base_ended": 0.908, "pass": True},
    ("g12b", "apache"): {"run": "V12_A1_gkl1_1ep_g12b_by", "read": "2026-11-25", "format": 1.0, "full_rows": 1.0, "said": 0.251,
                         "backed": 0.213, "probe": 159, "kl_picture": 0.048, "kl_text": 0.009, "kl_all": 0.034, "form": 0.0,
                         "ended": 0.842, "base_ended": 0.855, "pass": True},
    ("g12b", "nc"): {"run": "V12NC_A1_gkl1_1ep_g12b", "read": "2026-11-26", "format": 1.0, "full_rows": 1.0, "said": 0.229,
                     "backed": 0.253, "probe": 163, "kl_picture": 0.064, "kl_text": 0.018, "kl_all": 0.046, "form": 0.0,
                     "ended": 0.850, "base_ended": 0.855, "pass": True},
    ("north", "apache"): {"run": "V12_A1_gkl1_1ep_north_by", "read": "2026-11-26", "format": 1.0, "full_rows": 1.0, "said": 0.194,
                          "backed": 0.252, "probe": 149, "kl_picture": 0.028, "kl_text": 0.060, "kl_all": 0.044, "form": 0.0,
                          "ended": 1.000, "base_ended": 0.996, "pass": True},
    ("glm46v", "apache"): {"run": "V12_A1_gkl1_1ep_glm46v_by", "read": "2026-11-27", "format": 1.0, "full_rows": 1.0,
                           "said": 0.219, "backed": 0.251, "probe": 160, "kl_picture": 0.053, "kl_text": 0.049, "kl_all": 0.073,
                           "form": 0.0, "ended": 0.966, "base_ended": 0.953, "pass": True},
    # Anima's eyes: no bars were read for the release (the full fine-tune, released 2026-11-25 on a side-by-side look at Anima's
    # pictures in ComfyUI); read 2026-11-27 with this package's bar code (tools.bars on the project repo's records).
    # The full fine-tune misses T2 backed by 0.001 (the band's floor 0.195) and T3 probe (118, the floor 133); the LoRA beside
    # it, never released, wrote its rows and never ended them (T1 full rows 0.017)
    ("ae1bf", "apache"): {"run": "V12_A1_gkl1_1ep_ae1bf_by", "read": "2026-11-27", "format": 1.0, "full_rows": 1.0,
                          "said": 0.225, "backed": 0.194, "probe": 118, "kl_picture": 0.068, "kl_text": 0.057, "kl_all": 0.096,
                          "form": 0.0, "ended": 0.957, "base_ended": 0.961, "pass": False,
                          "note": "our release, whose bars were read after it: it was released on a side-by-side look at "
                                  "Anima's pictures in ComfyUI, with it in Anima's text-encoder slot against Anima's own encoder"},
    ("ae1b", "apache"): {"run": "V12_A1_gkl1_1ep_ae1b_by", "read": "2026-11-27", "format": 1.0, "full_rows": 0.017,
                         "said": 0.231, "backed": 0.186, "probe": 125, "kl_picture": 0.049, "kl_text": 0.046, "kl_all": 0.069,
                         "form": 0.0, "ended": 0.944, "base_ended": 0.961, "pass": False,
                         "note": "not released: it wrote each row and never ended it, so almost no row is whole"},
}
# The general answers' time on a G4, measured where a lane made them (the first start to the last end of its three runs).
GENERAL_ANSWERS_MINUTES = {
    "2b": {"minutes": 30.4, "source": "its lane's v11a_2b 24 min 51 s, v11a_2b_by 3 s, v11txt_2b 5 min 25 s, 2026-11-20"},
    "08b": {"minutes": 26.3, "source": "her lane's v11a_08b 21 min 40 s, v11a_08b_by 3 s, v11txt_08b 4 min 33 s, 2026-11-21"},
    "ge4b": {"minutes": 50.4, "source": "his lane's v11a_ge4b 40 min 43 s, v11a_ge4b_by 5 s, v11txt_ge4b 9 min 31 s, 2026-11-22"},
    "g12b": {"minutes": 103.2, "source": "his lane's v11a_g12b 77 min 36 s, v11a_g12b_by 5 s, v11txt_g12b 25 min 24 s, "
                                         "2026-11-24"},
    "north": {"minutes": 18.2, "source": "its lane's v11a_north 13 min 26 s, v11a_north_by 2 s, v11txt_north 4 min 43 s, "
                                         "2026-11-26"},
    "glm46v": {"minutes": 58.7, "source": "its lane's v11a_glm46v 43 min 45 s, v11a_glm46v_by 2 s, v11txt_glm46v 14 min 52 s, "
                                          "2026-11-26"},
    "ae1b": {"minutes": 139.1, "source": "its lane's v11a_ae1b 97 min 46 s, v11a_ae1b_by 2 s, v11txt_ae1b 41 min 17 s, "
                                         "2026-11-25"},
}

# An estimate: a v1.2 step costs about 1.5 times a v1 step of the same model on the Apache rows and about 1.4 times on the -nc
# rows (the measured v1.2 paces of the 9B, 4B and 2B against their own v1 paces), times the model's own measured v1 pace.
_EST = "estimate: the v1.2 / v1 step ratio (about 1.5 Apache, 1.4 -nc) x its v1 pace {v1} s ({run}); its smoke measures it"

_QWEN_TEMPLATE = ("Qwen3.5: thinking off through the chat template's enable_thinking switch; the assistant turn ends <|im_end|>; "
                  "the config's end id is <|endoftext|>, so generation also stops on the template's turn end (template.stop_ids); "
                  "the linear-attention kernels (causal-conv1d, flash-linear-attention) are optional and all or nothing")
_GEMMA_TEMPLATE = ("Gemma 4: the same enable_thinking switch (off: an empty thought block opens the answer); the assistant turn "
                   "ends <turn|>; the processor takes one list of pictures per text (screen.encode tries the flat form first); a "
                   "thought lives in a channel of special tokens, mapped for the parse")


def _version(version: str, line: str, adapter: str, subfolder: str | None, release: tuple | None, tag: str,
             keep_tag: str | None = None, wordings: str | None = None, source: str = "", ready: bool = True) -> dict:
    """One trained version: where its adapter lives (RECORD, PROJECT or a repo id) and in which folder; its release (repo, tag);
    the stem of its comparison reads' names (XT_<tag>, XT_<tag>_deep, XTA_<tag>, ...; K0_<keep_tag> for its keep read); the
    wordings its rows carry (None: version 1's, the default)."""
    return {"version": version, "line": line, "adapter": adapter, "subfolder": subfolder,
            "release": {"repo": release[0], "tag": release[1]} if release else None, "tag": tag, "keep_tag": keep_tag,
            "wordings": wordings, "ready": ready, "source": source}


def _pace(s: float, measured: bool, source: str) -> dict:
    return {"s": s, "measured": measured, "source": source}


ENTRIES = {
    # ------------------------------------------------------------------------------------------------ Lucia (Qwen3.5)
    "9b": {
        "key": "9b", "name": "Lucia", "line_name": "Lucia", "family": "Qwen3.5", "model_type": "qwen3_5",
        "base": "Qwen/Qwen3.5-9B", "base_revision": None,                 # not pinned (no lane pinned it)
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": True},
        "lines": ("apache", "nc"), "scheduled": True, "lane": "lucia_v12_9b",          # the lane both 9B lines ran in
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2, "max_len": 2048,   # V12_A1_gkl1_1ep_9b_by's spec
        "image_settings": None,
        "read_batch": 64, "keep_batch": KEEP_BATCH,                          # the 9B v1.2 lane's reads
        "general_answers": {"anchors": "anchors/v11a_9b", "anchors_by": "anchors/v11a_9b_by", "text": "anchors/v11txt_9b"},
        "template": _QWEN_TEMPLATE,
        "pace": {"apache": _pace(9.03, True, "measured: V12_A1_gkl1_1ep_9b_by, 10,137 steps in 1,525.9 min over two machines"),
                 "nc": _pace(8.14, True, "measured: V12NC_A1_gkl1_1ep_9b, 21,815 steps in 2,961.1 min over three machines "
                                         "(the second from step 8,340, the third from step 18,470)")},
        "peak": {"apache": {"gb": 58.7, "source": "measured: V12_A1_gkl1_1ep_9b_by (its smoke 51.3)"},
                 "nc": {"gb": 60.74, "source": "measured: V12NC_A1_gkl1_1ep_9b (its smoke 41.69)"}},
        "repos": {"apache": "AbstractPhil/qwen3.5-9b-think-symbolic-image", "nc": "AbstractPhil/qwen3.5-9b-think-symbolic-image-nc",
                  "full": None},                                   # no full fine-tune of the 9B (its states do not fit one card)
        "tags": {"AbstractPhil/qwen3.5-9b-think-symbolic-image": ["legacy-e-all-2ep", "lucia-v1", "lucia-v1.1", "lucia-v1.2"],
                 "AbstractPhil/qwen3.5-9b-think-symbolic-image-nc": ["lucia-v1.1", "lucia-v1.2"]},
        "licenses": {"apache": "Apache-2.0", "nc": "CC BY-NC-SA 4.0"},              # the release cards
        "versions": [
            _version("v1", "apache", RECORD, "lora/Qwen3.5-9B/r32/20261031_085915_G_med4_2ep/final",
                     ("AbstractPhil/qwen3.5-9b-think-symbolic-image", "lucia-v1"), "v1_9b_s1", "lucia1_9b",
                     source="G_med4_2ep's marker in the record repo"),
            _version("v1.1", "apache", PROJECT, "lora/Qwen3.5-9B/r32/20261112_121904_V11R_A1_gkl1_2ep_9b_by/final",
                     ("AbstractPhil/qwen3.5-9b-think-symbolic-image", "lucia-v1.1"), "v11_9b_by_s1", wordings="v1.1",
                     source="V11R_A1_gkl1_2ep_9b_by, trained in the project repo"),
            _version("v1.1", "nc", PROJECT, "lora/Qwen3.5-9B/r32/20261112_213632_V11RNC_A1_gkl1_2ep_9b/final",
                     ("AbstractPhil/qwen3.5-9b-think-symbolic-image-nc", "lucia-v1.1"), "v11nc_9b_s1", wordings="v1.1",
                     source="V11RNC_A1_gkl1_2ep_9b, trained in the project repo"),
            _version("v1.2", "apache", PROJECT, "lora/Qwen3.5-9B/r32/20261120_062038_V12_A1_gkl1_1ep_9b_by/final",
                     ("AbstractPhil/qwen3.5-9b-think-symbolic-image", "lucia-v1.2"), "v12_9b_by_s1", wordings="v1.1",
                     source="V12_A1_gkl1_1ep_9b_by"),
            _version("v1.2", "nc", PROJECT, "lora/Qwen3.5-9B/r32/20261121_100931_V12NC_A1_gkl1_1ep_9b/final",
                     ("AbstractPhil/qwen3.5-9b-think-symbolic-image-nc", "lucia-v1.2"), "v12_9b_nc_s1", wordings="v1.1",
                     source="V12NC_A1_gkl1_1ep_9b"),
        ],
        # the 9B lane compared the base and both v1.1 lines; their closer-look reads (XT_base_9b_noadapter, the v1.1 lines'
        # closer reads), Lucia v1.1's text preservation read and Lucia v1's reads existed from earlier runs ("earlier")
        "comparisons": [{"of": "base", "reads": ["deep", "lettering", "control_lettering", "control"],
                         "earlier": {"closer": "XT_base_9b_noadapter"}},
                        {"of": "v1.1", "line": "apache", "reads": ["deep", "lettering", "control_lettering", "control"],
                         "earlier": {"closer": "XT_v11_9b_s1", "keep_text": "KT_v11_9b_by"}},
                        {"of": "v1.1", "line": "nc", "reads": ["deep", "lettering", "control_lettering", "control"],
                         "earlier": {"closer": "XT_v11nc_9b_s1"}}],
        "space": {"lookers": {"v1": "lucia-9b", "v1.1": "lucia-9b-v1.1", "v1.1-nc": "lucia-9b-v1.1-nc", "v1.2": "lucia-9b-v1.2"},
                  "gb": 19.31, "tps": 20},
        "todo": [],
        "blockers": [],
    },
    "4b": {
        "key": "4b", "name": "Lucia 4B", "line_name": "Lucia", "family": "Qwen3.5", "model_type": "qwen3_5",
        "base": "Qwen/Qwen3.5-4B", "base_revision": None,
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": True},
        "lines": ("apache", "nc"), "scheduled": True, "lane": "lucia_v12_4b",          # both lines trained and released
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2, "max_len": 2048,   # V12_A1_gkl1_1ep_4b_by
        "image_settings": None,
        "read_batch": 64, "keep_batch": KEEP_BATCH,
        # the 4B came first: its general answers carry no size in their names
        "general_answers": {"anchors": "anchors/v11a", "anchors_by": "anchors/v11a_by", "text": "anchors/v11txt_4b"},
        "template": _QWEN_TEMPLATE,
        "pace": {"apache": _pace(6.23, True, "measured: V12_A1_gkl1_1ep_4b_by, 10,137 steps in 1,053 min"),
                 "nc": _pace(5.6, True, "measured: V12NC_A1_gkl1_1ep_4b, 5.63 s a training step (5.81 with the picture fetch)")},
        "peak": {"apache": {"gb": 50.3, "source": "measured: V12_A1_gkl1_1ep_4b_by"},
                 "nc": {"gb": 50.2, "source": "measured: V12NC_A1_gkl1_1ep_4b"}},
        "repos": {"apache": "AbstractPhil/qwen3.5-4b-think-symbolic-image", "nc": "AbstractPhil/qwen3.5-4b-think-symbolic-image-nc",
                  "full": "AbstractPhil/qwen3.5-4b-think-symbolic-image-full"},
        "tags": {"AbstractPhil/qwen3.5-4b-think-symbolic-image": ["lucia-v1", "lucia-v1.1", "lucia-v1.2"],
                 "AbstractPhil/qwen3.5-4b-think-symbolic-image-nc": ["lucia-v1.1", "lucia-v1.2"],
                 "AbstractPhil/qwen3.5-4b-think-symbolic-image-full": ["lucia-v1"]},
        "licenses": {"apache": "Apache-2.0", "nc": "CC BY-NC-SA 4.0"},
        "versions": [
            _version("v1", "apache", RECORD, "lora/Qwen3.5-4B/r32/20261101_010422_G_med4_2ep_4b/final",
                     ("AbstractPhil/qwen3.5-4b-think-symbolic-image", "lucia-v1"), "v1_4b_s1", "lucia1_4b",
                     source="G_med4_2ep_4b's marker in the record repo"),
            _version("v1.1", "apache", PROJECT, "lora/Qwen3.5-4B/r32/20261112_053335_V11R_A1_gkl1_2ep_4b_by/final",
                     ("AbstractPhil/qwen3.5-4b-think-symbolic-image", "lucia-v1.1"), "v11_4b_by_s1", wordings="v1.1",
                     source="V11R_A1_gkl1_2ep_4b_by, trained in the project repo"),
            _version("v1.1", "nc", PROJECT, "lora/Qwen3.5-4B/r32/20261112_034547_V11RNC_A1_gkl1_2ep_4b/final",
                     ("AbstractPhil/qwen3.5-4b-think-symbolic-image-nc", "lucia-v1.1"), "v11nc_4b_s1", wordings="v1.1",
                     source="V11RNC_A1_gkl1_2ep_4b, trained in the project repo"),
            _version("v1.2", "apache", PROJECT, "lora/Qwen3.5-4B/r32/20261118_112850_V12_A1_gkl1_1ep_4b_by/final",
                     ("AbstractPhil/qwen3.5-4b-think-symbolic-image", "lucia-v1.2"), "v12_4b_by_s1", wordings="v1.1",
                     source="V12_A1_gkl1_1ep_4b_by"),
            _version("v1.2", "nc", PROJECT, "lora/Qwen3.5-4B/r32/20261118_111935_V12NC_A1_gkl1_1ep_4b/final",
                     ("AbstractPhil/qwen3.5-4b-think-symbolic-image-nc", "lucia-v1.2"), "v12_4b_nc_s1", wordings="v1.1",
                     source="V12NC_A1_gkl1_1ep_4b"),
        ],
        # the 4B v1.2 lanes compared the base and both v1.1 lines (deep, lettering, both controls); the closer-look reads and
        # Lucia 4B v1.1's text preservation read existed from earlier runs ("earlier")
        "comparisons": [{"of": "base", "reads": ["deep", "lettering", "control_lettering", "control"],
                         "earlier": {"closer": "XT_base_4b_noadapter"}},
                        {"of": "v1.1", "line": "apache", "reads": ["deep", "lettering", "control_lettering", "control"],
                         "earlier": {"closer": "XT_v11_4b_s1", "keep_text": "KT_v11_4b_by"}},
                        {"of": "v1.1", "line": "nc", "reads": ["deep", "lettering", "control_lettering", "control"],
                         "earlier": {"closer": "XT_v11nc_4b_s1"}}],
        "space": {"lookers": {"v1": "lucia-4b", "v1.1": "lucia-4b-v1.1", "v1.1-nc": "lucia-4b-v1.1-nc", "v1.2": "lucia-4b-v1.2",
                              "v1.2-nc": "lucia-4b-v1.2-nc"}, "gb": 9.32, "tps": 24},
        "todo": [],
        "blockers": [],
    },
    "2b": {
        "key": "2b", "name": "Lucia 2B", "line_name": "Lucia", "family": "Qwen3.5", "model_type": "qwen3_5",
        "base": "Qwen/Qwen3.5-2B", "base_revision": None,
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": True},
        "lines": ("apache", "nc"), "scheduled": True, "lane": "lucia_v12_2b",          # the lane both 2B lines ran in
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2, "max_len": 2048,   # V12_A1_gkl1_1ep_2b_by
        "image_settings": None,
        "read_batch": 64, "keep_batch": KEEP_BATCH,
        "general_answers": {"anchors": "anchors/v11a_2b", "anchors_by": "anchors/v11a_2b_by", "text": "anchors/v11txt_2b"},
        "template": _QWEN_TEMPLATE,
        "pace": {"apache": _pace(3.18, True, "measured: V12_A1_gkl1_1ep_2b_by, 10,137 steps in 538.4 min"),
                 "nc": _pace(2.96, True, "measured: V12NC_A1_gkl1_1ep_2b, 21,815 steps in 1,075.4 min over two machines "
                                         "(the second from step 15,590)")},
        "peak": {"apache": {"gb": 44.2, "source": "measured: V12_A1_gkl1_1ep_2b_by (its smoke 33.3)"},
                 "nc": {"gb": 44.1, "source": "measured: V12NC_A1_gkl1_1ep_2b (its smoke 24.4)"}},
        "repos": {"apache": "AbstractPhil/qwen3.5-2b-think-symbolic-image", "nc": "AbstractPhil/qwen3.5-2b-think-symbolic-image-nc",
                  "full": "AbstractPhil/qwen3.5-2b-think-symbolic-image-full"},
        "tags": {"AbstractPhil/qwen3.5-2b-think-symbolic-image": ["lucia-v1", "lucia-v1.2"],
                 "AbstractPhil/qwen3.5-2b-think-symbolic-image-nc": ["lucia-v1.2"],
                 "AbstractPhil/qwen3.5-2b-think-symbolic-image-full": ["lucia-v1"]},
        "licenses": {"apache": "Apache-2.0", "nc": "CC BY-NC-SA 4.0"},
        "versions": [
            _version("v1", "apache", RECORD, "lora/Qwen3.5-2B/r32/20261101_041601_G_med4_2ep_2b/final",
                     ("AbstractPhil/qwen3.5-2b-think-symbolic-image", "lucia-v1"), "v1_2b_s1", "lucia1_2b",
                     source="G_med4_2ep_2b's marker in the record repo"),
            _version("v1.2", "apache", PROJECT, "lora/Qwen3.5-2B/r32/20261120_064248_V12_A1_gkl1_1ep_2b_by/final",
                     ("AbstractPhil/qwen3.5-2b-think-symbolic-image", "lucia-v1.2"), "v12_2b_by_s1", wordings="v1.1",
                     source="V12_A1_gkl1_1ep_2b_by"),
            _version("v1.2", "nc", PROJECT, "lora/Qwen3.5-2B/r32/20261120_161853_V12NC_A1_gkl1_1ep_2b/final",
                     ("AbstractPhil/qwen3.5-2b-think-symbolic-image-nc", "lucia-v1.2"), "v12_2b_nc_s1", wordings="v1.1",
                     source="V12NC_A1_gkl1_1ep_2b"),
        ],
        # the 2B lane compared the base (closer and deep text reads, the lettering read, both controls) and Lucia 2B v1 (her keep
        # read at strengths 0 and 1, her deep and closer text reads), in that order
        "comparisons": [{"of": "base", "reads": ["closer", "deep", "lettering", "control_lettering", "control"]},
                        {"of": "v1", "line": "apache", "reads": ["keep", "deep", "closer"]}],
        "space": {"lookers": {"v1": "lucia-2b", "v1.2": "lucia-2b-v1.2"}, "gb": 4.55, "tps": 28},
        "todo": [],
        "blockers": [],
    },
    "08b": {
        "key": "08b", "name": "Lucia 0.8B", "line_name": "Lucia", "family": "Qwen3.5", "model_type": "qwen3_5",
        "base": "Qwen/Qwen3.5-0.8B", "base_revision": None,
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": True},
        "lines": ("apache", "nc"), "scheduled": True,
        "lane": "lucia_v12_08b",                               # the lane both 0.8B lines ran in
        # the v1.2 recipe; Lucia 0.8B v1 is the recipe's verbatim arm (lr 2e-4, 8 x 2; G_med4_2ep_08b), chosen over a
        # three-epoch lr 3e-4 arm
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2, "max_len": 2048,
        "image_settings": None,
        "read_batch": 64, "keep_batch": KEEP_BATCH,                     # her v1 reads ran at 64 (P_probe_08b)
        "general_answers": {"anchors": "anchors/v11a_08b", "anchors_by": "anchors/v11a_08b_by", "text": "anchors/v11txt_08b"},
        "template": _QWEN_TEMPLATE,
        "pace": {"apache": _pace(2.09, True, "measured: V12_A1_gkl1_1ep_08b_by, 10,137 steps in 353.0 min"),
                 "nc": _pace(1.9, True, "measured: V12NC_A1_gkl1_1ep_08b, 21,815 steps in 689.1 min")},
        "peak": {"apache": {"gb": 40.51, "source": "measured: V12_A1_gkl1_1ep_08b_by (her smoke 29.92)"},
                 "nc": {"gb": 40.47, "source": "measured: V12NC_A1_gkl1_1ep_08b (her smoke 23.27)"}},
        "repos": {"apache": "AbstractPhil/qwen3.5-0.8b-think-symbolic-image",
                  "nc": "AbstractPhil/qwen3.5-0.8b-think-symbolic-image-nc",
                  "full": "AbstractPhil/qwen3.5-0.8b-think-symbolic-image-full"},
        "tags": {"AbstractPhil/qwen3.5-0.8b-think-symbolic-image": ["lucia-v1", "lucia-v1.2"],
                 "AbstractPhil/qwen3.5-0.8b-think-symbolic-image-nc": ["lucia-v1.2"],
                 "AbstractPhil/qwen3.5-0.8b-think-symbolic-image-full": ["lucia-v1"]},
        "licenses": {"apache": "Apache-2.0", "nc": "CC BY-NC-SA 4.0"},
        "versions": [
            _version("v1", "apache", RECORD, "lora/Qwen3.5-0.8B/r32/20261101_041418_G_med4_2ep_08b/final",
                     ("AbstractPhil/qwen3.5-0.8b-think-symbolic-image", "lucia-v1"), "v1_08b_s1", "lucia1_08b",
                     source="G_med4_2ep_08b's marker in the record repo"),
            _version("v1.2", "apache", PROJECT, "lora/Qwen3.5-0.8B/r32/20261121_150248_V12_A1_gkl1_1ep_08b_by/final",
                     ("AbstractPhil/qwen3.5-0.8b-think-symbolic-image", "lucia-v1.2"), "v12_08b_by_s1", wordings="v1.1",
                     source="V12_A1_gkl1_1ep_08b_by"),
            _version("v1.2", "nc", PROJECT, "lora/Qwen3.5-0.8B/r32/20261121_212736_V12NC_A1_gkl1_1ep_08b/final",
                     ("AbstractPhil/qwen3.5-0.8b-think-symbolic-image-nc", "lucia-v1.2"), "v12_08b_nc_s1", wordings="v1.1",
                     source="V12NC_A1_gkl1_1ep_08b"),
        ],
        "comparisons": None,
        "space": {"lookers": {"v1": "lucia-0.8b"}, "gb": 1.75, "tps": 30},
        "todo": ["comparisons: none recorded; the generator's default set applies until her lane's plan names one"],
        "blockers": [],
    },
    # ------------------------------------------------------------------------------------------------ Ulisse (Gemma 4)
    "ge4b": {
        "key": "ge4b", "name": "Ulisse E4B", "line_name": "Ulisse", "family": "Gemma 4", "model_type": "gemma4",
        "base": "google/gemma-4-E4B-it", "base_revision": None,
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": False},
        "lines": ("apache", "nc"), "scheduled": True, "lane": "ulisse_v12_ge4b",       # the generator's rule
        # the v1.2 recipe; Ulisse E4B v1 trained at 2e-4, 8 x 2, max_len 2,048 and saw (G_med4_2ep_ge4b)
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2, "max_len": 2048,
        "image_settings": None,                                      # his v1: the processor's default (280 picture tokens)
        "read_batch": 64, "keep_batch": KEEP_BATCH,                  # P_probe_ge4b ran at 64
        "general_answers": {"anchors": "anchors/v11a_ge4b", "anchors_by": "anchors/v11a_ge4b_by", "text": "anchors/v11txt_ge4b"},
        # Gemma 4 seldom ends a user turn it writes (it answers inside it), so its text requests come from the Qwen-written file
        # (from 0.1.70): his v11txr_ge4b (2,000) and v11txh_ge4b (200) asked them; his Apache-2.0 training came before the fix
        "general_requests": {"requests": "hub:text/requests_qwen_v1.json", "n_text": 2000, "n_held_text": 200,
                             "as_run": "Our Apache-2.0 version was trained before this file existed: its general text rows "
                                       "answered requests gemma-4-E4B-it wrote itself, most of which were not requests (the "
                                       "model goes on to answer inside the turn it writes), and none was held out. Its held "
                                       "text requests (for its bars) and every text request of our -nc line were then asked "
                                       "from this file, as this lane asks them."},
        "template": _GEMMA_TEMPLATE + "; a separate vision tower (frozen under the LoRA)",
        "pace": {"apache": _pace(4.75, True, "measured: V12_A1_gkl1_1ep_ge4b_by, 10,137 steps in 803.1 min"),
                 "nc": _pace(4.65, True, "measured: V12NC_A1_gkl1_1ep_ge4b, 21,815 steps in 1,689.5 min over two machines "
                                         "(the second from step 5,450)")},
        "peak": {"apache": {"gb": 50.16, "source": "measured: V12_A1_gkl1_1ep_ge4b_by (his smoke 44.55)"},
                 "nc": {"gb": 51.71, "source": "measured: V12NC_A1_gkl1_1ep_ge4b (his smoke 42.54)"}},
        "repos": {"apache": "AbstractPhil/gemma-4-e4b-think-symbolic-image",
                  "nc": "AbstractPhil/gemma-4-e4b-think-symbolic-image-nc", "full": None},
        "tags": {"AbstractPhil/gemma-4-e4b-think-symbolic-image": ["ulisse-v1", "ulisse-v1.2", "v1"],
                 "AbstractPhil/gemma-4-e4b-think-symbolic-image-nc": ["ulisse-v1.2"]},
        "licenses": {"apache": "Apache-2.0", "nc": "CC BY-NC-SA 4.0"},
        "versions": [
            _version("v1", "apache", RECORD, "lora/gemma-4-E4B-it/r32/20261101_093141_G_med4_2ep_ge4b/final",
                     ("AbstractPhil/gemma-4-e4b-think-symbolic-image", "ulisse-v1"), "v1_ge4b_s1", "ulisse1_ge4b",
                     source="G_med4_2ep_ge4b's marker in the record repo"),
            _version("v1.2", "apache", PROJECT, "lora/gemma-4-E4B-it/r32/20261122_101943_V12_A1_gkl1_1ep_ge4b_by/final",
                     ("AbstractPhil/gemma-4-e4b-think-symbolic-image", "ulisse-v1.2"), "v12_ge4b_by_s1", wordings="v1.1",
                     source="V12_A1_gkl1_1ep_ge4b_by"),
            _version("v1.2", "nc", PROJECT, "lora/gemma-4-E4B-it/r32/20261123_013549_V12NC_A1_gkl1_1ep_ge4b/final",
                     ("AbstractPhil/gemma-4-e4b-think-symbolic-image-nc", "ulisse-v1.2"), "v12_ge4b_nc_s1", wordings="v1.1",
                     source="V12NC_A1_gkl1_1ep_ge4b"),
        ],
        "comparisons": None,
        "space": {"lookers": {"v1": "ulisse-e4b"}, "gb": 15.99, "tps": 25},
        # 2026-11-27: the two open values settled (v0.1.77): his bars read with `reads ge4b apache bars --keep
        # K_v12_A1_ge4b_by_t` (BARS "keep"); his text requests from text/requests_qwen_v1.json (general_requests; his -nc
        # trained on the merged anchors/v11a_ge4b_r, its keep read anchors/v11a_ge4b_t, which one run of this lane makes)
        "todo": ["comparisons: none recorded; the generator's default set applies until his lane's plan names one"],
        "blockers": [],
    },
    "g12b": {
        "key": "g12b", "name": "Ulisse 12B", "line_name": "Ulisse", "family": "Gemma 4", "model_type": "gemma4_unified",
        "base": "google/gemma-4-12B-it", "base_revision": None,
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": False},
        "lines": ("apache", "nc"), "scheduled": True, "lane": "ulisse_v12_g12b",       # the generator's rule
        # lr 5e-5: at the recipe's 2e-4 his first v1 arm went blind to the picture (a loss spike near step 440); at 5e-5 two
        # seeds saw (G_med4_2ep_g12b_lr5, released; G_med4_2ep_g12b_lr5_s2)
        "lora": {"rank": 32, "alpha": 64}, "lr": 5e-5, "batch": 8, "grad_accum": 2, "max_len": 2048,
        "image_settings": None,
        "read_batch": 64, "keep_batch": KEEP_BATCH,                  # P_probe_g12b_lr5 ran at 64
        "general_answers": {"anchors": "anchors/v11a_g12b", "anchors_by": "anchors/v11a_g12b_by", "text": "anchors/v11txt_g12b"},
        # his v11a_g12b asked its text requests from the Qwen-written file (2,000 + 200), as Gemma 4 needs (the E4B's lesson)
        "general_requests": {"requests": "hub:text/requests_qwen_v1.json", "n_text": 2000, "n_held_text": 200},
        "template": (_GEMMA_TEMPLATE + "; encoder-free: the early language layers read the picture and sit under the LoRA, so "
                     "an arm's picture dependence is read at its early checkpoints (tools.picture_probe), not only at the end"),
        "pace": {"apache": _pace(9.04, True, "measured: V12_A1_gkl1_1ep_g12b_by, 10,137 steps in 1,527.8 min over two machines "
                                             "(the second from step 8,590)"),
                 "nc": _pace(8.53, True, "measured: V12NC_A1_gkl1_1ep_g12b, 21,815 steps in 3,103.0 min over three machines "
                                         "(the second from step 9,940, the third from step 19,740)")},
        "peak": {"apache": {"gb": 61.78, "source": "measured: V12_A1_gkl1_1ep_g12b_by (his smoke 58.51)"},
                 "nc": {"gb": 62.74, "source": "measured: V12NC_A1_gkl1_1ep_g12b (his smoke 56.1)"}},
        "repos": {"apache": "AbstractPhil/gemma-4-12b-think-symbolic-image",
                  "nc": "AbstractPhil/gemma-4-12b-think-symbolic-image-nc", "full": None},
        "tags": {"AbstractPhil/gemma-4-12b-think-symbolic-image": ["ulisse-v1", "ulisse-v1.2", "v1"],
                 "AbstractPhil/gemma-4-12b-think-symbolic-image-nc": ["ulisse-v1.2"]},
        "licenses": {"apache": "Apache-2.0", "nc": "CC BY-NC-SA 4.0"},
        "versions": [
            _version("v1", "apache", RECORD, "lora/gemma-4-12B-it/r32/20261101_151340_G_med4_2ep_g12b_lr5/final",
                     ("AbstractPhil/gemma-4-12b-think-symbolic-image", "ulisse-v1"), "v1_g12b_s1", "ulisse1_g12b",
                     source="G_med4_2ep_g12b_lr5's marker in the record repo"),
            _version("v1.2", "apache", PROJECT, "lora/gemma-4-12B-it/r32/20261124_053851_V12_A1_gkl1_1ep_g12b_by/final",
                     ("AbstractPhil/gemma-4-12b-think-symbolic-image", "ulisse-v1.2"), "v12_g12b_by_s1", wordings="v1.1",
                     source="V12_A1_gkl1_1ep_g12b_by"),
            _version("v1.2", "nc", PROJECT, "lora/gemma-4-12B-it/r32/20261124_064044_V12NC_A1_gkl1_1ep_g12b/final",
                     ("AbstractPhil/gemma-4-12b-think-symbolic-image-nc", "ulisse-v1.2"), "v12_g12b_nc_s1", wordings="v1.1",
                     source="V12NC_A1_gkl1_1ep_g12b"),
        ],
        "comparisons": None,
        "space": {"lookers": {"v1": "ulisse-12b"}, "gb": 23.92, "tps": 18},
        "todo": ["comparisons: none recorded; the generator's default set applies until his lane's plan names one"],
        "blockers": [],
    },
    # ------------------------------------------------------------------------------------------------ the other families
    "north": {
        "key": "north", "name": "North-Micro-Vision", "line_name": None, "family": "North", "model_type": "cohere_compass",
        "base": "CohereLabs/North-Micro-Vision-Instruct", "base_revision": None,
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": False},
        "lines": ("apache",), "scheduled": True, "lane": "north_v12",                  # the generator's rule
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2, "max_len": 2048,   # G_med4_2ep_north
        "image_settings": None,
        "read_batch": 64, "keep_batch": KEEP_BATCH,                  # P_probe_north ran at 64
        "general_answers": {"anchors": "anchors/v11a_north", "anchors_by": "anchors/v11a_north_by",
                            "text": "anchors/v11txt_north"},
        # its v1.2 lane's general text requests: the cleaned requests file, as Anima's eyes (ae1b) take them
        "general_requests": {"requests": "hub:text/requests_qwen_v1_clean.json", "n_text": 500, "n_held_text": 40},
        "template": "no family-specific handling: the shared path (the chat template's own picture parts and turn end)",
        "pace": {"apache": _pace(3.46, True, "measured: V12_A1_gkl1_1ep_north_by, 10,137 steps in 585.1 min")},
        "peak": {"apache": {"gb": 46.23, "source": "measured: V12_A1_gkl1_1ep_north_by (its smoke 34.44)"}},
        "repos": {"apache": "AbstractPhil/north-micro-vision-think-symbolic-image", "nc": None, "full": None},
        "tags": {"AbstractPhil/north-micro-vision-think-symbolic-image": ["v1", "v1.2"]},
        "licenses": {"apache": "Apache-2.0"},
        "versions": [
            _version("v1", "apache", RECORD, "lora/North-Micro-Vision-Instruct/r32/20261102_041917_G_med4_2ep_north/final",
                     ("AbstractPhil/north-micro-vision-think-symbolic-image", "v1"), "v1_north_s1", "v1_north",
                     source="G_med4_2ep_north's marker in the record repo"),
            _version("v1.2", "apache", PROJECT,
                     "lora/North-Micro-Vision-Instruct/r32/20261126_192053_V12_A1_gkl1_1ep_north_by/final",
                     ("AbstractPhil/north-micro-vision-think-symbolic-image", "v1.2"), "v12_north_by_s1", wordings="v1.1",
                     source="V12_A1_gkl1_1ep_north_by"),
        ],
        "comparisons": None,
        "space": {"lookers": {"v1": "north"}, "gb": 4.97, "tps": 45},
        "todo": ["comparisons: none recorded; the generator's default set applies until its lane's plan names one"],
        "blockers": [],
    },
    "glm46v": {
        "key": "glm46v", "name": "GLM-4.6V-Flash", "line_name": None, "family": "GLM-4.6V", "model_type": "glm4v",
        "base": "zai-org/GLM-4.6V-Flash", "base_revision": None,
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": False},
        "lines": ("apache",), "scheduled": True, "lane": "glm46v_v12",                 # the generator's rule
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2,
        "max_len": 2560,                                             # G_med4_2ep_glm46v: its pictures take more tokens
        "image_settings": None,
        "read_batch": 64, "keep_batch": KEEP_BATCH,                  # P_probe_glm46v ran at 64
        "general_answers": {"anchors": "anchors/v11a_glm46v", "anchors_by": "anchors/v11a_glm46v_by",
                            "text": "anchors/v11txt_glm46v"},
        # its v1.2 lane's general text requests: the cleaned requests file, as Anima's eyes (ae1b) take them
        "general_requests": {"requests": "hub:text/requests_qwen_v1_clean.json", "n_text": 500, "n_held_text": 40},
        "template": "GLM-4V: the MLP fuses its gate and up projections (gate_up_proj), which the LoRA targets name",
        "pace": {"apache": _pace(13.59, True, "measured: V12_A1_gkl1_1ep_glm46v_by, 10,137 steps in 2,295.8 min over two machines "
                                              "(the second from step 5,940)")},
        "peak": {"apache": {"gb": 55.76, "source": "measured: V12_A1_gkl1_1ep_glm46v_by (its smoke 45.57)"}},
        "repos": {"apache": "AbstractPhil/glm-4.6v-flash-think-symbolic-image", "nc": None, "full": None},
        "tags": {"AbstractPhil/glm-4.6v-flash-think-symbolic-image": ["v1", "v1.2"]},
        "licenses": {"apache": "MIT"},                               # its card follows the base's MIT license
        "versions": [
            _version("v1", "apache", RECORD, "lora/GLM-4.6V-Flash/r32/20261102_041223_G_med4_2ep_glm46v/final",
                     ("AbstractPhil/glm-4.6v-flash-think-symbolic-image", "v1"), "v1_glm46v_s1", "v1_glm46v",
                     source="G_med4_2ep_glm46v's marker in the record repo"),
            _version("v1.2", "apache", PROJECT, "lora/GLM-4.6V-Flash/r32/20261126_013856_V12_A1_gkl1_1ep_glm46v_by/final",
                     ("AbstractPhil/glm-4.6v-flash-think-symbolic-image", "v1.2"), "v12_glm46v_by_s1", wordings="v1.1",
                     source="V12_A1_gkl1_1ep_glm46v_by"),
        ],
        "comparisons": None,
        "space": {"lookers": {"v1": "glm"}, "gb": 20.59, "tps": 20},
        "todo": ["comparisons: none recorded; the generator's default set applies until its lane's plan names one"],
        "blockers": [],
    },
    "molmo2o": {
        "key": "molmo2o", "name": "Molmo2-O-7B", "line_name": None, "family": "Molmo2", "model_type": "molmo2",
        "base": "allenai/Molmo2-O-7B", "base_revision": "784410650d12be9bc086118fdefa32d2c3bced86",   # its v1 arm's pin
        "trust_remote_code": True, "transformers": "==4.57.6",       # its code was built on transformers 4.57
        "install": {"transformers": "transformers==4.57.6", "fast_path": False},
        "lines": ("apache",), "scheduled": False, "lane": "molmo2o_v12",               # training instructions only
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2,
        "max_len": 2560,                                             # G_med4_2ep_molmo2o_v4
        "image_settings": None,
        "read_batch": 32, "keep_batch": KEEP_BATCH,                  # P_probe_molmo2o_v4 ran at 32 (its attention cache is large)
        "general_answers": {"anchors": "anchors/v11a_molmo2o", "anchors_by": "anchors/v11a_molmo2o_by",
                            "text": "anchors/v11txt_molmo2o"},
        "template": ("Molmo2 (repository code): the processor assumes left padding, so the training mask is rebuilt from the pad "
                     "id; the picture features are added in place, so no input-embedding grad mark; a batch mixing picture "
                     "rows with text rows is built one picture row at a time (family.prepare_model)"),
        "pace": {"apache": _pace(11.5, False, _EST.format(v1=7.43, run="G_med4_2ep_molmo2o_v4"))},
        "peak": {"apache": {"gb": None, "source": "not measured: its smoke measures it"}},
        "repos": {"apache": "AbstractPhil/molmo2-o-7b-think-symbolic-image", "nc": None, "full": None},
        "tags": {"AbstractPhil/molmo2-o-7b-think-symbolic-image": ["v1"]},
        "licenses": {"apache": "Apache-2.0"},
        "versions": [
            _version("v1", "apache", RECORD, "lora/Molmo2-O-7B/r32/20261102_123335_G_med4_2ep_molmo2o_v4/final",
                     ("AbstractPhil/molmo2-o-7b-think-symbolic-image", "v1"), "v1_molmo2o_s1", "v1_molmo2o",
                     source="G_med4_2ep_molmo2o_v4's marker in the record repo"),
        ],
        "comparisons": None,
        "space": {"lookers": {"v1": "molmo2"}, "gb": 15.52, "disk_gb": 31.04, "tps": 16, "remote": True},
        "todo": ["peak apache: measured by its smoke",
                 "comparisons: none recorded; the generator's default set applies until its lane's plan names one"],
        "blockers": ["the general answers (tools.anchors) and the preservation reads (tools.keep_read) load the base without "
                     "its repository code and pinned revision: they cannot run for this model until they take "
                     "trust_remote_code and revision"],
    },
    "phi4rv": {
        "key": "phi4rv", "name": "Phi-4-reasoning-vision-15B", "line_name": None, "family": "Phi-4-reasoning-vision",
        "model_type": "phi4-siglip",
        "base": "microsoft/Phi-4-reasoning-vision-15B", "base_revision": "c3e4fac79ddace21976ced56fbf1564b8bd8c89f",   # its v1 pin
        "trust_remote_code": True, "transformers": "==4.57.6",       # its code subclasses an image-processor module 5.x rewrote
        "install": {"transformers": "transformers==4.57.6", "fast_path": False},
        "lines": ("apache",), "scheduled": False, "lane": "phi4rv_v12",                # training instructions only
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2, "max_len": 2048,   # G_med4_2ep_phi4rv_p1024_v3
        # 1,024 picture patches: at the processor's full budget a step took 39 s; its base grounded the same at 1,024
        "image_settings": {"max_num_patches": 1024},
        "read_batch": 64, "keep_batch": KEEP_BATCH,                  # P_probe_phi4rv_p1024_v3 ran at 64
        "general_answers": {"anchors": "anchors/v11a_phi4rv", "anchors_by": "anchors/v11a_phi4rv_by",
                            "text": "anchors/v11txt_phi4rv"},
        "template": ("Phi-4-reasoning-vision (repository code): the processor has no chat template (the tokenizer renders it); "
                     "a picture row's content is one string with <image>; one picture slot per row of a batch; batched "
                     "generation runs from left-padded embeddings; a <nothink> mode tag is stripped from answers"),
        "pace": {"apache": _pace(21.8, False, _EST.format(v1=14.07, run="G_med4_2ep_phi4rv_p1024_v3"))},
        "peak": {"apache": {"gb": None, "source": "not measured: its smoke measures it"}},
        "repos": {"apache": "AbstractPhil/phi-4-reasoning-vision-15b-think-symbolic-image", "nc": None, "full": None},
        "tags": {"AbstractPhil/phi-4-reasoning-vision-15b-think-symbolic-image": ["v1"]},
        "licenses": {"apache": "MIT"},                               # its card follows the base's MIT license
        "versions": [
            _version("v1", "apache", RECORD,
                     "lora/Phi-4-reasoning-vision-15B/r32/20261102_141924_G_med4_2ep_phi4rv_p1024_v3/final",
                     ("AbstractPhil/phi-4-reasoning-vision-15b-think-symbolic-image", "v1"), "v1_phi4rv_s1", "v1_phi4rv",
                     source="G_med4_2ep_phi4rv_p1024_v3's marker in the record repo"),
        ],
        "comparisons": None,
        "space": {"lookers": {"v1": "phi4"}, "gb": 30.24, "tps": 12, "remote": True, "settings": {"max_num_patches": 1024}},
        "todo": ["peak apache: measured by its smoke",
                 "comparisons: none recorded; the generator's default set applies until its lane's plan names one"],
        "blockers": ["the general answers (tools.anchors) and the preservation reads (tools.keep_read) load the base without "
                     "its repository code, pinned revision and picture budget: they cannot run for this model until they "
                     "take trust_remote_code, revision and image_settings"],
    },
    "mi14": {
        "key": "mi14", "name": "Ministral 3 14B", "line_name": None, "family": "Ministral 3", "model_type": "mistral3",
        "base": "mistralai/Ministral-3-14B-Instruct-2512-BF16", "base_revision": None,
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": False},
        "lines": ("apache",), "scheduled": False, "lane": "mi14_v12",                  # training instructions only
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2,
        "max_len": 2560,                                             # G_med4_2ep_mi14
        "image_settings": None,
        "read_batch": 64, "keep_batch": KEEP_BATCH,                  # its reads at 64 once its pictures went one at a time (0.1.38)
        "general_answers": {"anchors": "anchors/v11a_mi14", "anchors_by": "anchors/v11a_mi14_by", "text": "anchors/v11txt_mi14"},
        "template": ("Ministral 3: the image processor pads (its config leaves do_pad unset); its vision tower takes a batch's "
                     "pictures one at a time (in one packed sequence the later pictures came out wrong on the card)"),
        "pace": {"apache": _pace(31.6, False, _EST.format(v1=20.41, run="G_med4_2ep_mi14"))},
        "peak": {"apache": {"gb": None, "source": "not measured: its smoke measures it"}},
        "repos": {"apache": "AbstractPhil/ministral-3-14b-think-symbolic-image", "nc": None, "full": None},
        "tags": {"AbstractPhil/ministral-3-14b-think-symbolic-image": ["v1"]},
        "licenses": {"apache": "Apache-2.0"},
        "versions": [
            _version("v1", "apache", RECORD, "lora/Ministral-3-14B-Instruct-2512-BF16/r32/20261102_060730_G_med4_2ep_mi14/final",
                     ("AbstractPhil/ministral-3-14b-think-symbolic-image", "v1"), "v1_mi14_s1", "v1_mi14",
                     source="G_med4_2ep_mi14's marker in the record repo"),
        ],
        "comparisons": None,
        "space": {"lookers": {"v1": "ministral-14b"}, "gb": 27.89, "tps": 14, "skip": ["consolidated*"]},
        "todo": ["peak apache: measured by its smoke",
                 "comparisons: none recorded; the generator's default set applies until its lane's plan names one"],
        "blockers": [],
    },
}

FIELDS = ("key", "name", "line_name", "family", "model_type", "base", "base_revision", "trust_remote_code", "transformers",
          "install", "lines", "scheduled", "lane", "lora", "lr", "batch", "grad_accum", "max_len", "image_settings", "read_batch",
          "keep_batch", "general_answers", "template", "pace", "peak", "repos", "tags", "licenses", "versions", "comparisons",
          "space", "todo", "blockers")
# What an entry's lane may carry beyond the common recipe (lines.py reads them; the table below): a version-1 entry carries one
# only when its v1.2 lane needs it (North-Micro-Vision, GLM-4.6V-Flash and the two Gemma 4 models take their general text
# requests from a file; "as_run" in it, for the guide, says where our own version's lane differed).
EXTRA_FIELDS = ("prepare", "general_requests", "train_extra", "full", "retrains")

# Models added to the catalogue after the version-1 list: no version 1 and no Space looker yet. Beside FIELDS an entry here may
# carry what its lane needs beyond the common recipe (lines.py reads them):
# | prepare          | a tool run that comes first in the lane and makes the base model (its outputs in "makes"); every other
#                    | run of the lane waits for it ("waits_for_runs") and for those files ("waits_for") |
# | general_requests | the text requests of its general rows from a requests file (tools.anchors requests=), with the counts |
# | train_extra      | train keys its trainings carry beside the recipe (the slot term of Anima's eyes) |
# | full             | a full fine-tune beside the LoRA, on the same rows: its run key, settings, pace and peak |
# | retrains         | the LoRA trained again with some of its settings changed, each under its own run key: {key: the values
#                    | it sets}; registry.get(key) gives the model's entry with those values (retrain_entry), and its lane is
#                    | the one training with its reads. "no_smoke" (a reason) leaves its smoke out: the run it retrains measured
#                    | its rows and settings. Its comparisons name the reads the model's own lane made ("earlier") |
_AE = anima_eyes
LATER = {
    # ------------------------------------------------------------------------------- Anima's eyes (InternVL3.5-1B, last 4)
    "ae1b": {
        "key": "ae1b", "name": "Anima Eyes 1B", "line_name": None, "family": "InternVL3.5 + Anima's Qwen3-0.6B",
        "model_type": "internvl",
        "base": _AE.MODEL_ID, "base_revision": None,         # built by the lane's first run from two pinned sources (prepare)
        "trust_remote_code": False, "transformers": None, "install": {"transformers": "transformers>=5.0", "fast_path": False},
        "lines": ("apache",), "scheduled": True,              # Apache only, as the catalogue's other newer members
        "lane": "anima_eyes_v12",
        # the v1.2 recipe, as Lucia 0.8B's (the nearest size); the slot term rides beside it (train_extra)
        "lora": {"rank": 32, "alpha": 64}, "lr": 2e-4, "batch": 8, "grad_accum": 2, "max_len": 2048,
        "image_settings": None,                               # the processor's default: one 448-pixel tile, 256 picture tokens
        "read_batch": 64, "keep_batch": KEEP_BATCH,
        "general_answers": {"anchors": "anchors/v11a_ae1b", "anchors_by": "anchors/v11a_ae1b_by", "text": "anchors/v11txt_ae1b"},
        "template": ("InternVL3.5 (transformers' own InternVL class, no repository code): no thinking switch; the assistant turn "
                     "ends <|im_end|>; one 448-pixel tile a picture (256 tokens). Anima's eyes: InternVL3.5-1B with Anima's own "
                     "language layers 24-27 and final norm (Qwen3-0.6B-Base); that slot carries no LoRA module and stays frozen "
                     "in the full fine-tune; the slot term holds its final states to Anima's encoder's (anima_eyes.py)"),
        "pace": {"apache": _pace(2.01, True, "measured: V12_A1_gkl1_1ep_ae1b_by, 10,137 steps in 338.9 min")},
        "peak": {"apache": {"gb": 21.29, "source": "measured: V12_A1_gkl1_1ep_ae1b_by (its smoke S_v12_smoke_gc1_ae1b_t1 18.89)"}},
        # the full fine-tune is the release; the LoRA is not released (its rows never ended: the retrain ae1bt below)
        "repos": {"apache": None, "nc": None, "full": "AbstractPhil/anima-eyes-1b-think-symbolic-image-full"},
        "tags": {"AbstractPhil/anima-eyes-1b-think-symbolic-image-full": ["v1.2"]},
        "licenses": {"apache": "Apache-2.0"},                 # InternVL3.5-1B and Qwen3-0.6B-Base are both Apache-2.0
        "versions": [
            _version("v1.2", "apache", PROJECT, "full/InternVL3_5-1B-anima-last4/20261125_123803_V12_A1_gkl1_1ep_ae1bf_by/final",
                     ("AbstractPhil/anima-eyes-1b-think-symbolic-image-full", "v1.2"), "v12_ae1bf_by_s1", wordings="v1.1",
                     source="V12_A1_gkl1_1ep_ae1bf_by, the full fine-tune"),
        ],
        "comparisons": [{"of": "base", "reads": ["closer", "deep", "lettering", "control_lettering", "control"]}],
        "space": None,
        "todo": ["repos apache: the LoRA's release repo, named at a release (the full fine-tune's is released; the first LoRA's "
                 "rows never ended, its retrain is ae1bt)",
                 "an A/B in ComfyUI of the LoRA (Anima with it in its text-encoder slot against its own encoder) before it is "
                 "called done (the full fine-tune's: a side-by-side look in ComfyUI on 2026-11-25, before its release)"],
        "blockers": [],
        "prepare": {"name": "base_ae1b", "module": "anima_eyes_base", "args": {"head": _AE.HEAD},
                    "makes": [f"{_AE.FOLDER}/model.safetensors", f"{_AE.FOLDER}/config.json", f"{_AE.FOLDER}/anima_eyes.json"]},
        # the cleaned requests file (a look at 30 requests of the uncleaned file found 23 incomplete: cut mid-sentence, or a task
        # set up without its text or question); it keeps 530 training and 41 held requests: 500 and 40 are asked, so a further
        # cut after a second look still fits
        "general_requests": {"requests": "hub:text/requests_qwen_v1_clean.json", "n_text": 500, "n_held_text": 40},
        "train_extra": {"slot": 1.0, "slot_batch": 4},
        # the full fine-tune: the 0.8B's full rate (2e-5: 'a smaller model takes the higher end'), its 8 x 2; Anima's slot
        # frozen; the general rows' KL on a frozen reference copy; the slot term as the LoRA's; one pass, no resume (a full
        # fine-tune's state is the whole model), one ship at the end
        "about": ("Anima's eyes: [OpenGVLab/InternVL3_5-1B-HF](https://huggingface.co/OpenGVLab/InternVL3_5-1B-HF) (its vision "
                  "tower, projector, word table and language layers 0-23) with the language layers 24-27 and final norm of "
                  "[Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base), Anima's own text encoder (the "
                  "\"last-4 hybrid\", built by the lane's first run from both at pinned commits)"
                  + (", and the Base's own word table as its output head" if _AE.HEAD == "base" else
                     ", and InternVL's own output head")),
        "about_training": ("Anima reads this model's final states (after the last norm) in its text-encoder slot. So Anima's slot "
                           "(the Base's layers 24-27 and final norm) carries no LoRA module and stays frozen in the full fine-tune, "
                           "and beside every micro-batch four Anima-style prompts (the training rows' own generation prompts, tags "
                           "and captions) hold the model's final states to the Base's (the slot term at weight 1: the relative "
                           "squared error per token). Every log line reads the agreement on 24 held Anima-style prompts "
                           "(slot_held, the centered cosine; the hybrid starts at 0.805)."),
        "full": {"key": "ae1bf", "lr": 2e-5, "batch": 8, "grad_accum": 2, "max_len": 2048, "general_kl": 1.0,
                 "pace": _pace(1.87, True, "measured: V12_A1_gkl1_1ep_ae1bf_by, 10,137 steps in 315.7 min"),
                 "peak": {"gb": 35.67, "source": "measured: V12_A1_gkl1_1ep_ae1bf_by (its smoke S_v12_smoke_gc1_ae1bf_t1 31.99)"}},
        # The LoRA again, with its output head's turn-end row trained beside its pairs (ae1bt). The first LoRA
        # (V12_A1_gkl1_1ep_ae1b_by) wrote each row right and then never ended it: junk tokens and newlines to the token cap,
        # so every read rejected its rows (0 or 1 kept of every set; the full fine-tune kept 97-100% of the text reads). The
        # model's output head is the Base's word table, whose <|im_end|> row (151645) was never trained (norm 0.350, one of
        # 2,245 rows at that norm; trained rows have a median of 0.968); the reads stop on that token, and the LoRA of layers
        # 0-23 did not raise it through the frozen top layers. The full fine-tune trains its head (that row moved the most of
        # all rows, relative change 0.291) and ends its rows. The head is not part of Anima's slot: Anima reads the final states.
        "retrains": {
            "ae1bt": {
                "lora": {"rank": 32, "alpha": 64, "token_rows": {"lm_head": ["turn_end"]}},
                "no_smoke": ("the LoRA it retrains measured these rows and settings (V12_A1_gkl1_1ep_ae1b_by after its smoke "
                             "S_v12_smoke_gc1_ae1b_t1: 10,137 steps in 338.9 min, peak 21.29 GB); the head's row adds 1,024 "
                             "trained numbers to its 17,301,504"),
                "pace": {"apache": _pace(2.0, False, "estimate: the LoRA it retrains, V12_A1_gkl1_1ep_ae1b_by, measured 2.0 s a "
                                                     "step (10,137 steps in 338.9 min); the head's row adds one copy of the "
                                                     "head a forward")},
                "peak": {"apache": {"gb": 21.29, "source": "estimate: the LoRA it retrains peaked at 21.29 GB"}},
                # the base comparisons ran in the model's own lane, under these names (not made again)
                "comparisons": [{"of": "base", "reads": [],
                                 "earlier": {"closer": "XT_base_ae1b_noadapter_t1", "deep": "XT_base_ae1b_deep",
                                             "lettering": "XTA_base_ae1b", "control_lettering": "XCA2_base_ae1b",
                                             "control": "XC2_base_ae1b"}}],
                "todo": ["repos apache: a release repo, named at a release",
                         "its reads: whether its rows end (the first LoRA's never did), before it is called done"],
                "about_retrain": ("This lane trains the LoRA again (the first was V12_A1_gkl1_1ep_ae1b_by) on the same rows, "
                                  "with the same recipe, slot term and general rows, and one change: the output head's "
                                  "turn-end row (<|im_end|>, the token every read stops on) trains beside the LoRA pairs "
                                  "(1,024 numbers, PEFT's trainable token rows; the rest of the head stays frozen). The first "
                                  "LoRA wrote its rows and never ended them: its head is the Base's word table, whose "
                                  "turn-end row was never trained, and a LoRA of the lower layers did not raise that token. "
                                  "Anima reads the final states, not the head, so the row does not touch Anima's slot. The "
                                  "base comparisons and the general answers were made by the first lane and are not made "
                                  "again; its pace and memory were measured there, so it runs without a smoke."),
            },
        },
    },
}


def keys() -> list:
    return list(ENTRIES)


def all_keys() -> list:
    """Every model's key: the version-1 list, then the later additions (LATER)."""
    return list(ENTRIES) + list(LATER)


def retrain_keys() -> list:
    """The run keys of every retrain (an entry's "retrains"), in registry order."""
    return [k for e in list(ENTRIES.values()) + list(LATER.values()) for k in (e.get("retrains") or {})]


def get(name: str) -> dict:
    """An entry (a deep copy) by its run key, its base model id, one of its release repos or one of its Space looker keys
    (case does not matter); the version-1 list first, then the later additions (LATER). A retrain's run key gives the retrain's
    entry (retrain_entry), a full fine-tune's run key the full fine-tune's (full_entry: `reads ae1bf apache bars`)."""
    want = str(name).strip().lower()
    for key, e in list(ENTRIES.items()) + list(LATER.items()):
        names = {key, e["base"].lower(), e["base"].split("/")[-1].lower(), e["name"].lower()}
        names |= {r.lower() for r in e["repos"].values() if r} | {r.split("/")[-1].lower() for r in e["repos"].values() if r}
        names |= {str(k).lower() for k in (e.get("space") or {}).get("lookers", {}).values()}
        if want in names:
            return copy.deepcopy(e)
    for e in list(ENTRIES.values()) + list(LATER.values()):
        for key in e.get("retrains") or {}:
            if want == key.lower():
                return retrain_entry(e, key)
        if e.get("full") and want == e["full"]["key"].lower():
            return full_entry(e)
    raise KeyError(f"no model {name!r} in the registry; its keys: {', '.join(all_keys() + retrain_keys())}")


def full_entry(entry: dict) -> dict | None:
    """The entry of a model's full fine-tune (an entry's "full"), shaped as an entry the lane generator takes: its own run key,
    rank 0, its rate, batch, pace and peak; None when the model has none."""
    f = entry.get("full")
    if not f:
        return None
    e = copy.deepcopy(entry)
    e.update(key=f["key"], lr=f["lr"], batch=f["batch"], grad_accum=f["grad_accum"], max_len=f["max_len"],
             lora={"rank": 0, "alpha": 0})
    e["pace"] = {ln: copy.deepcopy(f["pace"]) for ln in entry["lines"]}
    e["peak"] = {ln: copy.deepcopy(f["peak"]) for ln in entry["lines"]}
    e["full"] = None
    e["is_full"] = True
    return e


def retrain_entry(entry: dict, key: str) -> dict:
    """The entry of a retrain of a model's LoRA (an entry's "retrains"), shaped as an entry the lane generator takes: the model's
    own entry with the retrain's run key and every value the retrain sets (its LoRA settings, pace, peak, comparisons, ...);
    "retrain_of" names the model's key. A retrain carries no full fine-tune and no retrains of its own."""
    r = (entry.get("retrains") or {}).get(key)
    if r is None:
        raise KeyError(f"{entry['key']} has no retrain {key!r}")
    e = copy.deepcopy(entry)
    e.update(copy.deepcopy(r))
    e.update(key=key, full=None, retrains=None, retrain_of=entry["key"])
    return e


def recipe(entry: dict) -> dict:
    """The train keys of the model's v1.2 training, in the order the v1.2 specs carry them (without its inputs, general rows
    and writers, which the lane fills): its own settings and V12_RECIPE. A LoRA whose settings name token rows (lora
    "token_rows": rows of a module trained beside the pairs, train.run token_rows) carries them after its alpha."""
    out = {"model_id": entry["base"]}
    if entry["trust_remote_code"]:
        out.update(trust_remote_code=True, revision=entry["base_revision"])
    if entry["image_settings"]:
        out["image_settings"] = copy.deepcopy(entry["image_settings"])
    out.update(rank=entry["lora"]["rank"], alpha=entry["lora"]["alpha"])
    if entry["lora"].get("token_rows"):
        out["token_rows"] = copy.deepcopy(entry["lora"]["token_rows"])
    out.update(batch=entry["batch"], grad_accum=entry["grad_accum"], max_len=entry["max_len"], epochs=V12_RECIPE["epochs"],
               lr=entry["lr"], seed=V12_RECIPE["seed"], seed_adapter=V12_RECIPE["seed_adapter"])
    return out


def writers(entry: dict) -> list:
    """The writers a v1.2 training of this model accepts: the release rows' writers, then its base (the general rows)."""
    return list(V12_WRITERS) + [entry["base"]]


def smoke_stop(entry: dict, line: str) -> dict:
    """The limits a smoke of this model's line stops at (queue: a train spec's "stop_if"): its steady pace after step 10 above
    1.25 x the pace a v1.2 run measured, or 1.5 x the estimate where none has, or its peak above 85 GB."""
    basis = entry["pace"][line]
    factor = STOP_FACTOR if basis["measured"] else STOP_FACTOR_ESTIMATE
    return {"steady_s_per_step": round(factor * basis["s"], 2), "peak_gb": STOP_PEAK_GB, "after_step": STOP_AFTER_STEP}


def hours(entry: dict, line: str) -> float:
    """The training's hours at the pace basis (the set's steps x the seconds a step), training alone."""
    return round(V12_SETS[line]["steps"] * entry["pace"][line]["s"] / 3600.0, 1)


def version(entry: dict, version: str, line: str = "apache") -> dict:
    """One of the model's trained versions, by its name and line."""
    for v in entry["versions"]:
        if v["version"] == version and v["line"] == line:
            return v
    raise KeyError(f"{entry['name']} has no version {version} on the {line} line")


def todo(keys_: list | None = None) -> list:
    """Every open value: (key, what is open and what decides it)."""
    return [(k, t) for k in (keys_ or list(ENTRIES)) for t in ENTRIES[k]["todo"]]


def describe(entry: dict) -> str:
    """The entry as lines of text: its settings, its paces and limits with their sources, its repos, versions and the open
    values."""
    e = entry
    out = [f"{e['key']}: {e['name']} ({e['family']}, {e['model_type']}) on {e['base']}"
           + (f" at {e['base_revision']}" if e["base_revision"] else ""),
           f"  lines {', '.join(e['lines'])}; " + ("trained" if e["scheduled"] else "training instructions only")
           + f"; lane {e['lane']}",
           f"  LoRA r{e['lora']['rank']} alpha {e['lora']['alpha']}"
           + (f" (token rows {json.dumps(e['lora']['token_rows'])})" if e["lora"].get("token_rows") else "")
           + f"; lr {e['lr']}; batch {e['batch']} x {e['grad_accum']}; "
           f"max_len {e['max_len']}; pictures {e['image_settings'] or 'the processor default'}; reads at batch {e['read_batch']}",
           f"  transformers {e['transformers'] or '>=5 (the queue default)'}; repository code "
           f"{'yes' if e['trust_remote_code'] else 'no'}; install {e['install']['transformers']}"
           + ("; the linear-attention kernels" if e["install"]["fast_path"] else "")]
    for line in e["lines"]:
        p, pk = e["pace"][line], e["peak"][line]
        out.append(f"  {line}: {p['s']} s a step ({p['source']}); about {hours(e, line)} h of training; peak "
                   f"{pk['gb'] if pk['gb'] is not None else '-'} GB ({pk['source']}); "
                   + ("no smoke" if e.get("no_smoke") else f"smoke stops at {smoke_stop(e, line)}"))
    if e.get("retrain_of"):
        out.insert(1, f"  a retrain of {e['retrain_of']}'s LoRA under its own run key"
                   + (f"; no smoke: {e['no_smoke']}" if e.get("no_smoke") else ""))
    f = full_entry(e)
    if f is not None:
        for line in e["lines"]:
            out.append(f"  {line} full fine-tune ({f['key']}): lr {f['lr']}; batch {f['batch']} x {f['grad_accum']}; "
                       f"{f['pace'][line]['s']} s a step ({f['pace'][line]['source']}); about {hours(f, line)} h of training; "
                       f"smoke stops at {smoke_stop(f, line)}")
    for key, r in (e.get("retrains") or {}).items():
        out.append(f"  retrain {key}: " + "; ".join(f"{k} {json.dumps(v)}" for k, v in r.items()
                                                     if k in ("lora", "lr", "batch", "grad_accum", "max_len")))
    if e.get("prepare"):
        out.append(f"  the base is made by the lane's first run {e['prepare']['name']} (tools.{e['prepare']['module']}): "
                   f"{', '.join(e['prepare']['makes'])}")
    if e.get("general_requests"):
        g = e["general_requests"]
        out.append(f"  general text requests from {g['requests']} ({g['n_text']} training, {g['n_held_text']} held)")
    if e.get("train_extra"):
        out.append(f"  train keys beside the recipe: {json.dumps(e['train_extra'])}")
    out.append(f"  repos {json.dumps(e['repos'])}")
    for v in e["versions"]:
        where = "the project repo" if v["adapter"] == PROJECT else v["adapter"]
        rel = f"{v['release']['repo']}@{v['release']['tag']}" if v["release"] else "not released"
        out.append(f"  {v['version']} {v['line']}: {where} {v['subfolder']} ({rel}){'' if v['ready'] else ' [not ready]'}")
    out.append(f"  comparisons: {json.dumps(e['comparisons']) if e['comparisons'] is not None else 'the default set'}")
    if e["space"]:
        out.append(f"  Space: lookers {json.dumps(e['space']['lookers'])}; {e['space']['gb']} GB; {e['space']['tps']} tokens a second")
    out += [f"  TODO {t}" for t in e["todo"]] + [f"  BLOCKER {b}" for b in e["blockers"]]
    return "\n".join(out)


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Print the registry's entries and their open values.")
    ap.add_argument("keys", nargs="*", help="model keys (default: every entry, then every retrain)")
    args = ap.parse_args(argv)
    for k in args.keys or all_keys() + retrain_keys():
        print(describe(get(k)) + "\n")


if __name__ == "__main__":
    main()
