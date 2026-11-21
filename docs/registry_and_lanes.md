# The model registry and the lane generator (from 0.1.68)

Training one more model at v1.2 used to mean drafting its queue by hand from an earlier model's, checking it, writing its
notebook and watching its smoke run. The package now does those steps from one record per model.

## The registry

`qwen_symbolic_thinking/registry.py` holds one entry for each model of the version-1 list: the four Qwen3.5 sizes (the Lucia
line), the two Gemma 4 models (the Ulisse line), North-Micro-Vision, GLM-4.6V-Flash, Molmo2-O-7B, Phi-4-reasoning-vision-15B and
Ministral 3 14B. Each entry says:

- what to train: the base model (and the revision its repository code is pinned to, for the two families whose code lives in
  the model repository), the transformers release its runs need, the LoRA settings, learning rate, batch, maximum length and
  picture budget;
- how long it takes: the seconds a training step takes on each line, measured on a v1.2 run where one exists, otherwise
  estimated from the model's own version-1 pace, and the limits its smoke stops at;
- what exists: its public repos and their tags, the adapters of its earlier versions, and its looker in the testbed Space;
- what is open: every value no record gives yet is left empty, and the entry's `todo` says what decides it. `blockers` says what
  keeps a model's lane from running.

Print it with `python -m qwen_symbolic_thinking.registry` (or name keys: `... registry 08b ge4b`).

| key | model | lines | learning rate | max length | seconds a step (Apache / -nc) | trained |
|---|---|---|---|---|---|---|
| 9b | Lucia (Qwen3.5-9B) | Apache-2.0, -nc | 2e-4 | 2,048 | 9.03 / 8.1 measured | yes |
| 4b | Lucia 4B | Apache-2.0, -nc | 2e-4 | 2,048 | 6.23 / 5.6 measured | yes (released) |
| 2b | Lucia 2B | Apache-2.0, -nc | 2e-4 | 2,048 | 3.18 / 2.87 measured | yes |
| 08b | Lucia 0.8B | Apache-2.0, -nc | 2e-4 | 2,048 | 2.1 / 1.9 estimated | yes |
| ge4b | Ulisse E4B (Gemma 4 E4B) | Apache-2.0, -nc | 2e-4 | 2,048 | 4.1 / 3.7 estimated | yes |
| g12b | Ulisse 12B (Gemma 4 12B) | Apache-2.0, -nc | 5e-5 | 2,048 | 7.0 / 6.3 estimated | yes |
| north | North-Micro-Vision | Apache-2.0 | 2e-4 | 2,048 | 3.6 estimated | yes |
| glm46v | GLM-4.6V-Flash | Apache-2.0 | 2e-4 | 2,560 | 14.4 estimated | yes |
| molmo2o | Molmo2-O-7B | Apache-2.0 | 2e-4 | 2,560 | 11.5 estimated | instructions only |
| phi4rv | Phi-4-reasoning-vision-15B | Apache-2.0 | 2e-4 | 2,048 | 21.8 estimated | instructions only |
| mi14 | Ministral 3 14B | Apache-2.0 | 2e-4 | 2,560 | 31.6 estimated | instructions only |

The Ulisse 12B trains at 5e-5: at 2e-4 its first version-1 arm stopped looking at the picture after a loss spike, and at 5e-5
two seeds kept seeing. Phi-4-reasoning-vision trains on 1,024 picture patches. Molmo2 and Phi-4-reasoning-vision run their
repository's code on transformers 4.57.6; their lanes cannot run yet, because the general-answers tool and the preservation read
do not load a model's repository code.

## A lane from the registry

    python -m qwen_symbolic_thinking.lines 08b both --project-repo <your-name>/<your-project> --out notebooks

writes `queue_<lane>.json` and `<lane>.ipynb`, prints every choice it made, and checks the queue against the project repo's
files (listed from the Hub with your `HF_TOKEN`, or given with `--files`). Nothing is uploaded. Putting the queue in the project
repo as `runs/queue_<lane>.json` and running the notebook starts the lane.

The queue runs, in order:

1. the base model's own answers to ordinary requests, if the project repo lacks them (the keeping objective holds a model near
   its base on the base's own answers, so every model needs its own): `v11a_<key>`, its Apache-2.0 filter `v11a_<key>_by`, and
   the answers about pictures with lettering `v11txt_<key>`;
2. for each line: the smoke (30 steps on the set's first 2,000 items, nothing shipped) and the training, followed by its screen
   and nine reads (the probe, the museum read, the preservation read, the closer and long text reads, the text preservation
   read, the lettering read and two controls);
3. after the first line: the comparison reads the cards need, of the base model and of the earlier versions.

`both` puts the two lines in one lane, the Apache-2.0 line first, so one card never sits idle between them. `apache` or `nc`
builds one line; an `nc` lane alone leaves the comparison reads to the Apache-2.0 lane unless given `--comparisons`.

**The project repo is yours.** Every file the lane reads is a path in it, except the public picture lists of the probe and
museum reads and the version-1 adapters, which live in public repos. These files must be in the project repo before the lane
starts (the generator names each one that is missing):

| path | what it holds |
|---|---|
| sft/v12_rows_a_by.json | the Apache-2.0 line's training set: the rows of release 5 whose pictures may be used that way |
| sft/v12_rows_nc.json | the -nc line's training set: every row of release 5 |
| sft/v11_rows_a.json | the rows whose pictures the general answers ask about |
| sft/v11_rows_a_held.json | pictures held out of training, for the held general answers |
| screens/inputs/screen_v1.json | the screen's 60 pictures and 40 specifications |
| licenses/coco_photo_licenses.json | each COCO photo's own license, for the Apache-2.0 filter |
| text/anchor_pool_v1.json, text/anchor_held_v1.json | pictures with lettering, for the general answers' text bank |
| text/text_held_v1.jsonl, text/text_held_v1_deep.jsonl | the 300 held pictures with lettering, at a closer and a long look |
| text/notext_control_v1.jsonl | 200 photos without lettering, for the controls |

Earlier versions trained in a project repo other than yours are left out of the comparisons, and the generator says so.

## The smoke stops itself

A smoke carries its limits, `"stop_if": {"steady_s_per_step": x, "peak_gb": y, "after_step": n}`. When it ends, its pace over
the steps after step n (read from its own step log) and its peak GPU memory are compared with them. Past either, its marker says
failed, with the reason in plain words. The registry sets the limits from one rule: 1.25 times the measured or estimated
seconds a step, 85 GB, after the first ten steps.

Its training says `"waits_for_runs": [<the smoke's name>]`: it starts only once the smoke's marker says done. If the smoke
failed, the training is held back (never started, never marked) and the lane goes on with whatever does not wait for the smoke.
The hold is read from the smoke's marker, so a fresh runtime keeps it. A spec without these keys runs as before, and every
generated spec says it needs 0.1.68, so a lane on an older package does not take it.

## The progress file

A training whose spec says `"progress_every_min": 10` (the generator writes it into every training) writes
`runs/progress/<run>.json` to the project repo at its first log step, every ten minutes after that, and when it ends: the step
and the total, the loss and the keeping term, the seconds a step (overall and since the last write), the peak memory, the last
saved state's step and the last shipped step, and the time. The upload runs on a thread of its own, one try; a write that fails
is printed and skipped, and the training never waits for it. Read every run's line with

    python -m qwen_symbolic_thinking.tools.progress <your-name>/<your-project> [<run> ...]
