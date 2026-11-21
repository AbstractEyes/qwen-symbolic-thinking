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
failed, with the reason in plain words. The registry sets the limits from one rule: 1.25 times the seconds a step a v1.2 run
measured, or 1.5 times the estimate where none has measured it yet (an estimate carries more doubt), 85 GB, after the first ten
steps.

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

## The late save (from 0.1.69)

A training with resume on saves its whole state every 2 hours and once more when the machine has run for 23.5 hours (it was
23.0): a Colab runtime ends at about 24, so the half hour later halves the steps the next runtime trains again. It is a setting:
`"resume_before_h"` among a training's keys sets another hour, `null` none (`resume.LATE_SAVE_H` is the default).

## The reads of a line, one command (from 0.1.69)

When a line's training and its reads are done, its checks are read from the project repo's records; nothing runs a model and
nothing is written to any repo:

    python -m qwen_symbolic_thinking.reads <model> <apache|nc> bars --project-repo <your-name>/<your-project> --out reads
    python -m qwen_symbolic_thinking.reads <model> <apache|nc> text --project-repo <your-name>/<your-project> --out reads

The run names are the lane generator's: the line's training and the nine reads after it, the comparison reads of the base model
and of the earlier versions (the registry's, with the reads earlier lanes made under other names), and for an -nc line its
Apache-2.0 sibling's reads. `--dry-run` prints the runs each read takes and reads nothing.

**bars** reads the training against its line's yardstick, an arm of the same line trained without the keeping objective
(the registry's `YARDSTICK`; `--yardstick-run`, `--yardstick-tag` and `--yardstick-repo` name another), and writes
`bars_<read tag>.txt` (the table and PASS or NOT READY) and `.json` (every value read):

| bar | pass mark |
|---|---|
| T1 full-form format, full rows | the screen's full-form answers in the row's format, and whole, at least 0.95 |
| T2 said, backed | the screen's hint said and backed within 0.03 of the yardstick's (above the band is never a miss) |
| T3 probe | the held-out probe's prompts that name the picture's kind within 10 of the yardstick's (above, never a miss) |
| K1 drift, picture and text | the drift from the base per answer token at most half the yardstick's |
| K2 row form | general answers taking the row's form: the excess over the base at most 0.02 and at most half the yardstick's |
| K3 ended | general answers that end on their own at least the base's less 0.02 |

The probe's picture kinds come from the picture list the probe ran on (a public file of about 75 MB, kept in the Hub cache);
`--probe-manifest` takes a local copy.

**text** writes seven reads, each as `.json`, `.md` and `.txt` (the table as printed):

| output | what it reads |
|---|---|
| `text_read_v12_<key>_<by,nc>` | how often the rows of the closer look name the text the picture shows (the whole row, the chain, the caption, the prompt, the questions; words named; a chance floor against another picture's words) |
| `text_read_v12_<key>_<by,nc>_deep` | the same for the long look |
| `text_read_lettering_ask_<key>_<by,nc>` | the same for the closer look with the lettering sentence |
| `keep_text_read_v12_<key>_<by,nc>` | how often an answer to "what does the text say" names it, beside the base's answer |
| `lettering_ask_read_text_<key>_<by,nc>` | the rows that write lettering (a `say` thought, a quoted pair) and keep the row's shape |
| `lettering_ask_read_ctl_<key>_<by,nc>` | the same on photos without lettering (lettering a picture does not have, shown verbatim) |
| `lettering_quote_precision_<key>_<by,nc>` | whether the quoted words are the picture's own (all, some, none), beside another picture's |

The held pictures' manifest is `text/text_held_v1_names.jsonl` in the project repo by default (`--manifest` names another): the
300 held pictures with lettering (`text/text_held_v1.jsonl`) with each picture's name screen added, flagged where its shown words
may hold a person's name, a credit or a web address. Every text read then also reads the flagged and the clear pictures apart.
Without it, `--manifest repo:text/text_held_v1.jsonl` gives the same numbers without that split. A comparison read the project
repo has no marker for is left out and said; a missing read of the line itself stops the command before it reads.

Each read is also a command of its own on any runs, e.g. `python -m qwen_symbolic_thinking.tools.text_names --project-repo
<repo> --manifest <manifest> --out <stem> <run> [...]` (likewise `tools.keep_text_names`, `tools.lettering_ask`,
`tools.quote_precision`, `tools.bars`), and every one takes `--records <folder>` to read a local copy of the project repo's
paths instead of the Hub.

## TRAINING.md for a model repo (from 0.1.69)

    python -m qwen_symbolic_thinking.training_md <model> --out TRAINING.md

writes the page a model repo carries on how to train its v1.2: what v1.2 is, the model's lines and their licenses, its settings,
its pace, peak memory and hours on a G4 (each measured or estimated, with the source), the files a project repo must hold and
where each comes from publicly, the steps from a fresh account, what the lane does, and the bars with their pass marks and our
numbers. For the models we do not train it says so, with what stops their lanes today. A file with no public source yet is
printed as `[PLACEHOLDER: ...]` and listed at the end of the command's output with what would fill it. Nothing is uploaded.
