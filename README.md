# qwen-symbolic-thinking

Teaching a Qwen vision-language model to think about a picture in writing, the way the rows of
[thought-symbolic-image-captions](https://huggingface.co/datasets/AbstractPhil/thought-symbolic-image-captions) do:
a short chain of plain-language thoughts, one step each, that ends in a caption.

```
see: One man in an apron stands in front of two ovens.
place: A working kitchen; pans hang on the wall behind him.
mood: A kitchen at work reads as warm and busy.
caption: A man in an apron standing by two ovens in a kitchen.
```

The steps a chain may use, in the order a careful looker takes them: see, name, place, count, relate, form, light, colour,
mood, mean, intend, story, compare, change, check, say. A quick look holds 1 to 2 thoughts, a closer look 3 to 5, a long look
6 to 10.

Weights, screens and results: [AbstractPhil/qwen-symbolic-thinking-image-training](https://huggingface.co/AbstractPhil/qwen-symbolic-thinking-image-training).

## The behaviour screen

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbstractEyes/qwen-symbolic-thinking/blob/main/notebooks/screen.ipynb)

Before any fine-tune: can the base model do the behaviour when asked plainly? `notebooks/screen.ipynb` runs it on a GPU
(Qwen3.5-9B and Qwen3.5-4B) over about a hundred rows in three prompt forms:

| form | what the model gets |
|---|---|
| `bare` | the picture and the instructions (the line form, the steps, the depth's counts, the rules) |
| `shown` | the same, with two worked examples written for other pictures |
| `spec` | no picture: a specification of one as records, to be thought through as if seen |

Each answer is parsed back into thoughts and read against the row it answers:

| read | meaning |
|---|---|
| `format_ok` | at least one `step: text` line and a `caption:` line |
| `in_range` | the number of thoughts fits the depth |
| `spec_recall` | the share of the specification's facts present in the text (numerals read as words) |
| `spec_recall_of_rows` | the same read over the dataset's own chains, the ceiling |
| `rows_with_gloss` | rows with a thought that defines a word instead of looking at the picture |
| `rows_with_remark` | rows with a thought about the task instead of the picture |
| `step_jaccard` | overlap of the steps used with the row's steps |
| `hint_said`, `hint_backed` | the picture's own captions against the thoughts and the caption: the share of the captions' content words the chain says, and the share of the chain's content words the captions back |
| `wordings_ok`, `questions_ok`, `full_ok` | for the whole-row forms: the depth's wordings present with a caption; the number of questions fits the depth; both, with every question well formed |
| `s_per_row`, `tok_per_s` | the time cost on the card |

Results land on the Hub under `screens/<stamp>/`: `results.jsonl` (one line per answer, with the text, the parse and the
scores), `summary_<model>.json`, `summary.json`, `summary.md`, `env.json`.

The picture rows come from COCO 2017 images, named by id in the dataset's seeds and fetched from the COCO site at screen
time; nothing is redistributed here. The specification rows are the dataset's own drawn rows.

## Install

```
pip install git+https://github.com/AbstractEyes/qwen-symbolic-thinking@v0.1.40
pip install "qwen-symbolic-thinking[screen] @ git+https://github.com/AbstractEyes/qwen-symbolic-thinking@v0.1.40"   # with torch, transformers 5, accelerate, peft
```

Qwen3.5 needs transformers 5 (the model code lives under `transformers.models.qwen3_5`); the preflight says so if it is missing.
Colab's torchao 0.10 breaks PEFT's import (PEFT 0.21 wants torchao 0.16 or newer, or none): the notebooks' install cell removes
torchao when PEFT fails to import, and `qwen_symbolic_thinking.checks.run()` reports it. The linear-attention kernels
(causal-conv1d, flash-linear-attention, triton) are optional and all-or-nothing; the same checks report them, and every run
prints whether transformers fell back to its reference implementation.

## The fine-tune

`qwen_symbolic_thinking.train.run` trains a LoRA adapter (rank 32, alpha 64, on the language model's attention,
linear-attention and MLP projections; the vision tower frozen) with plain Adam and a cosine schedule, the loss on the
assistant turn only. Each training item is taught under the one-line prompt forms it carries:

| form | the prompt | the answer |
|---|---|---|
| `short` | the picture, "think it through in writing, a closer look, ending with a caption" | the chain and the caption |
| `full` | the picture, the same, "then write a caption and a prompt or tags, and the questions a careful looker would still ask" | the whole row: the chain, the wordings in order (caption, prompt, tags, critique), then one `question: what -> answer [record, chain or unknown]` line each |
| `spec_short`, `spec_full` | the specification as records instead of a picture | the same two answers |

The adapter ships to the Hub during training under `lora/<model>/r<rank>/<stamp>_<tag>/`, and the trained model is read with
the same screen in the trained forms (`full`, `short`, `spec_full`). Rank 0 is a full fine-tune: the language model's weights
train in fp32 under bf16 autocast (plain Adam at a full fine-tune's rate, 1e-5 to 2e-5; the vision tower frozen as under the
LoRA), and the model ships in bf16 with its processor under `full/<model>/<stamp>_<tag>/final`, which every loader takes as the
model id `hub:full/<model>/<stamp>_<tag>/final`. A queue spec that says `"needs": "0.1.18"` waits for a lane with that package.

`keys` (from 0.1.40) sets how the answers' question keys and the specification forms' records are written. `"record"` (the
default) keeps the dataset's record names (`thing position`; a relation's parts `rel1 a`, `rel1 rel`, `rel1 b`). `"relation"`
writes every key as one word (`thing_position`), letters the things a row's relations join `relation_a`, `relation_b`,
`relation_c`, ... in order of first appearance (a thing that recurs keeps its letter) and names a relation's word by its two ends
(`relation_a_b`); the specification forms then list the records as `- key: value` under the same keys, so a key and its value are
the same tokens in the prompt and in the answer's `question:` line. A train spec's `keys` reaches its screen. `max_steps` sets a
run's length in steps instead of epochs (the cosine schedule and its warmup span those steps).

`qwen_symbolic_thinking.synth.generate` makes synthetic picture items: the base model, shown two worked examples, thinks
through pictures no row was written about; a chain passes the audit when it holds the form and the depth, defines nothing,
remarks on nothing, strays nowhere, and both says and is backed by the picture's own captions (0.2 each); the passing chains
ship as short-form items under `synth/`. `qwen_symbolic_thinking.queue.run_queue(lane)` lets a card work unattended through
`runs/queue_<lane>.json` on the Hub (fine-tunes, each followed by its screen; synthetic generation; throughput sweeps;
extractions), leaving `runs/done/<name>.json` per run; `notebooks/lane1.ipynb` to `lane3.ipynb` run it. A train spec may name
runs under `after`: extract specs (the held-out probe split, the museum pictures) that run as soon as its screen ends, reading the
adapter it trained with the model it trained, so one queue entry trains, screens and reads an arm.

## Training rules

- Every training item names its writer: the model, or the people, whose text the item teaches (`writer`). Rows written by an
  outside row-writing system carry theirs (`rows.build_api_rows` copies it from the row), and the general rows name the base model
  that answered them.
- A run lists the writers it accepts (`writers`). An item whose writer is missing, or not on the list, stops the run before
  anything loads (`keep.check_writers`); the summary records the items by writer. With no list nothing is checked.
- The optimizer is plain Adam with a cosine schedule and a short warmup, and no weight decay. Whatever regularization a run needs
  lives in the loss, where it can be read, not in the optimizer.
- Every keeping term is read against the base model through the adapter's off switch: the same weights with the LoRA disabled,
  no gradient, so no second copy of the model is held. A run with a keeping term is therefore a LoRA run.
- Every trained adapter is read for preservation beside the task reads, at each scale it may ship at: `tools.keep_read` loads the
  adapter without merging it, scales every LoRA layer by the scale, and reads the drift from the base on held general rows. A
  queue train spec names it under `after` as a `tool` entry, which fills in the model and the adapter the training shipped.
- The adapter's starting matrices follow the run's seed when `seed_adapter` is set (otherwise only the Trainer is seeded, and every
  process draws its own starting matrices), so two runs of one recipe start from the same adapter.

## Keeping the base model's behavior

Teaching one answer form moves more than the form: the form starts to appear where it was not asked for, and the model's other
answers drift away from the base model's. `keep.py` holds the adapted model near the base with three terms, each a mean over the
answer positions of its micro-batch, each weighed by a `train.run` argument:

| term | weight | what it computes |
|---|---|---|
| general KL | `general_kl` | `lambda_g * KL( p_base \|\| p_tuned )` at the answer positions of the general rows; the base side runs with the adapter off and no gradient |
| target-masked KL | `target_kl` | `lambda_t * KL( p_base^-y \|\| p_tuned^-y )` at the answer positions of the training rows, where `p^-y` is the next-token distribution with the training token `y` removed and the rest renormalized; positions where the base already gives `y` more than 1 - 1e-4 are left out |
| rehearsal | `rehearsal` | `mu * CE` on the general rows, the base's own answers as targets (no base pass at training time) |

With any term on, the loss of a step is `CE(training rows) + general_kl * KL_g + target_kl * KL_t + rehearsal * CE_g`; with every
weight at zero the plain Trainer runs, unchanged. The target-masked term keeps the base's ranking of the alternatives at every
position without opposing the cross-entropy on the training token (Xu et al., "Mask the Target: A Plug-and-Play Regularizer
Against LoRA Forgetting", arXiv 2605.29498). Rehearsal on the base model's own outputs is the approximate regularized replay of
Riemer et al., "The Effectiveness of Approximate Regularized Replay for Efficient Supervised Fine-Tuning of Large Language
Models" (arXiv 2512.22337), and general rows made by the model itself follow Wang et al., "Two-stage LLM Fine-tuning with Less
Specialization and More Generalization" (arXiv 2211.00635), whose second stage trains on the model's own outputs. For the
multimodal case, Li et al., "Fine-tuning MLLMs Without Forgetting Is Easier Than You Think" (arXiv 2603.14493) report that
forgetting in a vision-language model is largely recoverable and shrinks with a small adapter and rehearsal.

**General rows** are requests that do not ask for the row form, answered by the base model itself: items of kind `anchor`, with a
`prompt`, an `answer`, a `writer` and, for a picture, its locator. `qwen_symbolic_thinking.tools.anchors` writes them on a GPU with
no adapter, greedy decoding and thinking off:

| rows | the request | the answer |
|---|---|---|
| picture rows | one of 22 everyday questions about a training picture ("What is the main subject of this picture?", a caption, a poem, a question in French), one drawn for each picture, the pictures taken round-robin over their sources | the base model's, with the picture |
| text rows | a request the base model writes itself: the chat template's opening of a user turn is sampled from (temperature 1.0, top-p 0.95, seeded) until the turn ends, kept at 20 to 500 characters, repeats dropped (Xu et al., "Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing", arXiv 2406.08464) | the base model's |
| held rows | the same two kinds, for pictures and requests kept out of training; each held picture also gets one request for a detailed description | the base model's |

An answer that reaches its token cap never ended, so it is dropped. The tool ships `anchors/<tag>/anchors.json` (for training),
`held.json` and `summary.json` (counts, seconds a row, tokens a second, the drops). `train.run(anchors=..., anchor_batch=...)` draws
`anchor_batch` general rows beside every micro-batch of training rows, in a seeded order, cycling.

**Three reads** of how far an adapter moved the model, on the held general rows (`tools.keep_read`; `keep/<stamp>_<tag>/rows.jsonl`
keeps every row at every scale):

| read | what it measures |
|---|---|
| drift | teacher-forced on each held row's own answer: the mean KL(p_base \|\| p_tuned) over the answer positions (per row; reported as the mean, the 50th, 90th and 99th percentile, and the mean at each of the first eight positions), and exp(mean tuned NLL) / exp(mean base NLL) |
| takeover, stop, agreement | greedy answers to the held requests: the share written in the row form, the share that end before the token cap, the answer's length against the base's, and the word overlap (token F1) with the base's answer |
| detail inside the row form | the full row form asked of each held picture the base also described in detail: the share of the base's content words the row still holds (recall), and the row's length |

The scale of a read multiplies every LoRA layer's scaling, so a scale of 0.5 reads the adapter at half strength and 0 reads the
base itself (where the third read is the base model writing the row form on its own). `screen.run` and `extract.run` take the same
`adapter_scale` (default 1.0) and merge the adapter at that strength, interpolating between the base and the adapter as trained
(the delta form of weight interpolation, Wortsman et al., "Robust fine-tuning of zero-shot models", arXiv 2109.01903). Rows
written by outside writers become items through `rows.build_api_rows` (the writers and licenses a run accepts, the pictures held
out for the screens and the held-out reads, each drop counted), and `tools.build_rows` runs it inside a queue. With `hold_out`,
a seeded draw of the training pictures leaves the training set for a file of its own (`sft/<tag>_held.json`), so the held rows
of `tools.anchors` (`held_pictures` takes one input file or a list) can be pictures from the same sources the adapter trained on.

**Rows that are not public, and pictures that may not be shared.** A training set built from rows that are not public must not
ship to a public repo. Setting `QST_REPO` before the package is imported points every `hub:` path, queue, marker and shipped
result of that runtime at another repo, a private twin of the project repo; the notebooks set it in their queue cell. Pictures
that a row-writing system keeps in a Hugging Face storage bucket (a mutable store) are copied once into parquet shards of the
project repo by `pack.pack_staged`, with a manifest of `hf_parquet` locators; `tools.build_rows` does it first when given `pack`
(`{"bucket": "owner/name/folder", "dirs": {source: [folders]}}`), so the pack and the build read one snapshot of the rows. A queue
spec can name `waits_for` paths: it waits, unmarked, until another lane has built them.

## Extraction

`qwen_symbolic_thinking.extract.run` has the trained model write whole rows for the pictures of a manifest, the list kept at
[AbstractPhil/qwen-thinking-images](https://huggingface.co/datasets/AbstractPhil/qwen-thinking-images). A manifest row names its
picture by a locator (`url` with fallbacks, `hf_parquet` row of a shard, `hf_file`, `zip_member`, `url_archive` member of a tar
or zip fetched once, `hf_dataset_row` keyed in a split), the depth it is asked at (`depth_plan`: quick, closer or long) and
whatever captions its source carries. Pictures are fetched in groups (a shard, a zip or an archive downloads once), the next
chunk while the current one generates, each bounded to 1,024 px on its longer side; a batch that runs out of memory is retried
in halves. Each answer is parsed, scored and audited the way the synthetic rows were, with the grounding gates only where the
source has captions (the floors mean different things per caption pool, so a volume run sets them to 0 and selects offline);
the rows, kept and rejected alike, the pictures that did not answer (`missing.jsonl`, with why) and a summary ship to
`extractions/<stamp>_<tag>/` in the dataset repo. `extract.training_items(rows, manifest, floors)` turns a run's kept rows into
training items (the whole row under the `full` form, the captions as the hint), and the trainer fetches pictures by the same
locators.

```python
from qwen_symbolic_thinking.extract import run
run(manifest="hub:AbstractPhil/qwen-thinking-images/pilot_v1.jsonl", adapter="AbstractPhil/qwen3.5-9b-think-symbolic-image", tag="pilot")
```

`helpers=("caption", "tags")` (from 0.1.39) adds helper lines after the ask, for each picture that has them: `Helper caption:`
with the picture's plain-text captions, then `Helper tags:` with the booru tagger's structured caption written as one line
(`girl (solo) · pants (blue) · jeans · doing: standing`, `extract.tag_line`). Each row records the lines it was given. The
grounding read is against the same captions, so a helped extraction's said and backed are not comparable to an unhelped one's.

```python
from qwen_symbolic_thinking import chain
parsed = chain.parse(model_text)          # {thoughts: [{step, text}], wordings: {caption: ...}, questions: [...], stray: [...]}
scores = chain.score(parsed, row)         # the reads above, for one row of the dataset
text = chain.render(row, full=True)       # the whole row as the model is taught to write it
```

Tests: `pytest tests` (no GPU needed).

## Captions and tags for a zip of photos

`captions.run` gives every photo of a zip archive a plain caption from a stock VLM and booru tags from a tagger (`booru`: the
animetimm and WD v3 taggers, run the way their authors run them). The ask may carry the photo's own description as a helper
line (`{original}`, from a JSON of file name to description); the row keeps the file name, the caption, the tags with their
scores, the tags a reviewer's filter keeps (a leave-out list and a floor on each tag's published precision, clothing exempt),
and `copy5`, the share of the caption's five-word runs found word for word in the description (the description itself is not
written). The VLM runs a batch in one generation (halved on an out-of-memory), the next chunk of photos decodes meanwhile, and
the rows ship to `extractions/<stamp>/` while the run goes; `resume=` continues a shipped folder after a lost runtime.
Notebook: `notebooks/deepfashion_captions.ipynb` (DeepFashion-MultiModal's 44,096 photos, the stock Qwen3.5-4B and the
EVA-giant tagger).

### Body angles: a 3D body fit beside the caption

`pose` fits a 3D body to the photo with SAM 3D Body (Meta; [facebookresearch/sam-3d-body](https://github.com/facebookresearch/sam-3d-body),
pinned to commit `b5c765a`; its body rig, [MHR](https://github.com/facebookresearch/MHR), is Apache-2.0) and reads three frames from
the fit's named 3D keypoints: the hips (the hip joints), the chest (the shoulder line) and the head (the ear line and the eyes).
Each frame's columns are the subject's left, up and forward in the camera's frame; the hips and the chest share the torso's axis,
so their difference is a pure twist. With R the frames and F the frame of a person facing the camera upright:

```
hips, chest, head against the camera   F^T R_hips,  F^T R_chest,  F^T R_head
twist (the shoulders against the hips) R_hips^T R_chest   (a turn about the torso's axis)
the head against the chest             R_chest^T R_head
each one as turn / nod / tilt          intrinsic Y-X-Z: turn = atan2(R[0,2], R[2,2]), nod = asin(R[1,2]), tilt = -atan2(R[1,0], R[1,1])
```

A turn is + toward the subject's left (a person facing the camera reads 0, the back of a person 180), a nod + up, a tilt + toward
the subject's left shoulder; the torso's lean is its axis against the camera's vertical, forward and to the side. `pose.words`
turns a record into a pose line in the picture's own terms ("Pose (3D body fit): the body is in profile, facing the right of the
picture; the head is turned 30 degrees toward the left of the picture"), naming only what stands out (a twist of 15 degrees, a
head turn of 15, a nod of 15, a tilt or lean of 10). The picture's terms because a captioner reads "the subject's left" as the
picture's left: with a line in the subject's terms, the stock Qwen3.5-4B named the wrong side of the picture in 17 of the 22
captions that named one (49 DeepFashion photos turned 45-135 degrees). In `captions.run`, `pose_fit=pose.WEIGHTS` adds each photo's record (the angles, the three frames, the 70 keypoints) to its
row, `pose_ask=True` puts the pose line in the ask (`{pose}`, or a `Helper pose:` line), `pose_skip` leaves photos out by file name
(a flat lay), and `model_id=None` runs the fit without captions. `captions.append_pose(caption_rows, pose_rows, out)` joins a
caption run with a pose run by file name: each row gains the pose record (rebuilt from the stored keypoints with the current
angles), the pose sentence, and `caption_pose`, the caption with the fit's own sentence appended (no captioner in between). Without a person detector each picture is one person's box, and
the camera has the fit's default field of view: right for a studio photo of one person.

Photos of several people: `pose.predict_many(fit, picture, boxes)` fits every person box of a photo in one pass (up to 16 boxes a
pass) and keeps each record in its box's place (None where no body was fitted), with the box and the 70 keypoints in the
picture's pixels (`keypoints_2d`). `pose.apart(a, b)` says how far two fits of one photo lie apart: the median distance between
their body points (face, shoulders, elbows, wrists, hips, knees, ankles) over the larger fit's size. Two boxes drawn around one
person give fits a few hundredths apart, two people tenths or more, which is how the boxes of a dataset that draws one box per
action (HICO-DET) are merged into people. `notebooks/hico_det_bodies.ipynb` runs it on 1,000 HICO-DET photos on Colab.

The fit's inference path needs `roma pytorch-lightning yacs omegaconf timm einops braceexpand` (`pose.NEEDS`); its repository is
not a pip package, so `pose.install()` fetches the pinned commit once and puts it on the path. The weights
(`facebook/sam-3d-body-vith`, 631M parameters, the default: its backbone code is in the pinned repository; `-dinov3`, 840M,
scores the same on the published benchmarks but builds its backbone from GitHub through `torch.hub` at every load, unpinned) are
gated: request access on the Hub with the account whose `HF_TOKEN` the run uses. They come under the SAM License: free use, copies and changes, outputs given as they are, a copy of the
license with any redistribution of the model, its use acknowledged in research publications, and the end uses it forbids (military
and warfare, nuclear, espionage, weapons, work under arms-export controls).

### Helper facts two describers agree on

`garments.py` turns a caption, its booru tags and the fit into facts a helper line can carry. A garment color enters only where
the caption and the tags give the same base color to the same kind of garment (top, bottom, one-piece, outer layer, shoes):
`garments.agreed(caption, tags)`. Shoes or an outer layer enter only where both name one: `garments.worn(caption, tags, slot)`.
The camera angle comes from the hips' turn: `garments.angle(turn)` gives frontal up to 20 degrees, three quarter up to 70 and
profile up to 110; past 110 the back is to the camera and no angle is given. `garments.records_line(spec, colors, angle, worn)`
writes a specification's records with all of these as one line (`upper name: t-shirt; upper color: white; ...`), in US spelling.

### Lighting: a trial of two published readers

`lighting.trial(images_zip, files, out_repo=...)` asks two models about each photo's light. ShotVL-7B (Vchitect; a Qwen2.5-VL-7B
fine-tune for cinematography, Apache-2.0) answers its benchmark's two lighting questions, "What's the lighting type of this
shot?" and "What's the lighting of this shot?", once open and once as a choice over every option the benchmark uses
(`lighting.TYPES`, `lighting.LIGHTS`). CinemaCLIP (OZU-Technology; CLIP ViT-B/32 with classifier heads, OpenRAIL) scores its
lighting heads from its released ONNX encoder and label schema: cast (soft / neutral / hard), contrast (low / neutral / high),
direction (ambient, back, front, left, right, top, under; every label over its suggested threshold) and the edge-light and
silhouette likelihoods. The released encoder gives 101 probabilities (23 heads), which only the package's own schema (1.0.0, read
from its repository at a fixed commit) names; the schema beside the encoder on the Hub (2.0.0) has 315 labels, so the loader takes
whichever schema fits the encoder's output. One row a photo; the folder ships to a Hub dataset.

`lighting.heads_run(images_zip, out_repo=...)` runs CinemaCLIP alone over every photo of a zip (batched when the encoder takes a
batch) and writes, beside the heads, the photo's mean luminance and the `shot light` record values the heads support
(`lighting.shot_light`): silhouette; the key light's side (from the left / right / above, or backlit); rim light; low key
(high contrast, mean luminance under 0.35) or high key (low contrast, over 0.65). Left and right are the picture's. The head's
`under` label names no side (0.1.50): on FFHQ's smoke sheet the faces it read as lit from below were frontal flash shots. Every row
keeps all the head scores, so the record values can be recomputed from a shipped run. Notebook cell C6 of
`notebooks/deepfashion_captions.ipynb` runs the trial on 100 DeepFashion photos.

### FFHQ

`notebooks/ffhq_captions.ipynb` gives the 70,000 faces of FFHQ the same treatment: `captions.webdataset_zip` repacks the WebDataset
shards of a Hub copy (gaunernst/ffhq-1024-wds) into one zip without re-encoding; `captions.run` captions and tags them with
`captions.PORTRAIT_ASK` (`repo_folder` picks the dataset folder the rows ship to, `max_side` can shrink large photos);
`lighting.heads_run` adds the lighting. A 100-face smoke (C3) shows a sheet of faces with their readings before the full run, and
C3b tries reworded asks on the same faces. FFHQ is CC BY-NC-SA 4.0 and not for the development of facial recognition.

### Absences in captions

Asked for "any eyewear, headwear or jewelry" and "any print or text", the stock captioner also says what is not there ("no eyewear,
headwear, or jewelry", "with no visible print or text"): 68 and 36 of 100 FFHQ smoke captions, 30% of the DeepFashion ones.
Rewording the ask did not remove them (the best of three wordings still wrote one in 55 of 100 captions, with longer captions and
fewer accessories named). `absences.drop(caption)` removes those statements for the two families and leaves every other negative
("no visible smile"); `absences.left(text)` lists the ones still there. On the 400 smoke captions of four asks it changes 264 and
leaves 1; on the 42,544 DeepFashion captions it changes 12,684 and leaves 132; no cleaned caption gains a word it did not have
(0.1.50 matched "text" inside "texture" and broke 13 DeepFashion captions; every pattern word now ends at a word boundary), and a
caption with nothing to drop comes back byte for byte. Release tables keep the raw caption beside the clean one. On FFHQ's 70,000 captions
(0.1.52) it changes 58,032 and leaves 1,178, mostly exceptions that name what is there ("no jewelry beyond the earrings"), kept on purpose.

## Reading an arm

`qwen_symbolic_thinking.tools` reads a trained arm's runs from the Hub without a GPU. Files live under a data root (`--data`, else
`QST_DATA`, else `./data`) and are fetched when missing (`HF_TOKEN` in the environment while a repo is private; `--offline` reads
what is there).

```
python -m qwen_symbolic_thinking.tools.probe_read P_probe_med4          # the held-out probe: 500 NGA pictures, 100 a kind
python -m qwen_symbolic_thinking.tools.museum_read P_museum_med4        # 200 museum pictures against their catalogue lines
python -m qwen_symbolic_thinking.tools.museum_mismatch P_museum_med4    # one kind row by row (works on paper flagged)
python -m qwen_symbolic_thinking.tools.screen_conflicts <stamp> <stamp> # a screen's conflicts beside its rulers
```

The probe counts, per kind, the rows whose prompt line's first term names the kind the museum's own classification gives; the
`probe` split is held out of every training set, so it is the medium term's ruler. The museum read uses the catalogue line instead
(`tools.medium.expected`): its painting row understates a model, because watercolours and gouaches on paper are named drawings, the
National Gallery's convention, and a few object and print lines match the painting pattern; `museum_mismatch` shows which. The
conflicts counted on every row: an unknown question, a thought about light, a stock mood word (gentle, quiet, everyday, unhurried,
calm, plain), a defining pattern (means, is a, refers to) in a thought.

## Mixing the lookers

When several lookers have written the same pictures, `tools.mix` builds the mixes from the pool of their rows, with no reference
text: per picture the agreement pick (the row, among those that hold the form, whose words the other lookers' rows share most:
a weighted mean Jaccard), the caption and the prompt line the others agree with most, a vote of the prompt lines' first terms,
and the pools of every distinct caption and prompt line for caption sets that feed a picture generator (there the whole words boy,
boys, girl and girls become young man, young men, young woman and young women, counted). Lookers are weighed by family by default,
each family one share, so a family of many sizes trained on the same rows does not outvote another by its member count. It writes
one table per looker in the `rows` columns, the agreement pick as a split of its own, the mixed table, and a read that sets every
looker, the picks, a random looker and the oracle side by side (said / backed by source with standard errors, the probe's kinds,
the pick's backing by agreement decile). It runs as a queue `tool` run on a lane (the rows are many) or from the command line:

```
python -m qwen_symbolic_thinking.tools.mix lookers.json --manifest <manifest> --out mix_out [--ship]
```

## Releases

- **Lucia v1** (2026-10-31): the root adapter of
  [AbstractPhil/qwen3.5-9b-think-symbolic-image](https://huggingface.co/AbstractPhil/qwen3.5-9b-think-symbolic-image), also at the
  tag `lucia-v1`; the original release (2026-10-30) is kept under `legacy/E_all_2ep/` and at the tag `legacy-e-all-2ep`. The card
  gives her held-out reads. `extract.run` and the lane notebooks use the root adapter by default.

## Lucia at every size

The same recipe is trained on each dense Qwen3.5 base, one notebook and one queue a size, each queue entry training, screening
and reading its arm on the two held-out sets (`after`); the sizes that fit one card also get a full fine-tune of the same
recipe (rank 0). Each size is released as its own repositories, the adapter and the full fine-tune, with one card
(`docs/release_card_template.md`: the loading of either variant side by side, the reads, the training properties) and its own
reads; the sizes differ in what they can hold, and the cards say so.

| size | base | notebook | queue | notes |
|---|---|---|---|---|
| 9B | Qwen/Qwen3.5-9B | `lane1.ipynb` to `lane3.ipynb` | `runs/queue_lane<n>.json` | Lucia v1 |
| 4B | Qwen/Qwen3.5-4B | [`lucia_4b.ipynb`](https://colab.research.google.com/github/AbstractEyes/qwen-symbolic-thinking/blob/main/notebooks/lucia_4b.ipynb) | `runs/queue_lucia_4b.json` | the recipe verbatim |
| 2B | Qwen/Qwen3.5-2B | [`lucia_2b.ipynb`](https://colab.research.google.com/github/AbstractEyes/qwen-symbolic-thinking/blob/main/notebooks/lucia_2b.ipynb) | `runs/queue_lucia_2b.json` | the recipe verbatim; answers without thinking by default, as every Lucia is trained and read |
| 0.8B | Qwen/Qwen3.5-0.8B | [`lucia_0.8b.ipynb`](https://colab.research.google.com/github/AbstractEyes/qwen-symbolic-thinking/blob/main/notebooks/lucia_0.8b.ipynb) | `runs/queue_lucia_0.8b.json` | two arms, the recipe verbatim and a set tweaked for a small model (three epochs, lr 3e-4); the release takes the better by the held-out reads |

## Other families: Gemma 4 (the Ulisse line)

The released Gemma 4 adapters are the Ulisse line ([Ulisse E4B](https://huggingface.co/AbstractPhil/gemma-4-e4b-think-symbolic-image),
[Ulisse 12B](https://huggingface.co/AbstractPhil/gemma-4-12b-think-symbolic-image); tag `ulisse-v1`), beside the Lucia line on
Qwen3.5. The trainer, the screen and the extraction are written against the chat template, not against one model: the prompt is rendered
with the template's own thinking switch off, the training text is the generation prompt + the row + whatever the template writes
after an answer (`train.turn_end`, read from the template itself: `<|im_end|>` for Qwen3.5, `<turn|>` for Gemma 4), and the
LoRA targets are the language model's projection names, which Gemma 4 shares (both its architectures: `gemma4`, the E2B and E4B
with a vision encoder, and `gemma4_unified`, the encoder-free 12B). The Gemma line runs through
[`gemma.ipynb`](https://colab.research.google.com/github/AbstractEyes/qwen-symbolic-thinking/blob/main/notebooks/gemma.ipynb)
and `runs/queue_gemma.json`: first the base models' screens, as the Qwen bases were screened before any training (the bare,
shown and specification forms over the same 60 pictures and 40 specifications, thinking off and on), then the arms under Lucia
v1's recipe verbatim (`google/gemma-4-E4B-it`, `google/gemma-4-12B-it`; Apache-2.0; 280 image tokens a picture by default),
screened and read on the same held-out sets so the two families are read on one ruler. One setting differs by architecture: the
encoder-free 12B trains at a learning rate of 5e-5 in place of 2e-4. Its early language-model layers are its vision encoder and sit
under the adapter, and at 2e-4 a divergent step cost the first arm its dependence on the picture between steps 300 and 400 (it then
wrote one caption for every picture); two seeds at 5e-5 keep that dependence through training. `tools/picture_probe.py` (the queue's
`tool` kind) measures a checkpoint's dependence on the picture as the loss gap between the true picture and a blank one, so a new
architecture is read for its sight at its early checkpoints before its reads.

Further families take the same path: the model loads through the multimodal auto class (or the image-text-to-text class where the
first does not map its configuration), a batch's pictures go to the processor as one flat list or as one list per text, whichever it
takes, and the LoRA pattern also names the fused gate-and-up projection some language models use (`gate_up_proj` in GLM-4V). Where
a template writes nothing after an answer (GLM-4V opens the next turn instead), a training row ends with the tokenizer's
end-of-sequence token, one of the ids generation stops on. Ministral 3's processor config leaves `do_pad` null. Transformers 5 reads
that as no padding, so pictures of different sizes cannot share a batch. Its image processor is set to pad, which is the class
default; the model crops each picture back by its size. Its vision tower packs all of a batch's pictures into one sequence; in a
batch of 64 pictures on the card the pictures after the first few came out wrong (the held-out probe named the medium 322 times in
500 at batch 64 and 415 one picture at a time), so from 0.1.38 the vision tower takes a batch's pictures one at a time
(`family.prepare_model`) while the language model keeps the whole batch. A new family starts where Qwen and Gemma started: the
base screens, then an arm under Lucia v1's recipe; read a new family's arm one picture at a time beside its batched read before
trusting the batched one.

Two families run their makers' repository code: Phi-4-reasoning-vision-15B and Molmo2-O-7B. A spec asks for it with
`"trust_remote_code": true` and a pinned `"revision"` in its train, screen or extract settings, and the code loads from a local
snapshot at that commit. That code was written against transformers 4.56 and 4.57; Phi-4's image processor subclasses a module that
transformers 5 rewrote. So these runs carry `"transformers": "==4.57.6"` and go to their own queue, `runs/queue_remote_code.json`,
run by [`remote_code.ipynb`](https://colab.research.google.com/github/AbstractEyes/qwen-symbolic-thinking/blob/main/notebooks/remote_code.ipynb)
on that release. The queue checks each spec against the installed transformers (`queue.transformers_fits`); a spec without the
field runs on transformers 5 or later, so neither kind of card takes the other's runs. `family.py` holds what differs. Because
the weights load from a snapshot folder, PEFT would name that folder as the adapter's base model, and the Hub refuses such a card;
the trainer rewrites the saved adapter's config and card to the Hub id before every upload (`train.name_base`, from 0.1.35).

Phi-4-reasoning-vision:

- Its processor has no chat template; the tokenizer's renders the conversation.
- That template takes a message's content as one string, so a picture row is written `<image>`, a newline, then the text.
- The model assigns a batch's pictures to its rows in order, one per row, so in a batch with any picture, each row without one
  gets a blank picture whose features go unused.
- Its own generation from token ids keeps the text attention mask after the first step, which is wrong for every padded row. The
  multimodal embeddings are built first (left-padded), and batched generation runs from those.
- A leading `<nothink>` tag is dropped from its answers.
- Each picture expands inside the model to up to 3,600 tokens (16-px patches, no merge), about three times Qwen's count. A spec's
  `"image_settings"` (in its train, screen or extract settings) sets the image processor, e.g. `{"max_num_patches": 1024}` for a
  picture budget near Qwen's. An arm's screen and reads take its training's settings.

Molmo2:

- Its processor inserts its start token assuming left padding.
- Under the right padding of a training batch, the processor marks the padding as attended, so the mask is rebuilt from the pad
  id.
- It adds the picture features to the token embeddings in place, which autograd refuses once the embeddings' output is marked as
  requiring grad. PEFT installs that mark (transformers' input-embedding grad hook) when it wraps a model whose layers checkpoint, so
  the trainer removes the hook after building the LoRA (`family.drop_input_grads`); non-reentrant checkpointing trains the adapter
  without it.
- Its image batching (`build_batched_images`) fails its own check on a batch that mixes picture rows with text-only rows (the
  specification rows): an example without a picture gets a one-item offset list. `family.prepare_model` runs that function on the
  picture rows and gives the text-only rows its own padding (-1), which the vision backbone masks out.

Neither family keeps its language model inside a `language_model` module, so the LoRA pattern names their layers from the root:
`model.layers.N` (a fused `qkv_proj`) for Phi-4, and `model.transformer.blocks.N` (`att_proj`, `attn_out`, `ff_proj`, `ff_out`)
for Molmo2.

## The prompt enhancers of Qwen-Image 2.1

Qwen ships two 9B prompt enhancers beside Qwen-Image 2.1, [PE-I2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-I2I) (pictures and
an edit request in, thinking, then an edit prompt as JSON) and [PE-T2I](https://huggingface.co/Qwen/Qwen-Image-2.1-PE-T2I) (a short
prompt in, thinking, then a long picture prompt and an aspect ratio). Both carry the Qwen3.5-9B architecture this package trains on.
`qwen_symbolic_thinking.pe.run` has one of them read the manifest's pictures under a no-change request (its thinking of what stays
fixed becomes a full reading) or rewrite the seeds' captions, with the shipped system prompt and the model cards' sampling; each
answer is split at `</think>`, the thinking and the prompt are scored against the row's captions with the same said / backed rulers
the extraction uses, the thinking's sentences are mapped onto the sixteen step names by their words, and the rows and a summary by
source ship to `pe/<stamp>_<tag>/` in the project repo. The read puts a second 9B looker beside the trained adapter on the same
pictures; the enhancers themselves are under the Qwen Research Licence (research and evaluation only).

```python
from qwen_symbolic_thinking.pe import run
run(kind="i2i", manifest="hub:AbstractPhil/qwen-thinking-images/manifest_v1.2.jsonl", per_source=60, tag="PE_i2i_read")
run(kind="t2i", seeds="hub:synth/seeds_v1.json", n=1000, tag="PE_t2i_read")
```

## License

The code is MIT (from v0.1.56; the tags before it were released under Apache-2.0), so anyone may use it and replicate the
training. The Qwen3.5 models are Apache-2.0; the datasets' rows carry their own licenses per row. Trained weights and row datasets
are released under what their data allows: Apache-2.0 repos keep the plain name, non-commercial repos take the `-nc` suffix, and
each ships what is needed to replicate its training. The Qwen-Image 2.1 prompt enhancers that `pe.run` reads with are under the
Qwen Research License (research and evaluation only); nothing of them is redistributed here.
