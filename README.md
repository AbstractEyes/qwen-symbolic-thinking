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
pip install git+https://github.com/AbstractEyes/qwen-symbolic-thinking@v0.1.23
pip install "qwen-symbolic-thinking[screen] @ git+https://github.com/AbstractEyes/qwen-symbolic-thinking@v0.1.23"   # with torch, transformers 5, accelerate, peft
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

`qwen_symbolic_thinking.synth.generate` makes synthetic picture items: the base model, shown two worked examples, thinks
through pictures no row was written about; a chain passes the audit when it holds the form and the depth, defines nothing,
remarks on nothing, strays nowhere, and both says and is backed by the picture's own captions (0.2 each); the passing chains
ship as short-form items under `synth/`. `qwen_symbolic_thinking.queue.run_queue(lane)` lets a card work unattended through
`runs/queue_<lane>.json` on the Hub (fine-tunes, each followed by its screen; synthetic generation; throughput sweeps;
extractions), leaving `runs/done/<name>.json` per run; `notebooks/lane1.ipynb` to `lane3.ipynb` run it. A train spec may name
runs under `after`: extract specs (the held-out probe split, the museum pictures) that run as soon as its screen ends, reading the
adapter it trained with the model it trained, so one queue entry trains, screens and reads an arm.

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

```python
from qwen_symbolic_thinking import chain
parsed = chain.parse(model_text)          # {thoughts: [{step, text}], wordings: {caption: ...}, questions: [...], stray: [...]}
scores = chain.score(parsed, row)         # the reads above, for one row of the dataset
text = chain.render(row, full=True)       # the whole row as the model is taught to write it
```

Tests: `pytest tests` (no GPU needed).

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

## Other families: Gemma 4

The trainer, the screen and the extraction are written against the chat template, not against one model: the prompt is rendered
with the template's own thinking switch off, the training text is the generation prompt + the row + whatever the template writes
after an answer (`train.turn_end`, read from the template itself: `<|im_end|>` for Qwen3.5, `<turn|>` for Gemma 4), and the
LoRA targets are the language model's projection names, which Gemma 4 shares (both its architectures: `gemma4`, the E2B and E4B
with a vision encoder, and `gemma4_unified`, the encoder-free 12B). The Gemma line runs through
[`gemma.ipynb`](https://colab.research.google.com/github/AbstractEyes/qwen-symbolic-thinking/blob/main/notebooks/gemma.ipynb)
and `runs/queue_gemma.json`: first the base models' screens, as the Qwen bases were screened before any training (the bare,
shown and specification forms over the same 60 pictures and 40 specifications, thinking off and on), then the arms under Lucia
v1's recipe verbatim (`google/gemma-4-E4B-it`, `google/gemma-4-12B-it`; Apache-2.0; 280 image tokens a picture by default),
screened and read on the same held-out sets so the two families are read on one ruler.

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

Apache-2.0. The Qwen3.5 models are Apache-2.0; the dataset's rows carry their own licences per row. The Qwen-Image 2.1 prompt
enhancers that `pe.run` reads with are under the Qwen Research Licence (research and evaluation only); nothing of them is
redistributed here.
