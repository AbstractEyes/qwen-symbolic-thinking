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
pip install git+https://github.com/AbstractEyes/qwen-symbolic-thinking@v0.1.11
pip install "qwen-symbolic-thinking[screen] @ git+https://github.com/AbstractEyes/qwen-symbolic-thinking@v0.1.11"   # with torch, transformers 5, accelerate, peft
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
the same screen in the trained forms (`full`, `short`, `spec_full`).

`qwen_symbolic_thinking.synth.generate` makes synthetic picture items: the base model, shown two worked examples, thinks
through pictures no row was written about; a chain passes the audit when it holds the form and the depth, defines nothing,
remarks on nothing, strays nowhere, and both says and is backed by the picture's own captions (0.2 each); the passing chains
ship as short-form items under `synth/`. `qwen_symbolic_thinking.queue.run_queue(lane)` lets a card work unattended through
`runs/queue_<lane>.json` on the Hub (fine-tunes, each followed by its screen; synthetic generation; throughput sweeps;
extractions), leaving `runs/done/<name>.json` per run; `notebooks/lane1.ipynb` to `lane3.ipynb` run it.

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
