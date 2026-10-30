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
| `s_per_row`, `tok_per_s` | the time cost on the card |

Results land on the Hub under `screens/<stamp>/`: `results.jsonl` (one line per answer, with the text, the parse and the
scores), `summary_<model>.json`, `summary.json`, `summary.md`, `env.json`.

The picture rows come from COCO 2017 images, named by id in the dataset's seeds and fetched from the COCO site at screen
time; nothing is redistributed here. The specification rows are the dataset's own drawn rows.

## Install

```
pip install git+https://github.com/AbstractEyes/qwen-symbolic-thinking@v0.1.0
pip install "qwen-symbolic-thinking[screen] @ git+https://github.com/AbstractEyes/qwen-symbolic-thinking@v0.1.0"   # with torch, transformers 5, accelerate, peft
```

Qwen3.5 needs transformers 5 (the model code lives under `transformers.models.qwen3_5`); the preflight says so if it is missing.

## The fine-tune

`qwen_symbolic_thinking.train.run` trains a LoRA adapter (rank 32, alpha 64, on the language model's attention,
linear-attention and MLP projections; the vision tower frozen) with plain Adam and a cosine schedule, the loss on the
assistant turn only. Each training item is the picture (or the specification) with a one-line prompt, answered by the row's
own chain. The adapter ships to the Hub during training under `lora/<model>/r<rank>/<stamp>/`. The trained model is read with
the same screen, in the one-line prompt forms (`short`, `spec_short`).

```python
from qwen_symbolic_thinking import chain
parsed = chain.parse(model_text)          # {thoughts: [{step, text}], wordings: {caption: ...}, stray: [...]}
scores = chain.score(parsed, row)         # the reads above, for one row of the dataset
```

Tests: `pytest tests` (no GPU needed).

## License

Apache-2.0. The Qwen3.5 models are Apache-2.0; the dataset's rows carry their own licences per row.
