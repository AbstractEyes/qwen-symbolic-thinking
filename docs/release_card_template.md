---
license: apache-2.0
base_model: {{BASE}}
base_model_relation: {{RELATION}}
library_name: {{LIBRARY}}
pipeline_tag: image-text-to-text
datasets:
- AbstractPhil/thought-symbolic-image-captions
- AbstractPhil/qwen-symbolic-thinking-image-training
language:
- en
tags:
- {{VARIANT_TAG}}
- vision-language
- chain-of-thought
- image-captioning
- art
- photography
- museum
---

# {{TITLE}}

**{{NAME}}** is the {{SIZE_WORDS}} of the Lucia family, the thinking lookers: {{VARIANT_SENTENCE}} of
[{{BASE}}](https://huggingface.co/{{BASE}}) that looks at a picture and writes the row of
[thought-symbolic-image-captions](https://huggingface.co/datasets/AbstractPhil/thought-symbolic-image-captions): a short chain
of plain-language thoughts, one named step per line, then the wordings (a caption, a prompt, tags, a critique, by depth), then
the questions a careful looker would still ask, each with its answer and what settles it. Every size is taught the same rows under
the same recipe; the sizes differ in what they can hold, and the reads below say what this one holds. {{HOLDS}}

```
see: A single figure, seated, holding a book, in three-quarter view.
place: The figure sits at the centre, the book held low against the lap.
form: The sitter fills the frame from the knees up, the background a plain dark wash.
colour: Brown ink over graphite, the paper a warm cream.
mood: The lowered eyes and the still hands suggest absorption, a reader lost in the page.
caption: A seated figure reads a book held in the lap, drawn in brown ink over graphite.
prompt: drawing, seated figure reading, three-quarter view, brown ink over graphite, cream paper, plain dark wash
question: thing name -> book [record]
question: medium -> ink and graphite [record]
question: mood tone -> absorbed [chain]
question: sitter identity -> unknown [unknown]
```

The steps a chain may use, in the order a careful looker takes them: see, name, place, count, relate, form, light, colour, mood,
mean, intend, story, compare, change, check, say. A quick look holds 1 to 2 thoughts, a closer look 3 to 5, a long look 6 to 10.
Questions end in `[record]` (the answer is a fact of the picture), `[chain]` (the chain worked it out) or `[unknown]`. The prompt
line's first term names the kind of picture (painting, print, drawing, photograph).

## Loading

{{VARIANT_NOTE}} Every Lucia loads the same way and is run the same way: thinking off, greedy decoding, no system prompt (the
row's form is learned, and a system turn she was not trained with makes her over-write). Needs transformers 5; the adapter also
needs peft 0.18 or newer.

**The LoRA adapter** ([{{LORA_REPO}}](https://huggingface.co/{{LORA_REPO}})): the base model, the adapter merged into it, the
processor from the adapter repository (the base processor's files are kept there).

```python
import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("{{LORA_REPO}}")
model = AutoModelForMultimodalLM.from_pretrained("{{BASE}}", dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, "{{LORA_REPO}}").merge_and_unload().eval()
```

**The full fine-tune** ([{{FULL_REPO}}](https://huggingface.co/{{FULL_REPO}})): the model and the processor from the full
repository, nothing else to merge.

```python
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("{{FULL_REPO}}")
model = AutoModelForMultimodalLM.from_pretrained("{{FULL_REPO}}", dtype=torch.bfloat16, device_map="cuda").eval()
```

Then, for either:

```python
from PIL import Image

image = Image.open("picture.jpg").convert("RGB")
prompt = ("Look at this picture and think it through in writing, a closer look, ending with a caption; then write a caption "
          "and a prompt or tags, and the questions a careful looker would still ask.")
messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=800, do_sample=False)
print(processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0])
```

The depth words: "a quick look" (1 to 2 thoughts), "a closer look" (3 to 5), "a long look" (6 to 10). The parser, the scorer and
the trainer are in [qwen-symbolic-thinking](https://github.com/AbstractEyes/qwen-symbolic-thinking)
(`qwen_symbolic_thinking.chain.parse`, `chain.score`); the same package extracted every number below.

## The reads

{{READS}}

## How she was trained

{{TRAINING}}

## Versions

{{VERSIONS}}

## The family

{{FAMILY}}

## License

Apache-2.0, the base model's licence. The rows she was taught come from the datasets above under their own terms.
