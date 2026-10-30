"""qwen_symbolic_thinking: a Qwen vision-language model taught to think about a picture in writing, the way the rows of
huggingface.co/datasets/AbstractPhil/thought-symbolic-image-captions do.

    from qwen_symbolic_thinking import chain, prompts      # the chain as text: render, parse, score; the prompt forms
    from qwen_symbolic_thinking.data import build_screen   # the screen set from the accepted rows and the seeds
    from qwen_symbolic_thinking.screen import run          # the behaviour screen on a GPU (Colab)
    from qwen_symbolic_thinking.train import run           # the LoRA fine-tune (the chain, or the whole row)
    from qwen_symbolic_thinking.queue import run_queue     # a card working through a queue of runs kept on the Hub

Weights and results: huggingface.co/AbstractPhil/qwen-symbolic-thinking-image-training."""
__version__ = "0.1.5"

REPO = "AbstractPhil/qwen-symbolic-thinking-image-training"
DATASET = "AbstractPhil/thought-symbolic-image-captions"
