"""qwen_symbolic_thinking: a Qwen vision-language model taught to think about a picture in writing, the way the rows of
huggingface.co/datasets/AbstractPhil/thought-symbolic-image-captions do.

    from qwen_symbolic_thinking import chain, prompts      # the chain as text: render, parse, score; the prompt forms
    from qwen_symbolic_thinking.data import build_screen   # the screen set from the accepted rows and the seeds
    from qwen_symbolic_thinking.screen import run          # the behaviour screen on a GPU (Colab)
    from qwen_symbolic_thinking.train import run           # the LoRA fine-tune (the chain, or the whole row)
    from qwen_symbolic_thinking.queue import run_queue     # a card working through a queue of runs kept on the Hub
    from qwen_symbolic_thinking.captions import run        # captions and booru tags for a zip of photos (a stock VLM + a tagger)
    from qwen_symbolic_thinking import pose                # body angles from a 3D body fit (SAM 3D Body): hips, chest, head;
                                                           # every person box of a photo (predict_many) and same-person (apart)
    from qwen_symbolic_thinking import absences            # a caption without its "no eyewear" / "no print or text" statements

Weights and results: huggingface.co/AbstractPhil/qwen-symbolic-thinking-image-training."""
import os

__version__ = "0.1.57"

# The CUDA caching allocator grows by expandable segments, so a long batch late in a run does not fail on memory the process
# already holds in fragments (a 4B full fine-tune died at step 1,220 of 2,724 asking for 5 GB with 15 GB reserved but unusable).
# Read when CUDA first allocates, so it is set here, at import, before any model loads; an explicit setting in the environment wins.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# The project repo: every hub: input path, queue, run marker and shipped result is read from or written to it. A runtime points
# the whole package at another repo (a private twin, for training sets built from rows that are not public) by setting QST_REPO
# before the package is imported.
REPO = os.environ.get("QST_REPO") or "AbstractPhil/qwen-symbolic-thinking-image-training"
DATASET = "AbstractPhil/thought-symbolic-image-captions"
