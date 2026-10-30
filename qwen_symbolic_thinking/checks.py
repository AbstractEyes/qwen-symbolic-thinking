"""Checks before a run: are the linear-attention kernels present, and does PEFT build a LoRA beside the installed torchao?

    from qwen_symbolic_thinking.checks import run
    run()"""
from __future__ import annotations

import importlib
import os


def _version(name: str):
    try:
        return getattr(importlib.import_module(name), "__version__", "present")
    except Exception:
        return None


def fast_path() -> dict:
    """The packages transformers resolves for Qwen3.5's linear attention (fla and causal_conv1d; the kernels hub if installed).
    Absent packages mean the reference PyTorch implementation, which transformers announces once with 'falling back'."""
    out = {name: _version(name) for name in ("fla", "causal_conv1d", "triton", "kernels")}
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule  # noqa: F401
        out["fla_ops"] = True
    except Exception as e:
        out["fla_ops"] = f"{type(e).__name__}: {str(e)[:100]}"
    try:
        from causal_conv1d import causal_conv1d_fn, causal_conv1d_update  # noqa: F401
        out["causal_conv1d_ops"] = True
    except Exception as e:
        out["causal_conv1d_ops"] = f"{type(e).__name__}: {str(e)[:100]}"
    out["USE_HUB_KERNELS"] = os.environ.get("USE_HUB_KERNELS", "YES")
    out["verdict"] = ("kernels present" if out["fla_ops"] is True and out["causal_conv1d_ops"] is True
                      else "reference PyTorch path (install causal-conv1d and flash-linear-attention)")
    return out


def peft_beside_torchao() -> dict:
    """Does PEFT import and build a LoRA with the torchao that is installed? A tiny module gets two adapted layers and one
    backward pass. A failure naming torchao means: pip uninstall -y torchao, then restart the session."""
    out = {name: _version(name) for name in ("torch", "transformers", "peft", "torchao", "accelerate")}
    try:
        import torch
        from peft import LoraConfig, get_peft_model

        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = torch.nn.Linear(16, 16)
                self.down_proj = torch.nn.Linear(16, 4)

            def forward(self, x):
                return self.down_proj(torch.tanh(self.q_proj(x)))

        model = get_peft_model(Tiny(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "down_proj"]))
        model(torch.randn(2, 16)).sum().backward()
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        out.update(peft_lora_ok=True, trainable_params=n)
    except Exception as e:
        out.update(peft_lora_ok=False, error=f"{type(e).__name__}: {str(e)[:300]}")
        if "torchao" in out["error"].lower():
            out["advice"] = "torchao is in the way: pip uninstall -y torchao, then Runtime > Restart session"
    return out


def run() -> dict:
    result = {"fast_path": fast_path(), "peft": peft_beside_torchao()}
    for key, val in result.items():
        print(f"== {key}")
        for k, v in val.items():
            print(f"  {k}: {v}")
    return result
