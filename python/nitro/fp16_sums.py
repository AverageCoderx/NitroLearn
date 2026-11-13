"""FP16 running sums inside the graphics card's FP16 matrix products (`run.fp16_sums`).

PyTorch keeps each product's running sum in FP32, which GeForce cards compute at half their FP16 rate; with
`torch.backends.cuda.matmul.allow_fp16_accumulation` the sums stay in FP16, at the full rate. The flag is the
whole process's and is read when a product is launched or captured into a CUDA graph, so it is set only around
one thread's passes.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

PLAYING = ("playing", "forward", "all")  # the passes that choose actions (the bot's and its past versions')
FORWARD = ("forward", "all")  # learning's forward passes
BACKWARD = ("all",)  # learning's backward pass (the gradients)


def available() -> bool:
    """Whether this PyTorch can keep the sums in FP16 (2.7 and newer)."""
    return hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")


@contextlib.contextmanager
def kept(on: bool) -> Iterator[None]:
    """Inside, FP16 products launched or captured keep their sums in FP16 if `on`; else nothing changes."""
    if not on:
        yield
        return
    matmul = torch.backends.cuda.matmul
    before = matmul.allow_fp16_accumulation
    matmul.allow_fp16_accumulation = True
    try:
        yield
    finally:
        matmul.allow_fp16_accumulation = before
