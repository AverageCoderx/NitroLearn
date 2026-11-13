"""NVRTC, the CUDA compiler PyTorch ships: CUDA C source to a CUBIN for one graphics card. Needs no graphics
card and no CUDA install. Used by `model.fused_norm` and `model.small_attention`; `NITRO_NVRTC` may name
another NVRTC library (tests, or a machine without torch's).
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Compiled:
    """A compiled routine: a CUBIN for `arch` (e.g. 86), or PTX for `compute_{arch}`."""

    binary: bytes
    kind: Literal["cubin", "ptx"]
    arch: int
    log: str = ""


def library() -> ctypes.CDLL:
    """NVRTC: the library `NITRO_NVRTC` names, else the one torch ships (OSError when there is none)."""
    path = os.environ.get("NITRO_NVRTC")
    if path:
        return ctypes.CDLL(path)
    from torch.cuda._utils import _get_nvrtc_library

    nvrtc: ctypes.CDLL = _get_nvrtc_library()
    return nvrtc


def compile_source(
    source: str,
    name: str,
    arch: int | str,
    *,
    options: Sequence[str] = (),
    headers: Sequence[tuple[str, str]] = (),
    ptx: bool = False,
    nvrtc: ctypes.CDLL | None = None,
) -> Compiled:
    """`source` (file name `name`; `headers` are (name, text) it may include) compiled for `arch` with
    `options`: a CUBIN, or with `ptx` PTX for `compute_{arch}`. RuntimeError with NVRTC's log on failure."""
    nvrtc = nvrtc or library()

    def log_of(program: ctypes.c_void_p) -> str:
        size = ctypes.c_size_t()
        nvrtc.nvrtcGetProgramLogSize(program, ctypes.byref(size))
        text = ctypes.create_string_buffer(size.value)
        nvrtc.nvrtcGetProgramLog(program, text)
        return text.value.decode(errors="replace").strip()

    def check(code: int, program: ctypes.c_void_p | None = None) -> None:
        if code == 0:
            return
        detail = ": " + log_of(program) if program is not None else ""
        raise RuntimeError(f"NVRTC error {code}{detail}")

    program = ctypes.c_void_p()
    names = (ctypes.c_char_p * len(headers))(*[h[0].encode() for h in headers])
    texts = (ctypes.c_char_p * len(headers))(*[h[1].encode() for h in headers])
    check(
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(program),
            source.encode(),
            name.encode(),
            len(headers),
            texts if headers else None,
            names if headers else None,
        )
    )
    try:
        target = f"compute_{arch}" if ptx else f"sm_{arch}"
        flags = [f"--gpu-architecture={target}".encode(), *(o.encode() for o in options)]
        array = (ctypes.c_char_p * len(flags))(*flags)
        check(nvrtc.nvrtcCompileProgram(program, len(flags), array), program)
        log = log_of(program)
        size = ctypes.c_size_t()
        get_size, get = (
            (nvrtc.nvrtcGetPTXSize, nvrtc.nvrtcGetPTX)
            if ptx
            else (nvrtc.nvrtcGetCUBINSize, nvrtc.nvrtcGetCUBIN)
        )
        check(get_size(program, ctypes.byref(size)))
        binary = ctypes.create_string_buffer(size.value)
        check(get(program, binary))
        return Compiled(binary.raw, "ptx" if ptx else "cubin", int(arch), log)
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))
