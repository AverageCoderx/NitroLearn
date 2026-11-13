"""One GPU routine for a hidden layer's LayerNorm and ReLU (`model.fused_norm`), or leaky ReLU
(a slope for negative numbers, 0 for ReLU).

PyTorch runs LayerNorm and ReLU as separate passes over the layer's numbers, and under FP16 autocast it
also converts them to FP32 and back around the LayerNorm; at the presets' size that is about half of an
FP16 learning step. Here both run in one pass forward and one backward, with the statistics in FP32
and the numbers read and written in the layer's own precision. The kernels are CUDA C compiled on first
use with NVRTC, the runtime compiler PyTorch ships, so no CUDA install is needed; they read FP16 as raw
16-bit values and convert with PTX instructions, so they need no CUDA headers either. On the CPU, or when
the routine cannot be built, the usual modules run instead (with one warning in the log).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import nn

from nitro import nvrtc

log = logging.getLogger("nitro")

LEAKY_SLOPE = 0.01  # nn.LeakyReLU's, and the bot's (nitro_core::policy)
THREADS = 128  # per block, a multiple of 32: one row per block forward
ROWS_PER_BLOCK = 32  # backward: rows one block takes, adding up their weight gradients in one place

_HELPERS = r"""
typedef unsigned short half_t;

__device__ __forceinline__ float load_f32(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ float load_f16(const half_t* p, long long i) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[i]));
    return f;
}
__device__ __forceinline__ void store_f32(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void store_f16(half_t* p, long long i, float v) {
    half_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    p[i] = h;
}

// The block's sum, the same order every time (so results repeat bit for bit).
__device__ __forceinline__ float block_sum(float v, float* shared) {
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) shared[threadIdx.x >> 5] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) total += shared[w];
    return total;
}

// ReLU (slope 0) or leaky ReLU, keeping NaN as torch does, so the health checks still see it.
__device__ __forceinline__ float act(float z, float slope) {
    return (z > 0.0f || z != z) ? z : (slope == 0.0f ? 0.0f : slope * z);
}
// Its gradient, with the pre-activation `z`.
__device__ __forceinline__ float act_grad(float z, float g, float slope) {
    return z > 0.0f ? g : (slope == 0.0f ? 0.0f : slope * g);
}
"""

SOURCE = (
    _HELPERS
    + r"""
#define FORWARD(NAME, TIN, LOAD, TOUT, STORE)                                                            \
extern "C" __global__ void NAME(const TIN* x, const float* w, const float* b, TOUT* y, float* mean_out,   \
                                float* rstd_out, int n, double eps, double slope_in) {                  \
    __shared__ float shared[32];                                                                         \
    const long long base = (long long)blockIdx.x * n;                                                   \
    const float slope = (float)slope_in;                                                                 \
    float s = 0.0f;                                                                                      \
    for (int j = threadIdx.x; j < n; j += blockDim.x) s += LOAD(x, base + j);                            \
    const float mean = block_sum(s, shared) / n;                                                         \
    float v = 0.0f;                                                                                      \
    for (int j = threadIdx.x; j < n; j += blockDim.x) {                                                  \
        const float d = LOAD(x, base + j) - mean;                                                        \
        v += d * d;                                                                                      \
    }                                                                                                    \
    const float rstd = rsqrtf(block_sum(v, shared) / n + (float)eps);                                    \
    for (int j = threadIdx.x; j < n; j += blockDim.x)                                                    \
        STORE(y, base + j, act((LOAD(x, base + j) - mean) * rstd * w[j] + b[j], slope));                 \
    if (threadIdx.x == 0) {                                                                              \
        mean_out[blockIdx.x] = mean;                                                                     \
        rstd_out[blockIdx.x] = rstd;                                                                     \
    }                                                                                                    \
}

// dy has the output's precision, dx the input's. Each block writes its rows' share of the weight and bias
// gradients to its own row of the partial sums, which PyTorch then adds up.
#define BACKWARD(NAME, TIN, LOAD, STOREX, TOUT, LOADG)                                                   \
extern "C" __global__ void NAME(const TOUT* dy, const TIN* x, const float* w, const float* b,            \
                                const float* mean_in, const float* rstd_in, TIN* dx, float* dw_part,     \
                                float* db_part, int rows, int n, int rows_per_block, double slope_in) { \
    __shared__ float shared[32];                                                                         \
    const float slope = (float)slope_in;                                                                 \
    float* dwp = dw_part + (long long)blockIdx.x * n;                                                   \
    float* dbp = db_part + (long long)blockIdx.x * n;                                                   \
    for (int j = threadIdx.x; j < n; j += blockDim.x) {                                                  \
        dwp[j] = 0.0f;                                                                                   \
        dbp[j] = 0.0f;                                                                                   \
    }                                                                                                    \
    const int first = blockIdx.x * rows_per_block;                                                       \
    const int last = min(first + rows_per_block, rows);                                                  \
    for (int row = first; row < last; ++row) {                                                           \
        const long long base = (long long)row * n;                                                       \
        const float mean = mean_in[row], rstd = rstd_in[row];                                            \
        float s1 = 0.0f, s2 = 0.0f;                                                                      \
        for (int j = threadIdx.x; j < n; j += blockDim.x) {                                              \
            const float xh = (LOAD(x, base + j) - mean) * rstd;                                          \
            const float g = act_grad(xh * w[j] + b[j], LOADG(dy, base + j), slope);                      \
            const float dxh = g * w[j];                                                                  \
            s1 += dxh;                                                                                   \
            s2 += dxh * xh;                                                                              \
        }                                                                                                \
        s1 = block_sum(s1, shared) / n;                                                                  \
        s2 = block_sum(s2, shared) / n;                                                                  \
        for (int j = threadIdx.x; j < n; j += blockDim.x) {                                              \
            const float xh = (LOAD(x, base + j) - mean) * rstd;                                          \
            const float g = act_grad(xh * w[j] + b[j], LOADG(dy, base + j), slope);                      \
            dwp[j] += g * xh;                                                                            \
            dbp[j] += g;                                                                                 \
            STOREX(dx, base + j, rstd * (g * w[j] - s1 - xh * s2));                                      \
        }                                                                                                \
    }                                                                                                    \
}

FORWARD(fwd_f32_f32, float, load_f32, float, store_f32)
FORWARD(fwd_f16_f16, half_t, load_f16, half_t, store_f16)
FORWARD(fwd_f16_f32, half_t, load_f16, float, store_f32)
FORWARD(fwd_f32_f16, float, load_f32, half_t, store_f16)
BACKWARD(bwd_f32_f32, float, load_f32, store_f32, float, load_f32)
BACKWARD(bwd_f16_f16, half_t, load_f16, store_f16, half_t, load_f16)
BACKWARD(bwd_f16_f32, half_t, load_f16, store_f16, float, load_f32)
BACKWARD(bwd_f32_f16, float, load_f32, store_f32, half_t, load_f16)
"""
)

_TAGS = {torch.float32: "f32", torch.float16: "f16"}
KERNELS = [f"{kind}_{a}_{b}" for kind in ("fwd", "bwd") for a in ("f32", "f16") for b in ("f32", "f16")]

_built: dict[int, dict[str, Any]] = {}  # device index -> kernel name -> launcher
_failed: dict[int, str] = {}  # device index -> why it could not be built
_warned: set[int] = set()


def compile_to_cubin(architecture: str) -> bytes:
    """The kernels as a CUBIN for `architecture` (e.g. "75"), with NVRTC alone: no CUDA headers or install.
    Needs no graphics card, so the source can be checked anywhere NVRTC is."""
    # IEEE maths: no fast-math flags.
    return nvrtc.compile_source(SOURCE, "fused_norm.cu", architecture).binary


def kernels(device: torch.device) -> dict[str, Any] | None:
    """The launchers for `device`, built and checked on first use; None when they cannot be (the reason in
    `_failed`), or for now while a CUDA graph is being captured (the next eager use builds them)."""
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _built:
        return _built[index]
    if index in _failed or torch.cuda.is_current_stream_capturing():
        return None
    try:
        from torch.cuda._utils import _cuda_load_module

        with torch.cuda.device(index):
            major, minor = torch.cuda.get_device_capability(index)
            module = _cuda_load_module(compile_to_cubin(f"{major}{minor}"), KERNELS)
            if not isinstance(module, dict):
                raise TypeError(f"torch's module loader returned {type(module).__name__}")
            _self_test(module, torch.device("cuda", index))
        _built[index] = module
        return module
    except Exception as err:  # any failure leaves the usual LayerNorm and ReLU in place
        _failed[index] = f"{type(err).__name__}: {err}"
        return None


def _self_test(module: dict[str, Any], device: torch.device) -> None:
    """Every kernel once on a small block of numbers against LayerNorm then ReLU and leaky ReLU in FP64, so a
    routine that builds but computes wrongly (a changed torch launcher, say) is never used. Uses no random
    numbers."""
    rows, n = 3, 40
    grid = torch.arange(rows * n, device=device, dtype=torch.float64).view(rows, n)
    x = torch.sin(grid * 0.37) * 2 + 0.3
    weight = (1 + 0.01 * torch.arange(n, device=device, dtype=torch.float64)).float()
    bias = (0.05 * torch.cos(torch.arange(n, device=device, dtype=torch.float64))).float()
    for slope in (0.0, LEAKY_SLOPE):
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device.type, enabled=False):
            x64 = x.clone().requires_grad_()
            w64, b64 = weight.double().requires_grad_(), bias.double().requires_grad_()
            z64 = nn.functional.layer_norm(x64, (n,), w64, b64, 1e-5)
            # No gradient where rounding could flip the activation's side.
            dy = torch.where(z64.detach().abs() < 0.05, 0.0, torch.cos(grid * 0.53))
            nn.functional.leaky_relu(z64, slope).backward(dy)  # type: ignore[no-untyped-call]
            y64 = nn.functional.leaky_relu(z64.detach(), slope)
        for dtype_in in _TAGS:
            for dtype_out in _TAGS:
                tolerance = 1e-4 if dtype_in == dtype_out == torch.float32 else 2e-2
                xin = x.to(dtype_in)
                y, mean, rstd = _forward(module, xin, weight, bias, 1e-5, dtype_out, slope)
                dx, dw, db = _backward(module, dy.to(dtype_out), xin, weight, bias, mean, rstd, slope)
                for got, want in ((y, y64), (dx, x64.grad), (dw, w64.grad), (db, b64.grad)):
                    assert want is not None
                    error = float((got.double() - want).abs().max()) / max(1.0, float(want.abs().max()))
                    if not error <= tolerance:
                        raise RuntimeError(
                            f"self-test {_TAGS[dtype_in]}->{_TAGS[dtype_out]} slope {slope:g}: off by "
                            f"{error:.3g} (allowed {tolerance:g})"
                        )
    torch.cuda.synchronize(device)


def available(device: torch.device) -> bool:
    return device.type == "cuda" and kernels(device) is not None


def _forward(
    module: dict[str, Any],
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out: torch.dtype,
    slope: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, n = x.shape
    y = torch.empty((rows, n), dtype=out, device=x.device)
    mean = torch.empty(rows, dtype=torch.float32, device=x.device)
    rstd = torch.empty(rows, dtype=torch.float32, device=x.device)
    if rows:
        with torch.cuda.device(x.device):
            module[f"fwd_{_TAGS[x.dtype]}_{_TAGS[out]}"](
                grid=(rows, 1, 1),
                block=(THREADS, 1, 1),
                args=[x, weight, bias, y, mean, rstd, n, float(eps), float(slope)],
                stream=torch.cuda.current_stream(x.device),
            )
    return y, mean, rstd


def _backward(
    module: dict[str, Any],
    dy: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    slope: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, n = x.shape
    dx = torch.empty_like(x)
    blocks = max(1, math.ceil(rows / ROWS_PER_BLOCK))
    dw_part = torch.empty((blocks, n), dtype=torch.float32, device=x.device)
    db_part = torch.empty((blocks, n), dtype=torch.float32, device=x.device)
    kernel = module[f"bwd_{_TAGS[x.dtype]}_{_TAGS[dy.dtype]}"]
    if rows:
        with torch.cuda.device(x.device):
            kernel(
                grid=(blocks, 1, 1),
                block=(THREADS, 1, 1),
                args=[
                    *(dy, x, weight, bias, mean, rstd, dx, dw_part, db_part),
                    *(rows, n, ROWS_PER_BLOCK, float(slope)),
                ],
                stream=torch.cuda.current_stream(x.device),
            )
    else:
        dw_part.zero_()
        db_part.zero_()
    return dx, dw_part.sum(0), db_part.sum(0)


def _built_for(x: torch.Tensor) -> dict[str, Any]:
    built = kernels(x.device)
    assert built is not None
    return built


class _Function(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        eps: float,
        out: torch.dtype,
        slope: float = 0.0,
    ) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        y, mean, rstd = _forward(_built_for(flat), flat, weight, bias, eps, out, slope)
        ctx.save_for_backward(flat, weight, bias, mean, rstd)
        ctx.shape, ctx.slope = shape, slope
        return y.view(*shape[:-1], shape[-1])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, dy: torch.Tensor) -> tuple[Any, ...]:
        flat, weight, bias, mean, rstd = ctx.saved_tensors
        dy = dy.reshape(flat.shape).contiguous()
        dx, dw, db = _backward(_built_for(flat), dy, flat, weight, bias, mean, rstd, ctx.slope)
        return dx.view(ctx.shape), dw, db, None, None, None


class FusedLayerNormReLU(nn.LayerNorm):
    """LayerNorm then ReLU (or leaky ReLU) in one GPU routine, with nn.LayerNorm's weights (so saves are the
    same with and without it). `relu` is the module that would otherwise follow (a ReLU or an
    nn.LeakyReLU); it runs, after the usual LayerNorm, on the CPU or when the routine cannot be built, so the
    numbers there are exactly the unfused ones."""

    def __init__(self, width: int, relu: nn.Module, to_half: bool) -> None:
        super().__init__(width)
        self.relu = relu
        self.to_half = to_half
        self.slope = float(relu.negative_slope) if isinstance(relu, nn.LeakyReLU) else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fused(x):
            result: torch.Tensor = self.relu(super().forward(x))
            return result
        return self._run(x, self.weight, self.bias)

    def _run(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        device = x.device.type
        half = torch.is_autocast_enabled(device) and torch.get_autocast_dtype(device) == torch.float16
        out = torch.float16 if self.to_half and half else torch.float32
        result: torch.Tensor = _Function.apply(x, weight, bias, self.eps, out, self.slope)
        return result

    def _fused(self, x: torch.Tensor) -> bool:
        if x.device.type != "cuda" or x.dtype not in _TAGS or self.weight.dtype != torch.float32:
            return False
        if available(x.device):
            return True
        if torch.cuda.is_current_stream_capturing():
            return False  # built at the next eager use
        index = x.device.index if x.device.index is not None else torch.cuda.current_device()
        if index not in _warned:
            _warned.add(index)
            log.warning(
                "model.fused_norm: the GPU routine could not be built (%s); the usual LayerNorm and ReLU run "
                "instead, at their usual speed.",
                _failed.get(index, "unknown reason"),
            )
        return False


class AppliedReLU(nn.Module):
    """Where the ReLU was: the fused LayerNorm before it applied it already. Keeps the layer numbering (and
    so the saves) the same, and tells the bot export that a ReLU (`slope` 0) or a leaky ReLU stands here."""

    def __init__(self, slope: float = 0.0) -> None:
        super().__init__()
        self.slope = slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
