"""Attention over a few tokens as one GPU routine (`model.small_attention`).

The transformer encoder looks at 5 to 8 tokens (the cars and the ball). PyTorch's fused attention kernels are
made for long sequences: with a mask they run the memory-efficient kernel, whose 64-token tiles were mostly
empty here, and that one kernel took a quarter of a big 2v2 transformer's training time.
Here one warp (32 threads) takes one sample's head: each thread holds one of the head's (up to 32) numbers for
every token's query, key and value, the scores are sums across the warp, the softmax runs in FP32 in
registers, and the result is written straight in the order the output layer reads. Backward recomputes the
scores from the queries and keys (no score matrix is stored) and writes the gradients in the order the q, k,
v layer made them. FP32 sums whatever the numbers' precision (FP16, FP32 or FP64, which the checks use). The
kernels are compiled on first use with NVRTC, as `nitro.fused_norm`'s; on the CPU, with more tokens or wider
heads than the routine holds, or when it cannot be built, PyTorch's attention runs instead.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch

from nitro import nvrtc

log = logging.getLogger("nitro")

MAX_TOKENS = 8  # the scores of a head's tokens stay in registers
MAX_HEAD = 32  # one thread per number of a head
WARPS = 4  # per block

SOURCE = r"""
#define MAX_T 8
typedef unsigned short half_t;

__device__ __forceinline__ float to_f(const half_t* p, long long i) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(p[i]));
    return f;
}
__device__ __forceinline__ float to_f(const float* p, long long i) { return p[i]; }
__device__ __forceinline__ double to_f(const double* p, long long i) { return p[i]; }
__device__ __forceinline__ void put(half_t* p, long long i, float v) {
    half_t h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(v));
    p[i] = h;
}
__device__ __forceinline__ void put(float* p, long long i, float v) { p[i] = v; }
__device__ __forceinline__ void put(double* p, long long i, double v) { p[i] = v; }
__device__ __forceinline__ float ex(float x) { return expf(x); }
__device__ __forceinline__ double ex(double x) { return exp(x); }
__device__ __forceinline__ void lowest(float& x) { x = __int_as_float(0xff800000); }
__device__ __forceinline__ void lowest(double& x) { x = __longlong_as_double(0xfff0000000000000ULL); }

// The sum over the warp, in the same order every time (so results repeat bit for bit).
template <typename A> __device__ __forceinline__ A warp_sum(A v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

// One warp per sample's head; each thread holds number `lane` of the head for every token. Loops run to MAX_T
// with the tokens past `t` skipped, so every array index is known when compiling and the arrays stay in
// registers.
template <typename T, typename A> struct Head {
    A q[MAX_T], k[MAX_T], v[MAX_T];
    bool here[MAX_T];
    long long sample, w;
    int head, lane;

    __device__ __forceinline__ bool load(const T* qkv, const unsigned char* present, int b, int t, int h,
                                         int d) {
        const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
        lane = threadIdx.x & 31;
        if (warp >= (long long)b * h) return false;
        sample = warp / h;
        head = (int)(warp % h);
        w = (long long)h * d;
#pragma unroll
        for (int j = 0; j < MAX_T; ++j) {
            q[j] = k[j] = v[j] = 0;
            here[j] = false;
            if (j < t) {
                const long long base = (sample * t + j) * 3 * w + head * d + lane;
                if (lane < d) {
                    q[j] = to_f(qkv, base);
                    k[j] = to_f(qkv, base + w);
                    v[j] = to_f(qkv, base + 2 * w);
                }
                here[j] = present[sample * t + j] != 0;
            }
        }
        return true;
    }

    // Row i of the attention weights (0 for empty places and past t).
    __device__ __forceinline__ void weights(int i, int t, A scale, A* p) const {
        A top;
        lowest(top);
#pragma unroll
        for (int j = 0; j < MAX_T; ++j) {
            lowest(p[j]);
            if (j < t) {
                const A dot = warp_sum(q[i] * k[j]) * scale;
                if (here[j]) p[j] = dot;
                top = p[j] > top ? p[j] : top;
            }
        }
        A total = 0;
#pragma unroll
        for (int j = 0; j < MAX_T; ++j) {
            p[j] = (j < t && here[j]) ? ex(p[j] - top) : (A)0;
            total += p[j];
        }
#pragma unroll
        for (int j = 0; j < MAX_T; ++j) p[j] = p[j] / total;
    }
};

// qkv: [b, t, 3, h, d] (the q, k, v layer's output as it is); present: [b, t] (0 = an empty place); out:
// [b, t, h, d]. `A` is the type the sums run in (float, or double for the FP64 checks).
template <typename T, typename A>
__device__ void forward(const T* qkv, const unsigned char* present, T* out, int b, int t, int h, int d,
                        A scale) {
    Head<T, A> s;
    if (!s.load(qkv, present, b, t, h, d)) return;
#pragma unroll
    for (int i = 0; i < MAX_T; ++i) {
        if (i < t) {
            A p[MAX_T];
            s.weights(i, t, scale, p);
            A mixed = 0;
#pragma unroll
            for (int j = 0; j < MAX_T; ++j) mixed += p[j] * s.v[j];
            if (s.lane < d) put(out, (s.sample * t + i) * s.w + s.head * d + s.lane, mixed);
        }
    }
}

// grad_out: [b, t, h, d]; grad_qkv: [b, t, 3, h, d]. The weights are recomputed as forward made them.
template <typename T, typename A>
__device__ void backward(const T* qkv, const unsigned char* present, const T* grad_out, T* grad_qkv, int b,
                         int t, int h, int d, A scale) {
    Head<T, A> s;
    if (!s.load(qkv, present, b, t, h, d)) return;
    A g[MAX_T], dq[MAX_T], dk[MAX_T], dv[MAX_T];
#pragma unroll
    for (int j = 0; j < MAX_T; ++j) {
        g[j] = dq[j] = dk[j] = dv[j] = 0;
        if (j < t && s.lane < d) g[j] = to_f(grad_out, (s.sample * t + j) * s.w + s.head * d + s.lane);
    }
#pragma unroll
    for (int i = 0; i < MAX_T; ++i) {
        if (i < t) {
            A p[MAX_T];
            s.weights(i, t, scale, p);
            // dP_ij = g_i . v_j; the softmax's gradient dS_ij = P_ij (dP_ij - sum_j P_ij dP_ij).
            A dp[MAX_T];
            A row = 0;
#pragma unroll
            for (int j = 0; j < MAX_T; ++j) {
                dp[j] = j < t ? warp_sum(g[i] * s.v[j]) : (A)0;
                row += p[j] * dp[j];
            }
#pragma unroll
            for (int j = 0; j < MAX_T; ++j) {
                const A ds = p[j] * (dp[j] - row) * scale;
                dv[j] += p[j] * g[i];
                dq[i] += ds * s.k[j];
                dk[j] += ds * s.q[i];
            }
        }
    }
    if (s.lane < d) {
#pragma unroll
        for (int j = 0; j < MAX_T; ++j) {
            if (j < t) {
                const long long base = (s.sample * t + j) * 3 * s.w + s.head * d + s.lane;
                put(grad_qkv, base, dq[j]);
                put(grad_qkv, base + s.w, dk[j]);
                put(grad_qkv, base + 2 * s.w, dv[j]);
            }
        }
    }
}

#define KERNELS(TAG, T, A)                                                                                  \
extern "C" __global__ void fwd_##TAG(const T* qkv, const unsigned char* present, T* out, int b, int t,      \
                                     int h, int d, double scale) {                                         \
    forward<T, A>(qkv, present, out, b, t, h, d, (A)scale);                                                \
}                                                                                                           \
extern "C" __global__ void bwd_##TAG(const T* qkv, const unsigned char* present, const T* grad_out,        \
                                     T* grad_qkv, int b, int t, int h, int d, double scale) {              \
    backward<T, A>(qkv, present, grad_out, grad_qkv, b, t, h, d, (A)scale);                                \
}

KERNELS(f16, half_t, float)
KERNELS(f32, float, float)
KERNELS(f64, double, double)
"""

_TAGS = {torch.float16: "f16", torch.float32: "f32", torch.float64: "f64"}
KERNELS = [f"{kind}_{tag}" for kind in ("fwd", "bwd") for tag in _TAGS.values()]

_built: dict[int, dict[str, Any]] = {}  # device index -> kernel name -> launcher
_failed: dict[int, str] = {}  # device index -> why it could not be built
_warned: set[int] = set()


def compile_to_cubin(architecture: str) -> nvrtc.Compiled:
    """The kernels for `architecture` (e.g. "89"), with NVRTC alone; needs no graphics card. IEEE maths."""
    return nvrtc.compile_source(SOURCE, "small_attention.cu", architecture, options=["--ptxas-options=-v"])


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
            module = _cuda_load_module(compile_to_cubin(f"{major}{minor}").binary, KERNELS)
            if not isinstance(module, dict):
                raise TypeError(f"torch's module loader returned {type(module).__name__}")
            _self_test(module, torch.device("cuda", index))
        _built[index] = module
        return module
    except Exception as err:  # any failure leaves PyTorch's attention in place
        _failed[index] = f"{type(err).__name__}: {err}"
        return None


def reference(qkv: torch.Tensor, present: torch.Tensor, heads: int) -> torch.Tensor:
    """The same attention in PyTorch's operations (the checks' answer): qkv [b, t, 3 * w], present [b, t]
    bool; returns [b, t, w]."""
    b, t, three_w = qkv.shape
    w = three_w // 3
    q, k, v = qkv.view(b, t, 3, heads, w // heads).permute(2, 0, 3, 1, 4)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(w // heads)
    weights = scores.masked_fill(~present[:, None, None, :], float("-inf")).softmax(-1)
    mixed: torch.Tensor = (weights @ v).transpose(1, 2).reshape(b, t, w)
    return mixed


def _self_test(module: dict[str, Any], device: torch.device) -> None:
    """Each kernel once against `reference` in FP64 on fixed numbers (no random draws), so a routine that
    builds but computes wrongly is never used."""
    b, t, heads, d = 6, 7, 2, 5
    grid = torch.arange(b * t * 3 * heads * d, device=device, dtype=torch.float64)
    qkv = (torch.sin(grid * 0.37) * 1.5).view(b, t, 3 * heads * d)
    present = (torch.arange(b * t, device=device) % 5 != 3).view(b, t)
    present[:, 0] = True
    grad = torch.cos(torch.arange(b * t * heads * d, device=device, dtype=torch.float64) * 0.53).view(
        b, t, -1
    )
    with torch.inference_mode(False), torch.enable_grad(), torch.autocast(device.type, enabled=False):
        exact_in = qkv.clone().requires_grad_()
        exact = reference(exact_in, present, heads)
        exact.backward(grad)  # type: ignore[no-untyped-call]
    assert exact_in.grad is not None
    for dtype, tolerance in ((torch.float64, 1e-12), (torch.float32, 1e-5), (torch.float16, 2e-2)):
        out = _forward(module, qkv.to(dtype), present, heads)
        back = _backward(module, qkv.to(dtype), present, grad.to(dtype), heads)
        for got, want in ((out, exact.detach()), (back, exact_in.grad)):
            error = float((got.double() - want).abs().max()) / max(1.0, float(want.abs().max()))
            if not error <= tolerance:
                raise RuntimeError(f"self-test {_TAGS[dtype]}: off by {error:.3g} (allowed {tolerance:g})")
    torch.cuda.synchronize(device)


def _launch(
    module: dict[str, Any], name: str, rows: int, heads: int, args: list[Any], device: torch.device
) -> None:
    warps = rows * heads
    if warps:
        with torch.cuda.device(device):
            module[name](
                grid=(math.ceil(warps / WARPS), 1, 1),
                block=(32 * WARPS, 1, 1),
                args=args,
                stream=torch.cuda.current_stream(device),
            )


def _flags(present: torch.Tensor) -> torch.Tensor:
    """`present` as the bytes the kernels read (0 = an empty place): the bool tensor's own bytes, viewed
    rather than copied, so no GPU job per call."""
    return present.contiguous().view(torch.uint8)


def _forward(module: dict[str, Any], qkv: torch.Tensor, present: torch.Tensor, heads: int) -> torch.Tensor:
    b, t, three_w = qkv.shape
    w = three_w // 3
    out = torch.empty((b, t, w), dtype=qkv.dtype, device=qkv.device)
    scale = 1 / math.sqrt(w // heads)
    flags = _flags(present)
    _launch(
        module,
        f"fwd_{_TAGS[qkv.dtype]}",
        b,
        heads,
        [qkv, flags, out, b, t, heads, w // heads, scale],
        qkv.device,
    )
    return out


def _backward(
    module: dict[str, Any], qkv: torch.Tensor, present: torch.Tensor, grad: torch.Tensor, heads: int
) -> torch.Tensor:
    b, t, three_w = qkv.shape
    w = three_w // 3
    grad_qkv = torch.empty_like(qkv)
    scale = 1 / math.sqrt(w // heads)
    flags = _flags(present)
    args = [qkv, flags, grad, grad_qkv, b, t, heads, w // heads, scale]
    _launch(module, f"bwd_{_TAGS[qkv.dtype]}", b, heads, args, qkv.device)
    return grad_qkv


class _Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, qkv: torch.Tensor, present: torch.Tensor, heads: int) -> torch.Tensor:
        qkv = qkv.contiguous()
        built = kernels(qkv.device)
        assert built is not None
        ctx.save_for_backward(qkv, present)
        ctx.heads = heads
        return _forward(built, qkv, present, heads)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
        qkv, present = ctx.saved_tensors
        built = kernels(qkv.device)
        assert built is not None
        return _backward(built, qkv, present, grad.contiguous().to(qkv.dtype), ctx.heads), None, None


def usable(qkv: torch.Tensor, heads: int) -> bool:
    """Whether the routine can take this attention: on a built graphics card, few enough tokens, a narrow
    enough head, a precision it has. Warns once per card when it cannot be built."""
    t, three_w = qkv.shape[-2:]
    if (
        qkv.device.type != "cuda"
        or qkv.dtype not in _TAGS
        or t > MAX_TOKENS
        or three_w // 3 // heads > MAX_HEAD
    ):
        return False
    if kernels(qkv.device) is not None:
        return True
    if torch.cuda.is_current_stream_capturing():
        return False  # built at the next eager use
    index = qkv.device.index if qkv.device.index is not None else torch.cuda.current_device()
    if index not in _warned:
        _warned.add(index)
        log.warning(
            "model.small_attention: the GPU routine could not be built (%s); PyTorch's attention runs "
            "instead, at its usual speed.",
            _failed.get(index, "unknown reason"),
        )
    return False


def attention(qkv: torch.Tensor, present: torch.Tensor, heads: int) -> torch.Tensor:
    """Masked multi-head self-attention: `qkv` [b, t, 3 * w] (the q, k, v layer's output), `present` [b, t]
    bool (False: an empty place nobody looks at); returns [b, t, w] in qkv's precision. Call only where
    `usable` says so."""
    result: torch.Tensor = _Function.apply(qkv, present, heads)
    return result
