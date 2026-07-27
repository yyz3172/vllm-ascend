import json
import os
import time
from pathlib import Path

import torch
from vllm.model_executor.layers.batch_invariant import vllm_is_batch_invariant
from vllm.triton_utils import HAS_TRITON
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.ops.topk_topp_sampler import TopKTopPSampler
from vllm.v1.sample.sampler import Sampler

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.sample.penalties import apply_all_penalties
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, global_stream, npu_stream_switch

DEFAULT_LOGPROBS_MODE = "raw_logprobs"

# Set VLLM_ASCEND_DEBUG_TOPK_LOGITS=1 to dump per-call logits stats before
# apply_top_k_top_p (jsonl). Optional: VLLM_ASCEND_DEBUG_TOPK_LOGITS_PATH.
_TOPK_LOGITS_DEBUG = os.getenv("VLLM_ASCEND_DEBUG_TOPK_LOGITS", "0") == "1"
_TOPK_LOGITS_DEBUG_PATH = Path(
    os.getenv(
        "VLLM_ASCEND_DEBUG_TOPK_LOGITS_PATH",
        "/tmp/vllm_ascend_topk_logits_debug.jsonl",
    )
)
_TOPK_LOGITS_CALL_IDX = 0


def _debug_dump_topk_logits(logits: torch.Tensor, k, p, op_ms: float | None = None) -> None:
    """Lightweight host-side stats to bisect ApplyTopKTopPCustom remain-path triggers."""
    global _TOPK_LOGITS_CALL_IDX
    _TOPK_LOGITS_CALL_IDX += 1
    # Keep dumps cheap: sync once, reduce on device, then few scalars to CPU.
    torch.npu.synchronize()
    x = logits.detach()
    if x.dtype != torch.float32:
        x = x.float()
    finite = torch.isfinite(x)
    nan_n = int(torch.isnan(x).sum().item())
    inf_n = int(torch.isinf(x).sum().item())
    # Per-row: how many values equal the row max (ties at the top).
    row_max = x.amax(dim=-1, keepdim=True)
    n_at_max = int((x == row_max).sum().item())
    # Approx "flatness": fraction of vocab within 1e-3 of row max.
    n_near_max = int((x >= (row_max - 1e-3)).sum().item())
    std = float(x.std().item()) if x.numel() > 1 else 0.0
    rec = {
        "i": _TOPK_LOGITS_CALL_IDX,
        "shape": list(x.shape),
        "dtype": str(logits.dtype),
        "contig": bool(logits.is_contiguous()),
        "stride": list(logits.stride()),
        "nan": nan_n,
        "inf": inf_n,
        "finite_all": bool(finite.all().item()),
        "min": float(x.min().item()) if x.numel() else None,
        "max": float(x.max().item()) if x.numel() else None,
        "mean": float(x.mean().item()) if x.numel() else None,
        "std": std,
        "n_at_max": n_at_max,
        "n_near_max_1e-3": n_near_max,
        "frac_at_max": n_at_max / x.numel() if x.numel() else 0.0,
        "frac_near_max_1e-3": n_near_max / x.numel() if x.numel() else 0.0,
        "k_is_none": k is None,
        "p_is_none": p is None,
        "op_ms": op_ms,
    }
    _TOPK_LOGITS_DEBUG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _TOPK_LOGITS_DEBUG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def random_sample(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-NPU synchronization.
    """
    # NOTE(woosuk): To batch-process the requests without their own seeds,
    # which is the common case, we first assume that every request does
    # not have its own seed. Then, we overwrite the values for the requests
    # that have their own seeds.
    with npu_stream_switch(global_stream()):
        q = torch.empty_like(probs)
        if len(generators) != probs.shape[0]:
            q.exponential_()
        if generators:
            # TODO(woosuk): This can be slow because we handle each request
            # one by one. Optimize this.
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)
    torch.npu.current_stream().wait_stream(global_stream())
    return probs.div_(q).argmax(dim=-1).view(-1)


class AscendSampler(Sampler):
    @staticmethod
    def apply_penalties(
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        output_token_ids: list[list[int]],
    ) -> torch.Tensor:
        """Use Triton-Ascend penalties on NPU when Triton is available; else vLLM default."""
        if not HAS_TRITON:
            return Sampler.apply_penalties(logits, sampling_metadata, output_token_ids)

        if sampling_metadata.no_penalties:
            return logits
        assert sampling_metadata.prompt_token_ids is not None
        return apply_all_penalties(
            logits,
            sampling_metadata.prompt_token_ids,
            sampling_metadata.presence_penalties,
            sampling_metadata.frequency_penalties,
            sampling_metadata.repetition_penalties,
            output_token_ids,
        )

    def __init__(self, logprobs_mode=DEFAULT_LOGPROBS_MODE):
        # TODO: support logprobs_mode in vllm-ascend
        super().__init__(logprobs_mode=logprobs_mode)
        self.topk_topp_sampler = AscendTopKTopPSampler()
        self.async_exponential_event = torch.npu.Event()

    def set_q_event(self, q, event):
        self.topk_topp_sampler.set_q_event(q, event)

    def do_async_exponential(self, b_s, head_dim, generators):
        # Calculating exponential randoms in a different stream
        # and overlapping with model executing.
        with torch.npu.stream(global_stream()):
            global_stream().wait_stream(torch.npu.current_stream())
            q = torch.empty((b_s, head_dim), device="npu", dtype=torch.float32)
            # Goes to async exponential with AI-CPU exponential or default exponential.
            if len(generators) != q.shape[0]:
                q.exponential_()
            if generators:
                for i, generator in generators.items():
                    q[i].exponential_(generator=generator)
            self.async_exponential_event.record()
        self.set_q_event(q, self.async_exponential_event)


class AscendTopKTopPSampler(TopKTopPSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.apply_top_k_top_p = apply_top_k_top_p

    def set_q_event(self, q, event):
        # Pass in async exponential results.
        # Also pass in event to prevent synchronize errors.
        self.q = q
        self.async_event = event

    def forward_native(self, logits, generators, k, p):
        """Override pytorch native implementation to torch_npu"""
        # when batch_invariant mode is enabled, we should use vllm's implementation.
        # or it will make batch_invariant mode not working.
        if vllm_is_batch_invariant():
            return super().forward_native(logits, generators, k, p)
        if _TOPK_LOGITS_DEBUG:
            # Stats must be taken on *pre*-mask logits (remain-path trigger).
            pre = logits.detach()
            torch.npu.synchronize()
            t0 = time.perf_counter()
            logits = self.apply_top_k_top_p(logits, k, p)
            torch.npu.synchronize()
            _debug_dump_topk_logits(pre, k, p, op_ms=(time.perf_counter() - t0) * 1e3)
        else:
            logits = self.apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        if get_ascend_config().enable_async_exponential:
            # Add synchronize to prevent synchronize error.
            self.async_event.synchronize()
            return probs.div_(self.q).argmax(dim=-1).view(-1), logits_to_return
        return random_sample(probs, generators), logits_to_return


def _apply_top_k_top_p_pytorch(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is None and k is None:
        return logits

    probs = logits.softmax(dim=-1)
    probs_sort, _ = probs.sort(dim=-1, descending=False)

    if k is not None:
        top_k_count = probs_sort.size(1) - k.to(torch.long)  # shape: (batch, )
        top_k_count = top_k_count.unsqueeze(dim=1)
        top_k_cutoff = probs_sort.gather(-1, top_k_count)

        # Make sure the no top-k rows are no-op.
        no_top_k_mask = (k == logits.shape[1]).unsqueeze(dim=1)
        top_k_cutoff.masked_fill_(no_top_k_mask, -float("inf"))

        elements_to_discard = probs < top_k_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    if p is not None:
        cumprob = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = cumprob <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False  # at least one

        top_p_count = top_p_mask.sum(dim=-1).unsqueeze(1)
        top_p_cutoff = probs_sort.gather(-1, top_p_count)
        elements_to_discard = probs < top_p_cutoff
        logits.masked_fill_(elements_to_discard, -float("inf"))

    return logits


def _apply_top_k_top_p_ascendc(
    logits: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor,
) -> torch.Tensor:
    if p is None and k is None:
        return logits
    return torch.ops._C_ascend.npu_apply_top_k_top_p(logits, k=k, p=p)


def _use_ascendc_top_k_top_p() -> bool:
    """A2/A3 use AscendC op by default.

    Set VLLM_ASCEND_FORCE_PYTORCH_TOPK=1 to force the PyTorch reference path.
    Useful to bisect graph+FIA regressions in ApplyTopKTopPCustom (~25ms
    stalls observed at decode batch≈16 with top_k).
    """
    import os

    if os.getenv("VLLM_ASCEND_FORCE_PYTORCH_TOPK", "0") == "1":
        return False
    return get_ascend_device_type() in [AscendDeviceType.A2, AscendDeviceType.A3]


apply_top_k_top_p = (
    _apply_top_k_top_p_ascendc
    if _use_ascendc_top_k_top_p()
    else _apply_top_k_top_p_pytorch
)
