"""Model-agnostic inference latency profiling.

Provides a generic ``LatencyProfiler`` that times an arbitrary callable
on CPU or CUDA with warm-up, repeated runs, and summary statistics.
"""

from __future__ import annotations

import gc
import statistics
import time
from typing import Callable

import torch


class LatencyProfiler:
    """Measure wall-clock latency of an arbitrary callable.

    Handles:
        * CUDA / CPU device-aware timing
        * warm-up runs (discarded)
        * repeated runs with summary statistics

    The callable must take no arguments and run a single inference unit
    whose latency is to be measured.
    """

    def __init__(
        self,
        device: torch.device,
        n_warmup: int = 10,
        n_runs: int = 100,
    ):
        self.device = torch.device(device)
        self.n_warmup = int(n_warmup)
        self.n_runs = int(n_runs)

    def measure(self, fn: Callable[[], object]) -> dict:
        """Run ``fn`` repeatedly and return latency statistics in ms."""
        if self.device.type == "cuda":
            samples_ms = self._measure_cuda(fn)
        else:
            samples_ms = self._measure_cpu(fn)
        return self._summarize(samples_ms)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _measure_cuda(self, fn: Callable[[], object]) -> list:
        torch.cuda.synchronize(self.device)
        for _ in range(self.n_warmup):
            fn()
        torch.cuda.synchronize(self.device)

        starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.n_runs)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.n_runs)]

        for i in range(self.n_runs):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize(self.device)

        return [starts[i].elapsed_time(ends[i]) for i in range(self.n_runs)]

    def _measure_cpu(self, fn: Callable[[], object]) -> list:
        for _ in range(self.n_warmup):
            fn()

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            samples = []
            for _ in range(self.n_runs):
                t0 = time.perf_counter()
                fn()
                t1 = time.perf_counter()
                samples.append((t1 - t0) * 1000.0)
            return samples
        finally:
            if gc_was_enabled:
                gc.enable()

    @staticmethod
    def _summarize(samples_ms: list) -> dict:
        if not samples_ms:
            return {}
        sorted_samples = sorted(samples_ms)
        n = len(sorted_samples)
        p95_idx = min(n - 1, int(round(0.95 * (n - 1))))
        return {
            "median_ms": statistics.median(sorted_samples),
            "mean_ms": statistics.fmean(sorted_samples),
            "std_ms": statistics.pstdev(sorted_samples) if n > 1 else 0.0,
            "p95_ms": sorted_samples[p95_idx],
            "min_ms": sorted_samples[0],
            "max_ms": sorted_samples[-1],
            "n_runs": n,
        }
