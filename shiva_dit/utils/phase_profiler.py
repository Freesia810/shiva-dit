"""Per-phase timing of the REAL inference path.

`benchmark/phase_breakdown.py` times a synthetic single-block rig, and it runs the
ShivaRatioPolicy network even though deployment reads a LUT -- so its cost split does
not describe inference. This instruments the actual hook instead: enable it, run the
pipeline for real, and every phase is accumulated over all layers x all denoising
steps, with the LUT on the path exactly as deployed.

Disabled by default and short-circuited on a module flag, so the instrumented hook
costs nothing in normal runs.

    from shiva_dit.utils import phase_profiler as pp
    pp.enable(); pp.reset()
    pipe(prompt, num_inference_steps=50)
    totals, counts = pp.report()
"""

from collections import defaultdict

import torch

ENABLED = False
_totals = defaultdict(float)
_counts = defaultdict(int)
_pending = []


class phase:
    """CUDA-event timer for one phase. Events are recorded on the stream and only
    read after a synchronize, so timing never stalls the pipeline mid-run."""

    __slots__ = ("name", "_s", "_e")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if ENABLED:
            self._s = torch.cuda.Event(enable_timing=True)
            self._e = torch.cuda.Event(enable_timing=True)
            self._s.record()
        return self

    def __exit__(self, *exc):
        if ENABLED:
            self._e.record()
            _pending.append((self.name, self._s, self._e))
        return False


_open = {}


def start(name, key=None):
    """Begin a phase that cannot be expressed as a `with` block -- e.g. a pair of
    PyTorch forward pre/post hooks, which is how the whole-block cost is timed on
    both the pruned and the unpruned pipeline."""
    if ENABLED:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        _open[(name, key)] = e


def stop(name, key=None):
    if ENABLED:
        s = _open.pop((name, key), None)
        if s is not None:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            _pending.append((name, s, e))


def enable():
    global ENABLED
    ENABLED = True


def disable():
    global ENABLED
    ENABLED = False


def reset():
    _totals.clear()
    _counts.clear()
    _pending.clear()
    _open.clear()


def flush():
    if _pending:
        torch.cuda.synchronize()
        for name, s, e in _pending:
            _totals[name] += s.elapsed_time(e)
            _counts[name] += 1
        _pending.clear()


def report():
    """-> ({phase: total_ms}, {phase: n_calls}) accumulated since the last reset."""
    flush()
    return dict(_totals), dict(_counts)
