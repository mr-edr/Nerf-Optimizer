"""
core/fitness.py

Shared fitness evaluation for every optimizer. Wraps the train -> snapshot
-> render -> PSNR pipeline (core/eval_pipeline.py, which itself calls
instant-ngp/scripts/run.py by path -- nothing is ever written into the
instant-ngp checkout).

Fixes the two failure-handling gaps from the previous attempt:

  - non-zero exit code / timeout  -> hard fail, NOT "warning + continue"
  - NaN or missing PSNR           -> hard fail
  - OOM (CUDA out of memory)      -> hard fail, distinct penalty
  - VRAM usage during training    -> tracked via polling thread, folded
                                     into fitness as a soft penalty above
                                     a budget (so "just barely over" isn't
                                     as bad as an OOM crash)

All optimizers call `evaluate(cfg, scene, out_dir, args) -> FitnessResult`
and never touch subprocess/PSNR-parsing/eval_pipeline directly.
"""

from dataclasses import dataclass
from typing import Optional
import json
import os
import threading
import time
import uuid

from .eval_pipeline import run_one_frame_psnr

try:
    import pynvml
    _HAS_NVML = True
except ImportError:
    _HAS_NVML = False


# Large fixed penalty for anything that fails outright (OOM, crash, NaN,
# timeout, missing output). Keeping this MUCH lower than any real PSNR
# (PSNR is typically 15-40 for these scenes) ensures failed configs never
# get selected as elites, without producing -inf/NaN that could break
# optimizer math (e.g. CMA-ES covariance updates).
HARD_FAIL_PENALTY = -50.0


@dataclass
class FitnessResult:
    psnr: float                 # HARD_FAIL_PENALTY if failed
    ok: bool                    # True only if training+render+PSNR all succeeded
    peak_vram_mb: Optional[float]
    failure_reason: Optional[str]
    raw_stdout: str
    elapsed_s: float = 0.0


class _VRAMMonitor:
    """Polls GPU memory usage in a background thread during training."""

    def __init__(self, gpu_index: int = 0, interval_s: float = 0.5):
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self.peak_mb = 0.0
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        if _HAS_NVML:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def _poll_loop(self):
        while not self._stop.is_set():
            if self._handle is not None:
                info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                used_mb = info.used / (1024 ** 2)
                self.peak_mb = max(self.peak_mb, used_mb)
            time.sleep(self.interval_s)

    def __enter__(self):
        if self._handle is not None:
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if _HAS_NVML:
            pynvml.nvmlShutdown()


def _looks_like_oom(output: str) -> bool:
    lowered = output.lower()
    return ("out of memory" in lowered or "cuda_error_out_of_memory" in lowered
            or "cublas_status_alloc_failed" in lowered)


def _looks_like_nan(output: str) -> bool:
    lowered = output.lower()
    return "nan" in lowered and ("loss" in lowered or "psnr" in lowered)


def evaluate(cfg: dict, scene: str, out_dir: str, args,
             vram_budget_mb: float = 8000.0,
             vram_penalty_scale: float = 0.005) -> FitnessResult:
    """
    Writes `cfg` to disk, runs the train+render+PSNR pipeline in-process,
    and returns a FitnessResult. `args` must provide: ngp_dir, eval_steps,
    frame_idx, eval_width, eval_height, eval_spp, eval_timeout.

    vram_budget_mb: soft budget; usage above this incurs a penalty
        proportional to the overage, scaled by vram_penalty_scale.
        Tune these two once you know your GPU's actual VRAM ceiling.
    """
    os.makedirs(out_dir, exist_ok=True)
    cfg_path = os.path.join(out_dir, f"cfg_{uuid.uuid4().hex[:8]}.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    work_dir = os.path.join(out_dir, "eval_work")

    start = time.time()
    with _VRAMMonitor() as vram:
        psnr, ok, stdout = run_one_frame_psnr(
            ngp_dir=args.ngp_dir,
            scene=scene,
            config_path=cfg_path,
            steps=args.eval_steps,
            frame_idx=args.frame_idx,
            width=args.eval_width,
            height=args.eval_height,
            spp=args.eval_spp,
            timeout_train=args.eval_timeout,
            timeout_eval=max(300, args.eval_timeout // 4),
            work_dir=work_dir,
            verbose=False,
        )
    elapsed = time.time() - start

    peak_vram = vram.peak_mb if _HAS_NVML else None

    if not ok:
        if _looks_like_oom(stdout):
            return FitnessResult(HARD_FAIL_PENALTY, False, peak_vram, "OOM", stdout, elapsed)
        if _looks_like_nan(stdout):
            return FitnessResult(HARD_FAIL_PENALTY, False, peak_vram, "NaN_loss", stdout, elapsed)
        return FitnessResult(HARD_FAIL_PENALTY, False, peak_vram, "train_or_render_failed", stdout, elapsed)

    # --- Soft VRAM penalty (only applied to otherwise-successful runs) ---
    fitness = psnr
    if peak_vram is not None and peak_vram > vram_budget_mb:
        overage = peak_vram - vram_budget_mb
        fitness -= overage * vram_penalty_scale

    return FitnessResult(fitness, True, peak_vram, None, stdout, elapsed)

