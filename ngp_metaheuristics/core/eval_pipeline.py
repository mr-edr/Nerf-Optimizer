"""
core/eval_pipeline.py

Owns the actual "run Instant-NGP training for N steps, render one frame,
compute PSNR against ground truth" logic. This file lives entirely in
ngp_metaheuristics -- it CALLS instant-ngp/scripts/run.py by path, but
nothing is ever written into the instant-ngp repo itself. That repo
should stay a clean, unmodified upstream dependency (it's a git
submodule) so it can be updated/rebuilt independently of your study code.

You pass the path to instant-ngp's checkout via --ngp_dir (or the
NGP_DIR env var) so this script knows where scripts/run.py lives.

This is invoked as a subprocess from core/fitness.py -- it is intended
to be run standalone too, for manual debugging:

    python3 -m core.eval_pipeline \\
        --ngp_dir ../instant-ngp \\
        --scene ../instant-ngp/data/nerf/fox \\
        --config /tmp/some_generated_config.json \\
        --steps 100 --frame_idx 0 --width 32 --height 32 --spp 1
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
import uuid

from PIL import Image
import numpy as np


def call_run_py(ngp_dir, args_list, timeout=None, verbose=True):
    """
    Invokes instant-ngp's own scripts/run.py as a subprocess, returning
    (success, combined_stdout). When verbose=True, streams output live
    (useful for standalone debugging via `python -m core.eval_pipeline`).
    When verbose=False (the default when called from fitness.py during
    real optimizer runs), output is captured silently -- it's still
    available in the returned string for failure diagnosis, it's just
    not printed line-by-line to the terminal.

    Uses the SAME python interpreter (sys.executable) that's running
    this script -- if pyngp is only importable in a specific conda env
    (e.g. `ga-ngp` or `build-env`), you MUST launch this whole pipeline
    from within that env's python, since subprocess inherits the
    interpreter, not just the shell's activated env.
    """
    run_py = os.path.join(ngp_dir, "scripts", "run.py")
    if not os.path.exists(run_py):
        raise FileNotFoundError(
            f"Could not find {run_py} -- check --ngp_dir points at your "
            f"instant-ngp checkout root (the directory containing scripts/run.py)."
        )

    cmd = [sys.executable, run_py] + args_list

    if not verbose:
        try:
            proc = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, timeout=timeout,
            )
            return proc.returncode == 0, proc.stdout
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") + "\n*** TIMEOUT ***"
            return False, out

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True,
    )

    out_lines = []
    start = time.time()
    try:
        for line in iter(proc.stdout.readline, ""):
            if line == "":
                break
            out_lines.append(line)
            print(line, end="")
            if timeout and (time.time() - start) > timeout:
                proc.kill()
                return False, "".join(out_lines) + "\n*** TIMEOUT ***"
        proc.wait()
    except KeyboardInterrupt:
        proc.kill()
        raise

    return proc.returncode == 0, "".join(out_lines)


def compute_psnr(rendered_path, gt_path):
    a = Image.open(rendered_path).convert("RGB")
    b = Image.open(gt_path).convert("RGB")
    if a.size != b.size:
        b = b.resize(a.size, Image.BICUBIC)

    a = np.asarray(a).astype(np.float32) / 255.0
    b = np.asarray(b).astype(np.float32) / 255.0

    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(1.0 / mse)


def find_transforms(transforms_arg, scene):
    if transforms_arg:
        return transforms_arg
    cand = os.path.join(scene, "transforms.json")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(
        "No transforms.json found -- pass --transforms explicitly or "
        "ensure the scene directory contains transforms.json."
    )


def resolve_gt_path(transforms_path, scene, frame):
    gt_rel = frame.get("file_path") or frame.get("file") or frame.get("filePath")
    if not gt_rel:
        raise ValueError("Selected frame has no 'file_path' entry pointing at a ground-truth image.")

    transforms_dir = os.path.dirname(transforms_path)
    gt_path = gt_rel if os.path.isabs(gt_rel) else os.path.normpath(os.path.join(transforms_dir, gt_rel))

    if os.path.exists(gt_path):
        return gt_path
    for ext in (".png", ".jpg", ".jpeg"):
        if os.path.exists(gt_path + ext):
            return gt_path + ext

    alt = os.path.join(scene, os.path.basename(gt_path))
    if os.path.exists(alt):
        return alt

    raise FileNotFoundError(f"Ground-truth image not found for frame (tried {gt_path}).")


def run_one_frame_psnr(ngp_dir, scene, config_path, steps, frame_idx,
                        width=64, height=64, spp=1, transforms=None,
                        timeout_train=3600, timeout_eval=300,
                        work_dir="ga_eval_work", verbose=False):
    """
    Returns (psnr: float, ok: bool, combined_stdout: str).
    Raises only on programmer-error-type problems (missing scripts/run.py,
    malformed transforms.json); anything that's a legitimate "this config
    failed to train/render" outcome is returned as ok=False instead of
    raised, so callers (core/fitness.py) can turn it into a fitness
    penalty rather than crashing the whole optimizer run.

    verbose: if True, streams the raw instant-ngp training/render output
        live to the terminal (useful for standalone debugging). Defaults
        to False so optimizer runs (GA/DE/PSO/etc, hundreds of evals)
        stay readable -- callers should print their own concise
        generation/individual/fitness/time summary instead.
    """
    os.makedirs(work_dir, exist_ok=True)
    snaps_dir = os.path.join(work_dir, "snaps")
    renders_dir = os.path.join(work_dir, "renders")
    os.makedirs(snaps_dir, exist_ok=True)
    os.makedirs(renders_dir, exist_ok=True)

    snap_path = os.path.join(snaps_dir, f"snap_{uuid.uuid4().hex[:8]}.ingp")

    # --- 1) Train and save snapshot ---
    train_args = [
        "--scene", scene,
        "--network", config_path,
        "--train",
        "--n_steps", str(steps),
        "--width", str(width), "--height", str(height),
        "--save_snapshot", snap_path,
    ]
    ok_train, train_out = call_run_py(ngp_dir, train_args, timeout=timeout_train, verbose=verbose)
    if not ok_train:
        return -1.0, False, train_out  # hard fail -- do NOT continue to eval

    if not os.path.exists(snap_path):
        return -1.0, False, train_out + "\n*** snapshot file was never written ***"

    # --- 2) Locate ground-truth frame ---
    transforms_path = find_transforms(transforms, scene)
    with open(transforms_path) as f:
        transforms_data = json.load(f)
    frames = transforms_data.get("frames") or transforms_data.get("images")
    if not frames:
        raise ValueError(f"{transforms_path} contains no 'frames' list.")
    if not (0 <= frame_idx < len(frames)):
        raise ValueError(f"frame_idx {frame_idx} out of range (0..{len(frames)-1}).")

    gt_path = resolve_gt_path(transforms_path, scene, frames[frame_idx])

    # --- 3) Render single frame from the saved snapshot (n_steps=0, no retrain) ---
    run_render_dir = os.path.join(renders_dir, uuid.uuid4().hex[:8])
    os.makedirs(run_render_dir, exist_ok=True)
    eval_args = [
        "--scene", scene,
        "--load_snapshot", snap_path,
        "--screenshot_transforms", transforms_path,
        "--screenshot_dir", run_render_dir,
        "--screenshot_frames", str(frame_idx),
        "--screenshot_spp", str(spp),
        "--width", str(width),
        "--height", str(height),
        "--n_steps", "0",
    ]
    ok_render, render_out = call_run_py(ngp_dir, eval_args, timeout=timeout_eval, verbose=verbose)
    combined_out = train_out + "\n" + render_out
    if not ok_render:
        return -1.0, False, combined_out

    pngs = sorted(
        glob.glob(os.path.join(run_render_dir, "*.png"))
        + glob.glob(os.path.join(run_render_dir, "*.jpg"))
        + glob.glob(os.path.join(run_render_dir, "*.jpeg"))
    )
    if not pngs:
        return -1.0, False, combined_out + "\n*** no rendered image (png/jpg) produced ***"

    gt_basename = os.path.basename(gt_path)
    rendered_path = next((p for p in pngs if gt_basename in os.path.basename(p)), pngs[0])

    try:
        psnr_val = compute_psnr(rendered_path, gt_path)
    except Exception as e:
        return -1.0, False, combined_out + f"\n*** PSNR computation failed: {e} ***"

    return psnr_val, True, combined_out + f"\n=== PSNR (frame {frame_idx}) === {psnr_val:.6f}"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ngp_dir", default=os.environ.get("NGP_DIR", "../instant-ngp"),
                   help="path to instant-ngp checkout root (contains scripts/run.py)")
    p.add_argument("--scene", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--frame_idx", type=int, default=0)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--height", type=int, default=64)
    p.add_argument("--spp", type=int, default=1)
    p.add_argument("--transforms", default=None)
    p.add_argument("--timeout_train", type=int, default=3600)
    p.add_argument("--timeout_eval", type=int, default=300)
    p.add_argument("--work_dir", default="ga_eval_work")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    psnr, ok, out = run_one_frame_psnr(
        a.ngp_dir, a.scene, a.config, a.steps, a.frame_idx,
        width=a.width, height=a.height, spp=a.spp, transforms=a.transforms,
        timeout_train=a.timeout_train, timeout_eval=a.timeout_eval,
        work_dir=a.work_dir, verbose=True,
    )
    print(f"\nok={ok} psnr={psnr}")
    sys.exit(0 if ok else 1)
