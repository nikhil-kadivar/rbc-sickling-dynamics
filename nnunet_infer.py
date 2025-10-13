#!/usr/bin/env python3
"""
SickleCell — nnU-Net v2 Inference Orchestrator
----------------------------------------------

Runs nnUNetv2_predict on an imagesTs folder with flexible parallelism that works
on **GPU(s), CPU, or Apple MPS**, and automatically chooses `-num_parts` based on
the compute you give it (GPUs or CPU workers). You can still override parts
manually when you want.

Key behavior
* **Device-agnostic**: `--device auto|cuda|cpu|mps` (default: auto-detects; prefers CUDA, then MPS, else CPU).
* **Auto parts**:
  - **CUDA**: if you pass `--gpus 0,1,2`, it spawns **3 parts** (one per GPU).
    If you pass `--gpus 0,1,2,3,4` but want 4 parts, add `--parts 4` (round‑robin onto GPUs).
  - **CPU**: if you pass `--cpu-workers 4`, it spawns **4 parts** on CPU.
  - **MPS** (Apple): default is **1 part** (you can raise with `--cpu-workers N`, but be mindful of oversubscription).
* **npp / nps**: per‑process worker pools inside nnU‑Net for preprocessing and
  export; they are **independent** of `-num_parts` (process count).
* **Clean logs**: by default the script passes `--disable_progress_bar` to
  `nnUNetv2_predict` so parallel tqdm bars don’t overlap. Use
  `--enable-progress-bar` to show them.

**nnU‑Net `-device` values**: 'cuda' (GPU), 'cpu' (CPU), 'mps' (Apple M1/M2).
Do **not** use `-device` to pick a GPU ID — use `CUDA_VISIBLE_DEVICES` instead
(the script handles this for you per process).

Requirements: nnU‑Net v2 installed and `nnUNetv2_predict` available on PATH.

Examples
--------
# CUDA: parts auto from GPUs passed (0,1,2) -> parts = 3 (one per GPU)
python sicklecell_nnunet_infer.py \
  --imagesTs ./Task109_try1/imagesTs \
  --out-dir ./gpu_out \
  --dataset-id 102 --configuration 2d --checkpoint checkpoint_best.pth \
  --gpus 0,1,2

# CUDA: five GPUs listed, cap parts to 4
python sicklecell_nnunet_infer.py \
  --imagesTs ./Task109_try1/imagesTs \
  --out-dir ./gpu_out \
  --dataset-id 102 --configuration 2d --checkpoint checkpoint_best.pth \
  --gpus 0,1,2,3,4 --parts 4

# CPU: 4 parallel parts (processes), 2 math threads per process, lightweight nnU‑Net pools
python sicklecell_nnunet_infer.py \
  --imagesTs ./Task109_try1/imagesTs \
  --out-dir ./cpu_out \
  --dataset-id 102 --configuration 2d \
  --device cpu --cpu-workers 4 --torch-threads 2 --npp 1 --nps 1

# MPS (Apple Silicon): single process by default
python sicklecell_nnunet_infer.py \
  --imagesTs ./Task109_try1/imagesTs \
  --out-dir ./mps_out \
  --dataset-id 102 --configuration 2d \
  --device mps
"""
from __future__ import annotations

import argparse
import os
import platform
import shlex
import subprocess as sp
from pathlib import Path
from typing import List, Optional


# ---------------------------- CLI & Utilities ---------------------------- #

def comma_list_to_ints(s: str) -> List[int]:
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        raise argparse.ArgumentTypeError("--gpus must be a comma-separated list of integers, e.g., 0,1,2")


def detect_gpu_ids() -> List[int]:
    """Detect available GPU IDs using nvidia-smi. Returns [] if none."""
    try:
        out = sp.check_output(["nvidia-smi", "-L"], stderr=sp.DEVNULL, text=True)
    except Exception:
        return []
    ids: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("GPU "):
            try:
                gid = int(line.split()[1].rstrip(":"))
                ids.append(gid)
            except Exception:
                continue
    return ids


def is_mps_available() -> bool:
    if platform.system() != "Darwin":
        return False
    try:
        import torch  # type: ignore
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def build_predict_cmd(
    imagesTs: Path,
    out_dir: Path,
    dataset_id: int,
    configuration: str,
    checkpoint: str,
    npp: int,
    nps: int,
    parts: int,
    part_id: int,
    device: str,
    disable_progress_bar: bool,
) -> List[str]:
    cmd = [
        "nnUNetv2_predict",
        "-i", str(imagesTs),
        "-o", str(out_dir),
        "-d", str(dataset_id),
        "-c", configuration,
        "-chk", checkpoint,
        "-npp", str(npp),
        "-nps", str(nps),
        "-num_parts", str(parts),
        "-part_id", str(part_id),
        "-device", device,
    ]
    if disable_progress_bar:
        cmd.append("--disable_progress_bar")
    return cmd


def pretty_cmd_for_print(cmd: List[str]) -> str:
    """Join command for logging, but show -device value wrapped in single quotes.
    (Execution still passes the unquoted value; this is just for readability.)
    """
    out = []
    i = 0
    while i < len(cmd):
        tok = cmd[i]
        if tok == "-device" and i + 1 < len(cmd):
            out.append(shlex.quote(tok))
            out.append("'" + cmd[i + 1] + "'")  # human-friendly quotes in log
            i += 2
            continue
        out.append(shlex.quote(tok))
        i += 1
    return " ".join(out)


def set_cpu_thread_env(env: dict, threads: Optional[int]) -> None:
    if threads and threads > 0:
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            env[var] = str(threads)


def run_parts(
    imagesTs: Path,
    out_dir: Path,
    dataset_id: int,
    configuration: str,
    checkpoint: str,
    npp: int,
    nps: int,
    parts: int,
    device: str,
    gpu_ids: List[int],
    dry_run: bool,
    torch_threads: Optional[int],
    disable_progress_bar: bool,
) -> int:
    procs: List[sp.Popen] = []

    for part_id in range(parts):
        cmd = build_predict_cmd(
            imagesTs, out_dir, dataset_id, configuration, checkpoint,
            npp, nps, parts, part_id, device, disable_progress_bar
        )
        env = os.environ.copy()

        if device == "cuda":
            if not gpu_ids:
                print("[WARN] --device cuda requested but no GPUs detected. Falling back to CPU for this part.")
                cmd[-1] = "cpu"  # change device value but keep '-device' flag
                env.pop("CUDA_VISIBLE_DEVICES", None)
                set_cpu_thread_env(env, torch_threads)
                log_map = f"CPU{f' (threads per proc: {torch_threads})' if torch_threads else ''}"
            else:
                chosen = gpu_ids[part_id % len(gpu_ids)]
                env["CUDA_VISIBLE_DEVICES"] = str(chosen)
                log_map = f"CUDA_VISIBLE_DEVICES={chosen}"
        elif device == "mps":
            env.pop("CUDA_VISIBLE_DEVICES", None)
            set_cpu_thread_env(env, torch_threads)
            log_map = "MPS"
        else:  # cpu
            env.pop("CUDA_VISIBLE_DEVICES", None)
            set_cpu_thread_env(env, torch_threads)
            log_map = f"CPU{f' (threads per proc: {torch_threads})' if torch_threads else ''}"

        print(f"[map] part_id={part_id} -> {log_map}")
        print(f"[nnUNet] part {part_id+1}/{parts}: {pretty_cmd_for_print(cmd)}")
        if dry_run:
            continue
        procs.append(sp.Popen(cmd, env=env))

    if dry_run:
        return 0

    exit_code = 0
    for p in procs:
        try:
            rc = p.wait()
        except KeyboardInterrupt:
            for q in procs:
                try:
                    q.terminate()
                except Exception:
                    pass
            raise
        if rc != 0:
            exit_code = rc
    return exit_code


# --------------------------------- Main --------------------------------- #

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Orchestrate nnUNetv2_predict across parts on GPU(s)/CPU/MPS with smart defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--imagesTs", type=Path, required=True, help="Path to nnU-Net imagesTs directory")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for predictions")
    ap.add_argument("--dataset-id", type=int, required=True, help="nnU-Net dataset id (e.g., 102)")
    ap.add_argument("--configuration", type=str, default="2d", help="nnU-Net configuration (e.g., 2d, 3d_lowres, 3d_fullres)")
    ap.add_argument("--checkpoint", type=str, default="checkpoint_best.pth", help="Checkpoint name or path")

    ap.add_argument("--npp", type=int, default=3, help="nnU-Net: number of preprocessing workers (per process)")
    ap.add_argument("--nps", type=int, default=3, help="nnU-Net: number of segmentation/export workers (per process)")

    # Parallelism controls
    ap.add_argument("--parts", type=int, default=0, help="Override number of parts/processes (0 = auto)")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto", help="Compute device for nnUNetv2_predict -device ('cuda' 'cpu' 'mps')")
    ap.add_argument("--gpus", type=comma_list_to_ints, default=None, help="Comma-separated GPU IDs when using CUDA (e.g., 0,1,2)")
    ap.add_argument("--cpu-workers", type=int, default=0, help="Number of parallel processes on CPU/MPS when parts is auto (0 = auto)")
    ap.add_argument("--torch-threads", type=int, default=0, help="Set CPU BLAS/OMP threads per process (0 = leave as is)")

    # Logging / UX
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    ap.add_argument("--enable-progress-bar", action="store_true", help="Show tqdm progress bars from nnU-Net (disabled by default)")

    args = ap.parse_args(argv if argv is not None else None)

    # Validate paths
    if not args.imagesTs.exists() or not args.imagesTs.is_dir():
        ap.error(f"imagesTs directory not found: {args.imagesTs}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve device & GPUs
    detected_gpus = detect_gpu_ids()
    gpu_ids = args.gpus if args.gpus is not None else detected_gpus

    if args.device == "auto":
        device = "cuda" if gpu_ids else ("mps" if is_mps_available() else "cpu")
    else:
        device = args.device

    # Decide parts
    if args.parts > 0:
        parts = args.parts
    else:
        if device == "cuda":
            parts = max(1, len(gpu_ids))
        elif device == "mps":
            parts = args.cpu_workers if args.cpu_workers > 0 else 1
        else:  # cpu
            logical = os.cpu_count() or 2
            auto_cpu = max(1, logical // 2)
            parts = args.cpu_workers if args.cpu_workers > 0 else auto_cpu

    print("Configuration")
    print("=============")
    print(f"imagesTs       : {args.imagesTs}")
    print(f"out-dir        : {args.out_dir}")
    print(f"dataset-id     : {args.dataset_id}")
    print(f"configuration  : {args.configuration}")
    print(f"checkpoint     : {args.checkpoint}")
    print(f"npp / nps      : {args.npp} / {args.nps} (per process)")
    print(f"device         : {device}")
    if device == "cuda":
        print(f"GPU IDs        : {gpu_ids if gpu_ids else '[] (none detected)'}")
    print(f"parts (processes): {parts}")
    if device in ("cpu", "mps") and args.torch_threads > 0:
        print(f"torch threads  : {args.torch_threads} per process")

    # --- run prediction across parts ---
    rc = run_parts(
        imagesTs=args.imagesTs,
        out_dir=args.out_dir,
        dataset_id=args.dataset_id,
        configuration=args.configuration,
        checkpoint=args.checkpoint,
        npp=args.npp,
        nps=args.nps,
        parts=parts,
        device=device,
        gpu_ids=gpu_ids,
        dry_run=args.dry_run,
        torch_threads=(args.torch_threads if args.torch_threads > 0 else None),
        disable_progress_bar=(not args.enable_progress_bar),
    )

    if rc == 0:
        print("All nnU-Net prediction parts completed successfully.")
    else:
        print(f"One or more nnU-Net prediction parts failed with exit code {rc}.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

