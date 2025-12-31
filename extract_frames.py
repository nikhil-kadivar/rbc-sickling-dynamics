import os
import re
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, Iterable
import concurrent.futures as cf

import numpy as np
from PIL import Image
import cv2


# ----------------------------
# Constants / Supported formats
# ----------------------------
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpeg", ".mpg", ".wmv"}


# ----------------------------
# Helpers you referenced
# ----------------------------
def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "video"


def parse_size(s: str) -> Optional[Tuple[int, int]]:
    """
    Parse WIDTHxHEIGHT string -> (W, H).
    Accepts e.g. '1080x1620' or '1080X1620'.
    """
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in {"none", "null", "no", "false"}:
        return None
    m = re.match(r"^\s*(\d+)\s*[x]\s*(\d+)\s*$", s)
    if not m:
        raise argparse.ArgumentTypeError(f"Invalid --target-size '{s}'. Expected WIDTHxHEIGHT, e.g. 1080x1620.")
    w = int(m.group(1))
    h = int(m.group(2))
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive integers.")
    return (w, h)


class Sampler:
    """
    Sampling policy:
      - all_frames: save every frame
      - every_sec: save one frame every N seconds (based on timestamp if available, else index/fps)
      - every_n_frames: save one frame every N frames
    If none provided, defaults to every_n_frames=1 (save all).
    """
    def __init__(self, every_sec: Optional[float], every_n_frames: Optional[int], all_frames: bool):
        self.every_sec = every_sec
        self.every_n_frames = every_n_frames
        self.all_frames = all_frames

        if not all_frames and every_sec is None and every_n_frames is None:
            self.every_n_frames = 1

        self._next_t = 0.0  # for every_sec mode

    def should_save(self, t_sec: Optional[float], frame_idx: int) -> bool:
        if self.all_frames:
            return True

        if self.every_n_frames is not None:
            n = max(1, int(self.every_n_frames))
            return (frame_idx % n) == 0

        if self.every_sec is not None:
            # If we don't have time, fall back to saving every frame (conservative)
            if t_sec is None:
                return True
            if t_sec + 1e-12 >= self._next_t:
                self._next_t = t_sec + float(self.every_sec)
                return True
            return False

        return True


# ----------------------------
# CLAHE preprocessing (fixed params; matches your request)
# ----------------------------
_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID = 8


def _to_uint8(gray: np.ndarray) -> np.ndarray:
    """Ensure grayscale is uint8 in [0,255]."""
    if gray.dtype == np.uint8:
        return gray
    g = gray.astype(np.float32)
    gmin = float(np.min(g))
    gmax = float(np.max(g))
    if gmax <= gmin + 1e-12:
        return np.zeros_like(gray, dtype=np.uint8)
    g = (g - gmin) / (gmax - gmin)
    return (255.0 * g).clip(0, 255).astype(np.uint8)


def _apply_clahe(gray_u8: np.ndarray) -> np.ndarray:
    """Apply CLAHE to uint8 grayscale."""
    clahe = cv2.createCLAHE(
        clipLimit=float(_CLAHE_CLIP_LIMIT),
        tileGridSize=(int(_CLAHE_TILE_GRID), int(_CLAHE_TILE_GRID)),
    )
    return clahe.apply(gray_u8)


# ----------------------------
# Your original functions (kept) + CLAHE inserted in process_video
# ----------------------------
def save_frame(
    out_path: str,
    gray_array,
    target_size: Optional[Tuple[int, int]] = None,
) -> str:
    """Resize (if requested) and write a grayscale PNG to out_path. Returns out_path."""
    img = Image.fromarray(gray_array)
    if target_size is not None:
        # PIL expects (width, height)
        img = img.resize(target_size, Image.BILINEAR)
    img.save(out_path)
    return out_path


def is_video_file(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXTS


def scan_videos(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and is_video_file(p):
            yield p


def process_video(
    video_path: Path,
    imagesTs_dir: Path,
    prefix: str,
    sampler: Sampler,
    target_size: Optional[Tuple[int, int]],
    workers: int,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (frame_count / fps) if fps > 0 else None

    if duration_sec is not None:
        print(f"→ {video_path.name} | FPS: {fps:.3f} | Frames: {frame_count} | Duration: {duration_sec:.2f}s")
    else:
        print(f"→ {video_path.name} | FPS: {fps:.3f} | Frames: {frame_count} | Duration: unknown")

    video_slug = slugify(video_path.stem)  # kept (even if unused)
    saved = 0

    # Concurrency: use process pool to parallelize the CPU work of resizing/encoding
    # Use all cores if workers == 0
    max_workers = (os.cpu_count() or 1) if workers == 0 else max(1, workers)
    inflight_limit = max_workers * 4  # modest backpressure to avoid memory blow-up
    futures = []

    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Prefer container timestamp, fallback to index-based
            t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t_ms and t_ms > 0:
                t_sec = t_ms / 1000.0
            else:
                t_sec = (frame_idx / fps) if fps > 0 else None

            if sampler.should_save(t_sec, frame_idx):
                # ------------------------------
                # Added preprocessing:
                # grayscale -> uint8 -> CLAHE
                # ------------------------------
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_u8 = _to_uint8(gray)
                gray_u8 = _apply_clahe(gray_u8)

                out_name = f"{prefix}_{saved:03d}_0000.png"
                out_path = str(imagesTs_dir / out_name)

                # Backpressure: wait for the earliest future if too many are inflight
                if len(futures) >= inflight_limit:
                    cf.wait([futures.pop(0)], timeout=None)

                futures.append(ex.submit(save_frame, out_path, gray_u8, target_size))
                saved += 1

            frame_idx += 1

        # finalize
        for fut in cf.as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] Saving frame failed: {e}")

    cap.release()
    print(f"   Saved {saved} frames → {imagesTs_dir}")
    return saved


def probe_frame_size(video_path: Path) -> Optional[Tuple[int, int]]:
    """Return (width, height) of the first frame in the given video, or None if unreadable."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        return None
    h, w = frame.shape[:2]
    return (w, h)


def write_size_json(dst_dir: Path, size_wh: Tuple[int, int]) -> None:
    """Write a single JSON file with the (width, height) that frames are saved at."""
    w, h = size_wh
    payload = {"width": int(w), "height": int(h)}
    with open(dst_dir / "image_size.json", "w") as f:
        json.dump(payload, f, indent=2)


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract frames from videos into nnU-Net imagesTs format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--video",
        type=Path,
        help="Path to a single video file, this is just to process single video. If you prefer lots of videos to be processed together. Please use the input-dir command",
    )
    src.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing multiple videos to process",
    )

    ap.add_argument(
        "--out-base-dir",
        type=Path,
        default="Frames_for_inference",
        help="Base output directory where extracted and renamed frames will be saved",
    )
    ap.add_argument("--task-id", type=int, default=101, help="nnU-Net Task ID (e.g., 101)")
    ap.add_argument("--task-name", type=str, default="Experiment1", help="nnU-Net Task Name (e.g., experiment1)")
    ap.add_argument("--prefix", type=str, default="Flow", help="Prefix for saved frames")

    # Sampling policy (mutually exclusive)
    pol = ap.add_mutually_exclusive_group()
    pol.add_argument("--every-sec", type=float, help="Save one frame every N seconds")
    pol.add_argument("--every-n-frames", type=int, help="Save one frame every N frames")
    pol.add_argument("--all-frames", action="store_true", help="Save all frames")

    ap.add_argument(
        "--target-size",
        type=parse_size,
        default="1000x1000",
        help="Resize to WIDTHxHEIGHT (PIL expects width,height), The number 1080*1620 is default and we arrived at this since we trained our nnunet model on this width and height. We observed doing inference on the same size at which nnunet is trained gives better accuracy. We recommend using this width and height if you are planning to use the pre-trained nnunet weights from this github repo. If you plan to train from scratch on your own experiment videos or frames. Please adjust this accordingly according to the training dataset.",
    )
    ap.add_argument("--workers", type=int, default=0, help="Number of processes for saving (0 = use all cores)")

    args = ap.parse_args(list(argv) if argv is not None else None)

    # Resolve sampling policy
    sampler = Sampler(args.every_sec, args.every_n_frames, args.all_frames)

    # Prepare nnU-Net imagesTs directory
    task_folder = f"Task{args.task_id:03d}_{args.task_name}"
    imagesTs_dir = args.out_base_dir / task_folder / "imagesTs"
    imagesTs_dir.mkdir(parents=True, exist_ok=True)

    # Figure out which videos to process
    videos: Iterable[Path]
    if args.video is not None:
        if not args.video.exists():
            ap.error(f"Video does not exist: {args.video}")
        videos = [args.video]
    else:
        if not args.input_dir.exists() or not args.input_dir.is_dir():
            ap.error(f"--input-dir must be a directory: {args.input_dir}")
        videos = list(scan_videos(args.input_dir))
        if not videos:
            ap.error(f"No video files found in {args.input_dir} (supported: {sorted(VIDEO_EXTS)})")

    # Decide the output size (shared for all frames)
    if args.target_size is not None:
        save_size = args.target_size  # (W, H) from --target-size
    else:
        save_size = None
        for vp in videos:
            sz = probe_frame_size(vp)
            if sz is not None:
                save_size = sz
                break
        if save_size is None:
            ap.error("Could not determine frame size from any input video and no --target-size was provided.")

    # Persist a single JSON with the size all frames will have
    write_size_json(imagesTs_dir, save_size)

    # Summary
    print("\nConfiguration")
    print("=============")
    print(f"Task folder : {imagesTs_dir.parent}")
    print(f"imagesTs    : {imagesTs_dir}")
    print(f"Prefix      : {args.prefix}")
    if args.target_size:
        print(f"Target size : {args.target_size[0]}x{args.target_size[1]} (W×H)")
    else:
        print("Target size : native (no resize)")
    if args.all_frames:
        print("Sampling    : ALL frames")
    elif args.every_sec is not None:
        print(f"Sampling    : every {args.every_sec} second(s)")
    elif args.every_n_frames is not None:
        print(f"Sampling    : every {args.every_n_frames} frame(s)")
    print(f"Workers     : {'ALL CORES' if args.workers == 0 else args.workers}")

    # Preprocess summary (fixed, no CLI change)
    print(f"Preprocess  : grayscale + CLAHE (clip_limit={_CLAHE_CLIP_LIMIT}, tile_grid={_CLAHE_TILE_GRID})")

    total_saved = 0
    for vid in videos:
        total_saved += process_video(
            vid,
            imagesTs_dir,
            args.prefix,
            sampler,
            args.target_size,
            args.workers,
        )

    print("\nDone.")
    print(f"Total frames saved: {total_saved}")
    print(f"Output imagesTs: {imagesTs_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

