"""
SickleCell Mask Counter + Visualizer (+ optional Watershed) — PARALLEL PLOTTING
-------------------------------------------------------------------------------
Given an nnU-Net-style imagesTs folder (original frames) and a folder of
corresponding mask PNGs (0=background, 1=healthy, 2=sickle), this tool:

1) Counts cells per class (healthy / sickle) per frame.
2) (Optional) Applies watershed splitting to separate touching cells.
3) Renders per-frame montages in parallel (same pool as counting).
4) Writes a summary table (CSV and optional Excel).
5) Exports a video stitched from the montage frames.
6) Plots and saves the sickling ratio curve: sickle / (healthy + sickle).

Robust pairing between mask and image is supported via a parsed frame number
from the mask filename's trailing integer token (e.g. "..._0007.png" → 7).
By default the script looks for images named like "*_{frame:03d}_0000.png",
but you can control the zero-fill with --frame-zfill.

Dependencies: numpy, pandas, pillow, matplotlib, scipy, scikit-image, opencv-python

Examples
--------
# Basic, no watershed, 8 workers, 3-digit frame IDs
python sicklecell_count_and_visualize.py --mask-folder /path/to/masks --image-folder /path/to/imagesTs --out-dir ./cell_count_outputs --min-chunk-size 1000 --workers 8

# With watershed (min_distance 22), write Excel too, 4-digit frame IDs, 6 fps video
python sicklecell_count_and_visualize.py --mask-folder /path/to/masks --image-folder /path/to/imagesTs --out-dir ./cell_count_outputs_ws --watershed --min-distance 22 --excel-out counts.xlsx --frame-zfill 4 --fps 6
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import math
import concurrent.futures as cf
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")  # headless (safe in subprocesses)
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from scipy.ndimage import center_of_mass
from skimage.morphology import remove_small_objects
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as sk_watershed
import cv2
import json


from matplotlib.colors import ListedColormap, BoundaryNorm

# 0=background, 1=healthy, 2=sickled
cmap = ListedColormap([
    (0.0, 0.0, 0.0, 1.0),  # background (black)
    (0.0, 1.0, 0.0, 1.0),  # healthy (green)
    (1.0, 0.0, 0.0, 1.0),  # sickled (red)
])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def load_size_json(dst_dir: Path) -> Optional[Tuple[int, int]]:
    """Load (width, height) from imagesTs/image_size.json if present."""
    path = dst_dir / "image_size.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        w = int(data["width"])
        h = int(data["height"])
        return (w, h)
    except Exception as e:
        print(f"[WARN] Could not read {path.name}: {e}")
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Count sickle/healthy cells per frame with optional watershed, save montages, table, and a video.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--mask-folder", type=Path, default = "nnunet_masks_out", help="Folder with mask PNGs (0=bg,1=healthy,2=sickle)")
    ap.add_argument("--image-folder", type=Path, default = "Frames_for_inference/Task101_Experiment1/imagesTs/", help="imagesTs folder with original frames")
    ap.add_argument("--out-dir", type=Path, default = "Outputs", help="Output directory (figures, tables, video)")

    ap.add_argument("--min-chunk-size", type=int, default=100, help="Remove objects smaller than this area (px)")
    ap.add_argument("--workers", type=int, default=0, help="# processes (0 = use all cores)")

    ap.add_argument("--watershed", action="store_true", help="Enable watershed splitting for touching cells")
    ap.add_argument("--min-distance", type=int, default=22, help="Min peak distance (px) for watershed seeds")

    ap.add_argument("--frame-zfill", type=int, default=3, help="Zero-fill used to locate imagesTs files (e.g., 3 → 001)")
    ap.add_argument("--fps", type=int, default=4, help="FPS for output video of montages")

    ap.add_argument("--csv-out", type=str, default="counts.csv", help="Filename for CSV summary (inside out-dir)")
    ap.add_argument("--excel-out", type=str, default="", help="Optional Excel filename (inside out-dir); if blank, skip")
    ap.add_argument("--video-out", type=str, default="montage.mp4", help="Filename for output video (inside out-dir)")
    ap.add_argument("--display-size", type=parse_size, help="Resize image/masks/labels for visualization to WIDTHxHEIGHT (W×H). Masks/labels use nearest-neighbor to preserve classes.")
    return ap.parse_args()


def ensure_out_dirs(base: Path) -> Tuple[Path, Path]:
    figs = base / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    return base, figs


def extract_frame_number_from_mask(mask_name: str) -> Optional[int]:
    """Extract trailing integer token from a mask filename.
    Examples:
      AA_001_0007.png → 7,  frame_0003.png → 3,  mymask_42.png → 42
    """
    base = Path(mask_name).stem
    m = re.search(r"([0-9]+)$", base)  # trailing digits
    if not m:
        return None
    return int(m.group(1))


def find_image_for_frame(image_folder: Path, frame_num: int, zfill: int) -> Optional[Path]:
    """Try several common imagesTs patterns to find the original image for a frame.
    Patterns tried (in order):
      * *_{frame:0z}_0000.png
      * *_{frame}.png (rare)
    """
    patt1 = f"*_{frame_num:0{zfill}d}_0000.png"
    patt2 = f"*_{frame_num}.png"
    cands = list(image_folder.glob(patt1))
    if not cands:
        cands = list(image_folder.glob(patt2))
    return sorted(cands)[0] if cands else None


def parse_size(size_str: str | None):
    """Parse 'WIDTHxHEIGHT' -> (W, H) or None."""
    if not size_str:
        return None
    m = re.match(r"^(\d+)x(\d+)$", size_str.strip())
    if not m:
        raise argparse.ArgumentTypeError("Size must be WIDTHxHEIGHT, e.g., 1920x1080")
    return int(m.group(1)), int(m.group(2))

def _resize_gray(img: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    W, H = wh
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

def _resize_bool(mask: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    W, H = wh
    arr = (mask.astype(np.uint8) * 255)
    out = cv2.resize(arr, (W, H), interpolation=cv2.INTER_NEAREST)
    return out > 127

def _resize_label(lbl: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    W, H = wh
    out = cv2.resize(lbl.astype(np.int32), (W, H), interpolation=cv2.INTER_NEAREST)
    return out.astype(lbl.dtype)


# ------------------------------------------------------------
# Segmentation helpers
# ------------------------------------------------------------

'''

def split_cells_watershed(binary_mask: np.ndarray, min_distance: int) -> np.ndarray:
    # distance transform
    dist = ndi.distance_transform_edt(binary_mask)
    # local maxima as markers
    coords = peak_local_max(
        dist,
        min_distance=min_distance,
        footprint=np.ones((min_distance, min_distance), bool),
        labels=binary_mask,
    )
    markers = np.zeros_like(dist, dtype=int)
    for idx, (r, c) in enumerate(coords, start=1):
        markers[r, c] = idx
    return sk_watershed(-dist, markers, mask=binary_mask)


def label_and_count(binary_mask: np.ndarray, use_ws: bool, min_distance: int) -> Tuple[np.ndarray, int]:
    if use_ws:
        lbl = split_cells_watershed(binary_mask, min_distance)
        count = int(lbl.max())
        return lbl, count
    else:
        lbl, count = ndi.label(binary_mask)
        return lbl, int(count)

'''



# ------------------------------------------------------------
# Segmentation helpers (PER-CLASS WS, NO MIXING)
# ------------------------------------------------------------

WS_PEAK_THRESH_REL = 0.35     # lower -> more seeds (more splitting); try 0.18–0.40
WS_DIST_SMOOTH_SIGMA = 1.5    # try 0.8–2.0


def split_cells_watershed(
    binary_mask: np.ndarray,
    min_distance: int,
    peak_thresh_rel: float = WS_PEAK_THRESH_REL,
    dist_smooth_sigma: float = WS_DIST_SMOOTH_SIGMA,
) -> Tuple[np.ndarray, int]:
    """
    Watershed split on ONE class binary mask.
    Returns:
      inst_lbl: int32 labels (0..N)
      N: number of instances
    """
    fg = (binary_mask > 0).astype(bool)
    if fg.sum() == 0:
        return np.zeros_like(binary_mask, dtype=np.int32), 0

    # Connected components for "at least one seed per CC"
    cc_lbl, cc_count = ndi.label(fg)
    if cc_count == 0:
        return np.zeros_like(binary_mask, dtype=np.int32), 0

    dist = ndi.distance_transform_edt(fg).astype(np.float32)
    if dist_smooth_sigma and dist_smooth_sigma > 0:
        dist = cv2.GaussianBlur(dist, (0, 0), float(dist_smooth_sigma))

    dmax = float(dist.max())
    if dmax <= 1e-6:
        # can't split; fall back to connected components
        return cc_lbl.astype(np.int32), int(cc_count)

    threshold_abs = float(peak_thresh_rel) * dmax
    coords = peak_local_max(
        dist,
        labels=fg.astype(np.uint8),
        min_distance=int(min_distance),
        threshold_abs=float(threshold_abs),
        exclude_border=False,
    )

    peak_mask = np.zeros_like(fg, dtype=bool)
    if coords.size > 0:
        peak_mask[tuple(coords.T)] = True

    # Ensure at least one marker per connected component
    has_marker = np.zeros(cc_count + 1, dtype=bool)
    if coords.size > 0:
        cc_ids_at_peaks = cc_lbl[coords[:, 0], coords[:, 1]]
        has_marker[cc_ids_at_peaks] = True

    for cc_id in range(1, cc_count + 1):
        if not has_marker[cc_id]:
            rr, cc = ndi.maximum_position(dist, labels=cc_lbl, index=cc_id)
            peak_mask[int(rr), int(cc)] = True

    markers, n_markers = ndi.label(peak_mask)
    if n_markers == 0:
        return cc_lbl.astype(np.int32), int(cc_count)

    inst_lbl = sk_watershed(-dist, markers, mask=fg)
    return inst_lbl.astype(np.int32), int(inst_lbl.max())


def split_per_class_instances(
    healthy_mask: np.ndarray,
    sickle_mask: np.ndarray,
    min_distance: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Watershed split separately per class (NO MIXING).

    Returns:
      healthy_ws(bool), sickle_ws(bool),
      lbl_h(int32), lbl_s(int32),
      count_h(int), count_s(int)
    """
    lbl_h, count_h = split_cells_watershed(healthy_mask, min_distance=min_distance)
    lbl_s, count_s = split_cells_watershed(sickle_mask,  min_distance=min_distance)

    healthy_ws = (lbl_h > 0)
    sickle_ws  = (lbl_s > 0)
    return healthy_ws, sickle_ws, lbl_h, lbl_s, int(count_h), int(count_s)


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------

def draw_montage(
    img_gray: np.ndarray,
    # "no watershed" inputs
    raw_healthy_mask: np.ndarray,
    raw_sickle_mask: np.ndarray,
    raw_lbl_h: np.ndarray,
    raw_lbl_s: np.ndarray,
    raw_count_h: int,
    raw_count_s: int,
    # "with watershed" inputs (used only if show_ws=True)
    healthy_mask: np.ndarray,
    sickle_mask: np.ndarray,
    lbl_h: np.ndarray,
    lbl_s: np.ndarray,
    count_h: int,
    count_s: int,
    frame_num: int,
    out_path: Path,
    show_ws: bool,
    display_size: Optional[tuple[int, int]] = None,
    split_line_thickness: int = 2,  # dilation iterations
) -> None:
    """
    If show_ws False: [Original | Overlay (NO WS)]
    If show_ws True : [Original | Overlay (NO WS) | Overlay (WS) + thick yellow split lines]
    Yellow lines are drawn ONLY for instances that were actually split by watershed
    WITHIN THE SAME CLASS (healthy-healthy or sickle-sickle). No class mixing.
    """

    
    def _draw_overlay(ax, img, hm, sm, lh, ls, ch, cs, title: str) -> None:
        ol_healthy = ListedColormap(["none", "green"])
        ol_sickle  = ListedColormap(["none", "red"])


        ax.imshow(img, cmap="gray")
        ax.imshow(hm, cmap=ol_healthy, alpha=0.6, interpolation="none")
        ax.imshow(sm, cmap=ol_sickle,  alpha=0.6, interpolation="none")

        h_cent = center_of_mass(hm, lh, range(1, ch + 1)) if ch > 0 else []
        s_cent = center_of_mass(sm, ls, range(1, cs + 1)) if cs > 0 else []

        for i, (r, c) in enumerate(h_cent, start=1):
            ax.text(c, r, str(i), color="yellow", ha="center", va="center", fontsize=6)
        for i, (r, c) in enumerate(s_cent, start=ch + 1):
            ax.text(c, r, str(i), color="cyan", ha="center", va="center", fontsize=6)

        ax.set_title(title, pad=12)
        ax.axis("off")

    def _instance_boundaries(inst_lbl: np.ndarray) -> np.ndarray:
        """Full outlines (instance-vs-background and instance-vs-instance), 4-neighborhood."""
        b = np.zeros_like(inst_lbl, dtype=bool)
        b[1:, :]  |= (inst_lbl[1:, :]  != inst_lbl[:-1, :])
        b[:-1, :] |= (inst_lbl[:-1, :] != inst_lbl[1:, :])
        b[:, 1:]  |= (inst_lbl[:, 1:]  != inst_lbl[:, :-1])
        b[:, :-1] |= (inst_lbl[:, :-1] != inst_lbl[:, 1:])
        b &= (inst_lbl > 0)
        return b

    # Resize only for visualization
    if display_size is not None:
        img_gray = _resize_gray(img_gray, display_size)

        raw_healthy_mask = _resize_bool(raw_healthy_mask, display_size)
        raw_sickle_mask  = _resize_bool(raw_sickle_mask,  display_size)
        raw_lbl_h        = _resize_label(raw_lbl_h, display_size)
        raw_lbl_s        = _resize_label(raw_lbl_s, display_size)

        if show_ws:
            healthy_mask = _resize_bool(healthy_mask, display_size)
            sickle_mask  = _resize_bool(sickle_mask,  display_size)
            lbl_h        = _resize_label(lbl_h, display_size)
            lbl_s        = _resize_label(lbl_s, display_size)

    ncols = 3 if show_ws else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))

    # Panel 0: Original
    axes[0].imshow(img_gray, cmap="gray")
    axes[0].set_title("Original", pad=12)
    axes[0].axis("off")

    # Panel 1: Overlay (NO WS)
    _draw_overlay(
        axes[1],
        img_gray,
        raw_healthy_mask,
        raw_sickle_mask,
        raw_lbl_h,
        raw_lbl_s,
        int(raw_count_h),
        int(raw_count_s),
        title=f"Overlay (NO WS)  H={int(raw_count_h)}, S={int(raw_count_s)}",
    )

    # Panel 2: Overlay (WS) + thick yellow split lines (PER-CLASS ONLY)
    if show_ws:
        _draw_overlay(
            axes[2],
            img_gray,
            healthy_mask,
            sickle_mask,
            lbl_h,
            lbl_s,
            int(count_h),
            int(count_s),
            title=f"Overlay (WS)  H={int(count_h)}, S={int(count_s)}",
        )

        # ------------------------------------------------------------
        # PER-CLASS split detection:
        # A raw healthy CC is "split" if it overlaps >=2 WS healthy IDs.
        # A raw sickle  CC is "split" if it overlaps >=2 WS sickle  IDs.
        # Then draw full outlines for those split WS instances only.
        # ------------------------------------------------------------
        split_h_ids = set()
        for cc_id in range(1, int(raw_count_h) + 1):
            region = (raw_lbl_h == cc_id)
            ids = np.unique(lbl_h[region])
            ids = ids[ids > 0]
            if ids.size >= 2:
                split_h_ids.update(ids.tolist())

        split_s_ids = set()
        for cc_id in range(1, int(raw_count_s) + 1):
            region = (raw_lbl_s == cc_id)
            ids = np.unique(lbl_s[region])
            ids = ids[ids > 0]
            if ids.size >= 2:
                split_s_ids.update(ids.tolist())

        b = np.zeros_like(img_gray, dtype=bool)

        if split_h_ids:
            split_h_mask = np.isin(lbl_h, np.fromiter(split_h_ids, dtype=np.int32))
            b |= (_instance_boundaries(lbl_h) & split_h_mask)

        if split_s_ids:
            split_s_mask = np.isin(lbl_s, np.fromiter(split_s_ids, dtype=np.int32))
            b |= (_instance_boundaries(lbl_s) & split_s_mask)

        if np.any(b):
            iters = max(1, int(split_line_thickness))
            b_u8 = (b.astype(np.uint8) * 255)
            kernel = np.ones((3, 3), np.uint8)
            b_thick = cv2.dilate(b_u8, kernel, iterations=iters) > 0

            ol_split = ListedColormap(["none", "yellow"])
            axes[2].imshow(b_thick, cmap=ol_split, alpha=1.0, interpolation="none")

    # Prevent title clipping
    fig.suptitle(f"Frame {frame_num:04d}", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# ------------------------------------------------------------
# Per-frame processing (now also draws montage INSIDE the worker)
# ------------------------------------------------------------


def process_one(
    mask_path: Path,
    image_folder: Path,
    zfill: int,
    min_chunk_size: int,
    use_ws: bool,
    min_distance: int,
    figs_dir: Path,
    display_size: Optional[tuple[int, int]],
) -> Dict[str, Any]:
    frame_num = extract_frame_number_from_mask(mask_path.name)
    if frame_num is None:
        raise ValueError(f"Could not parse frame number from mask filename: {mask_path.name}")

    img_path = find_image_for_frame(image_folder, frame_num, zfill)
    if img_path is None:
        raise FileNotFoundError(f"No matching image found for frame {frame_num} in {image_folder}")

    # Load
    mask = np.array(Image.open(mask_path))
    img  = np.array(Image.open(img_path).convert("L"))

    # Binary masks from predicted classes
    healthy = (mask == 1)
    sickle  = (mask == 2)

    # Clean small bits
    healthy = remove_small_objects(healthy, min_size=min_chunk_size)
    sickle  = remove_small_objects(sickle,  min_size=min_chunk_size)

    # ---- NO SPLIT (raw CC) counts (always computed)
    raw_h_lbl, raw_h_count = ndi.label(healthy)
    raw_s_lbl, raw_s_count = ndi.label(sickle)
    raw_total = int(raw_h_count + raw_s_count)
    raw_ratio = (raw_s_count / raw_total) if raw_total > 0 else math.nan

    # ---- SPLIT (watershed) counts (only if use_ws=True)
    if use_ws:

        healthy_ws, sickle_ws, lbl_h, lbl_s, count_h, count_s = split_per_class_instances(
            healthy_mask=healthy,
            sickle_mask=sickle,
            min_distance=min_distance,
        )
        
    else:
        # If watershed is off, "final" equals raw
        healthy_ws, sickle_ws = healthy, sickle
        lbl_h, count_h = raw_h_lbl, int(raw_h_count)
        lbl_s, count_s = raw_s_lbl, int(raw_s_count)

    total = int(count_h + count_s)
    ratio = (count_s / total) if total > 0 else math.nan

    # Draw montage
    fig_path = figs_dir / f"frame_{frame_num:04d}.png"
    draw_montage(
        img_gray=img,

        raw_healthy_mask=healthy,
        raw_sickle_mask=sickle,
        raw_lbl_h=raw_h_lbl,
        raw_lbl_s=raw_s_lbl,
        raw_count_h=int(raw_h_count),
        raw_count_s=int(raw_s_count),

        healthy_mask=healthy_ws,
        sickle_mask=sickle_ws,
        lbl_h=lbl_h,
        lbl_s=lbl_s,
        count_h=int(count_h),
        count_s=int(count_s),

        frame_num=frame_num,
        out_path=fig_path,
        show_ws=use_ws,
        display_size=display_size,
    )

    # Return BOTH (raw and ws) so main can write both when watershed is enabled
    return {
        "frame": frame_num,

        # no-split (raw CC)
        "healthy_raw": int(raw_h_count),
        "sickle_raw": int(raw_s_count),
        "total_raw": int(raw_total),
        "sickling_ratio_raw": float(raw_ratio) if not math.isnan(raw_ratio) else math.nan,

        # split (watershed) / final
        "healthy_ws": int(count_h),
        "sickle_ws": int(count_s),
        "total_ws": int(total),
        "sickling_ratio_ws": float(ratio) if not math.isnan(ratio) else math.nan,

        "fig_path": str(fig_path),
    }



# ------------------------------------------------------------
# Video writing
# ------------------------------------------------------------

def write_video_from_images(img_paths: List[Path], out_path: Path, fps: int) -> None:
    if not img_paths:
        print("[WARN] No images supplied for video.")
        return
    # Read first frame to init writer
    first = cv2.imread(str(img_paths[0]))
    if first is None:
        raise RuntimeError(f"Failed to read first frame: {img_paths[0]}")
    h, w, _ = first.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))
    for p in img_paths:
        frame = cv2.imread(str(p))
        if frame is None:
            print(f"[WARN] Skipping unreadable montage frame: {p}")
            continue
        vw.write(frame)
    vw.release()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> int:
    args = parse_args()

    out_dir, figs_dir = ensure_out_dirs(args.out_dir)

    # Gather mask files
    mask_files = [p for p in sorted(args.mask_folder.iterdir()) if p.suffix.lower() == ".png"]
    if not mask_files:
        raise SystemExit(f"No PNG masks found in {args.mask_folder}")

    print("Configuration")
    print("==============")
    print(f"Mask folder    : {args.mask_folder}")
    print(f"Image folder   : {args.image_folder}")
    print(f"Output dir     : {out_dir}")
    print(f"Figures dir    : {figs_dir}")
    print(f"Watershed      : {'ON' if args.watershed else 'OFF'}")
    if args.watershed:
        print(f"  min_distance : {args.min_distance}")
    print(f"Min chunk size : {args.min_chunk_size}")
    max_workers = (os.cpu_count() or 1) if args.workers == 0 else max(1, args.workers)
    print(f"Workers        : {max_workers}")
    print(f"Frame zfill    : {args.frame_zfill}")
    print(f"Video FPS      : {args.fps}")
    if args.display_size:
        print(f"Display size   : {args.display_size[0]}x{args.display_size[1]} (W×H)")
    else:
        print("Display size   : native (no resize)")

    # Try to read the shared size if user didn't pass --display-size
    if args.display_size:
        w, h = args.display_size
    else:
        loaded = load_size_json(args.image_folder)
        if loaded is not None:
            w, h = loaded
        else:
            w = h = None

    if w is not None:
        print(f"Display size   : {w}x{h} (W×H)")
    else:
        print("Display size   : native (no resize)")

    # Process (count + montage) in parallel
    results: List[Dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [
            ex.submit(
                process_one,
                p,
                args.image_folder,
                args.frame_zfill,
                args.min_chunk_size,
                args.watershed,
                args.min_distance,
                figs_dir,
                args.display_size,
            )
            for p in mask_files
        ]
        for i, fut in enumerate(cf.as_completed(futs), start=1):
            try:
                res = fut.result()
                results.append(res)
                print(f"Processed {int(res['frame']):04d}  (#{i}/{len(mask_files)})")
            except Exception as e:
                print(f"[ERROR] {e}")

    if not results:
        raise SystemExit("No frames processed successfully.")

    # Sort by frame
    results.sort(key=lambda d: d["frame"])
    montage_paths: List[Path] = [Path(r["fig_path"]) for r in results]

    # Build dataframe and (optionally) add time in seconds
    df_all = pd.DataFrame(results).sort_values("frame").reset_index(drop=True)

    # time in seconds (ONLY needed/used when watershed is enabled)
    if args.watershed:
        if args.fps is None or float(args.fps) <= 0:
            raise SystemExit("--fps must be > 0 to plot time (seconds). "
                             "Use fps as your sampling rate (#saved masks per second).")
        t_sec = np.arange(len(df_all), dtype=np.float32) / float(args.fps)
        df_all.insert(1, "time_sec", t_sec)

    csv_path = out_dir / args.csv_out

    if args.watershed:
        # CSV with BOTH raw + ws columns (+ time_sec)
        cols_csv = [
            "frame", "time_sec",
            "healthy_raw", "sickle_raw", "total_raw", "sickling_ratio_raw",
            "healthy_ws",  "sickle_ws",  "total_ws",  "sickling_ratio_ws",
        ]
        df_all[cols_csv].to_csv(csv_path, index=False)
        print(f"Saved CSV (raw + ws) → {csv_path}")

        # Excel: 2 sheets (only if user asked for Excel)
        if args.excel_out:
            xlsx_path = out_dir / args.excel_out

            df_no = df_all[["frame", "time_sec", "healthy_raw", "sickle_raw", "total_raw", "sickling_ratio_raw"]].copy()
            df_ws = df_all[["frame", "time_sec", "healthy_ws",  "sickle_ws",  "total_ws",  "sickling_ratio_ws"]].copy()

            df_no.columns = ["frame", "time_sec", "healthy", "sickle", "total", "sickling_ratio"]
            df_ws.columns = ["frame", "time_sec", "healthy", "sickle", "total", "sickling_ratio"]

            try:
                with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                    df_no.to_excel(writer, sheet_name="no_split", index=False)
                    df_ws.to_excel(writer, sheet_name="split", index=False)
                print(f"Saved Excel (2 sheets) → {xlsx_path}")
            except Exception as e:
                print(f"[WARN] Could not write Excel ({e}); CSV already saved.")

        # Sickling ratio plot with BOTH curves (x-axis = seconds)
        ratio_fig = out_dir / "sickling_ratio_compare.png"
        plt.figure(figsize=(10, 4))
        plt.plot(df_all["time_sec"].values, df_all["sickling_ratio_raw"].values, marker="o", label="No split (raw CC)")
        plt.plot(df_all["time_sec"].values, df_all["sickling_ratio_ws"].values,  marker="o", label="Watershed split")
        plt.xlabel("Time (s)")
        plt.ylabel("Sickling ratio (S / (H+S))")
        plt.title("Sickling Ratio vs Time: Before vs After Watershed")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(ratio_fig, dpi=150, bbox_inches="tight", pad_inches=0.15)
        plt.close()
        print(f"Saved sickling ratio comparison plot → {ratio_fig}")

    else:
        # No watershed: ONLY raw (single result) + no ratio plot
        cols_csv = ["frame", "healthy_raw", "sickle_raw", "total_raw", "sickling_ratio_raw"]
        df_one = df_all[cols_csv].copy()
        df_one.columns = ["frame", "healthy", "sickle", "total", "sickling_ratio"]

        df_one.to_csv(csv_path, index=False)
        print(f"Saved CSV (no split only) → {csv_path}")

        if args.excel_out:
            xlsx_path = out_dir / args.excel_out
            try:
                df_one.to_excel(xlsx_path, index=False)
                print(f"Saved Excel (single sheet) → {xlsx_path}")
            except Exception as e:
                print(f"[WARN] Could not write Excel ({e}); CSV already saved.")

    # Video from montages (unchanged)
    video_path = out_dir / args.video_out
    write_video_from_images(montage_paths, video_path, args.fps)
    print(f"Saved video → {video_path}")

    print("Done.")
    print(f"Frames processed : {len(df_all)}")
    print(f"Figures dir      : {figs_dir}")
    print(f"Table CSV        : {csv_path}")
    if args.excel_out:
        print(f"Table Excel      : {out_dir / args.excel_out}")
    if args.watershed:
        print(f"Ratio plot       : {out_dir / 'sickling_ratio_compare.png'}")
    print(f"Montage video    : {video_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

