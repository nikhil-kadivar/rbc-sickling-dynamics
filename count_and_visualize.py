#!/usr/bin/env python3
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
    ap.add_argument("--image-folder", type=Path, default = "Frames_for_inference", help="imagesTs folder with original frames")
    ap.add_argument("--out-dir", type=Path, default = "Outputs", help="Output directory (figures, tables, video)")

    ap.add_argument("--min-chunk-size", type=int, default=1000, help="Remove objects smaller than this area (px)")
    ap.add_argument("--workers", type=int, default=0, help="# processes (0 = use all cores)")

    ap.add_argument("--watershed", action="store_true", help="Enable watershed splitting for touching cells")
    ap.add_argument("--min-distance", type=int, default=22, help="Min peak distance (px) for watershed seeds")

    ap.add_argument("--frame-zfill", type=int, default=3, help="Zero-fill used to locate imagesTs files (e.g., 3 → 001)")
    ap.add_argument("--fps", type=int, default=5, help="FPS for output video of montages")

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


# ------------------------------------------------------------
# Visualization
# ------------------------------------------------------------


def draw_montage(
    img_gray: np.ndarray,
    healthy_mask: np.ndarray,
    sickle_mask: np.ndarray,
    lbl_h: np.ndarray,
    lbl_s: np.ndarray,
    count_h: int,
    count_s: int,
    frame_num: int,
    out_path: Path,
    include_ws_panel: bool,
    raw_lbl_h: Optional[np.ndarray] = None,
    raw_lbl_s: Optional[np.ndarray] = None,
    raw_count_h: Optional[int] = None,
    raw_count_s: Optional[int] = None,
    display_size: Optional[tuple[int, int]] = None,   # <— NEW
) -> None:

    """Create and save a per-frame figure.

    Panels:
      - No watershed: [Input | Combined labels | Final overlay]
      - With watershed: [Input | Raw CC | Final overlay (WS) | WS combined]
    """

    # Optional resample for visualization (does NOT affect counting)
    if display_size is not None:
        img_gray = _resize_gray(img_gray, display_size)
        healthy_mask = _resize_bool(healthy_mask, display_size)
        sickle_mask  = _resize_bool(sickle_mask,  display_size)
        lbl_h = _resize_label(lbl_h, display_size)
        lbl_s = _resize_label(lbl_s, display_size)
        if raw_lbl_h is not None and raw_lbl_s is not None:
            raw_lbl_h = _resize_label(raw_lbl_h, display_size)
            raw_lbl_s = _resize_label(raw_lbl_s, display_size)


    if include_ws_panel:
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    cmap_jet = plt.cm.jet
    ol_healthy = ListedColormap(['none', 'green'])
    ol_sickle  = ListedColormap(['none', 'red'])

    # Build combined maps for display
    # Raw combined if provided
    if raw_lbl_h is not None and raw_lbl_s is not None:
        raw_combined = np.zeros_like(img_gray, dtype=int)
        raw_combined[healthy_mask] = raw_lbl_h[healthy_mask]
        raw_combined[sickle_mask]  = raw_lbl_s[sickle_mask] + (raw_count_h or 0)
    else:
        raw_combined = None

    # Final (possibly watershed) combined
    final_combined = np.zeros_like(img_gray, dtype=int)
    final_combined[healthy_mask] = lbl_h[healthy_mask]
    final_combined[sickle_mask]  = lbl_s[sickle_mask] + count_h

    # Centroids for text labels (final labels)
    fin_h_cent = center_of_mass(healthy_mask, lbl_h, range(1, count_h + 1)) if count_h > 0 else []
    fin_s_cent = center_of_mass(sickle_mask,  lbl_s, range(1, count_s + 1)) if count_s > 0 else []

    # Panel 0: Original
    axes[0].imshow(img_gray, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Panel 1: If watershed → Raw CC; else → Combined labels (final)
    if include_ws_panel:
        axes[1].imshow(img_gray, cmap='gray')
        axes[1].imshow(raw_combined, cmap=cmap_jet, alpha=0.6, interpolation='none')
        axes[1].set_title('Raw CC Total=' + str(int((raw_count_h or 0) + (raw_count_s or 0))))
        axes[1].axis('off')
    else:
        axes[1].imshow(img_gray, cmap='gray')
        axes[1].imshow(final_combined, cmap=cmap_jet, alpha=0.6, interpolation='none')
        axes[1].set_title('Labels Total=' + str(count_h + count_s))
        axes[1].axis('off')

    # Panel 2: Final overlay with counts
    axes[2].imshow(img_gray, cmap='gray')
    axes[2].imshow(healthy_mask, cmap=ol_healthy, alpha=0.6, interpolation='none')
    axes[2].imshow(sickle_mask,  cmap=ol_sickle,  alpha=0.6, interpolation='none')
    # labels overlaid (final)
    for i, (r, c) in enumerate(fin_h_cent, start=1):
        axes[2].text(c, r, str(i), color='yellow', ha='center', va='center', fontsize=6)
    for i, (r, c) in enumerate(fin_s_cent, start=count_h + 1):
        axes[2].text(c, r, str(i), color='cyan',   ha='center', va='center', fontsize=6)
    axes[2].set_title(f'Final Overlay  H={count_h}, S={count_s}')
    axes[2].axis('off')

    # Panel 3 (only if watershed): WS combined
    if include_ws_panel:
        axes[3].imshow(img_gray, cmap='gray')
        axes[3].imshow(final_combined, cmap=cmap_jet, alpha=0.6, interpolation='none')
        axes[3].set_title('Watershed Combine Total=' + str(count_h + count_s))
        axes[3].axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
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
    display_size: Optional[tuple[int, int]],  # <— NEW
) -> Dict[str, Any]:
    frame_num = extract_frame_number_from_mask(mask_path.name)
    if frame_num is None:
        raise ValueError(f"Could not parse frame number from mask filename: {mask_path.name}")

    img_path = find_image_for_frame(image_folder, frame_num, zfill)
    if img_path is None:
        raise FileNotFoundError(f"No matching image found for frame {frame_num} in {image_folder}")

    # Load
    mask = np.array(Image.open(mask_path))
    img  = np.array(Image.open(img_path).convert('L'))

    # Binary masks
    healthy = (mask == 1)
    sickle  = (mask == 2)

    # Clean small bits
    healthy = remove_small_objects(healthy, min_size=min_chunk_size)
    sickle  = remove_small_objects(sickle,  min_size=min_chunk_size)

    # Raw counts
    raw_h_lbl, raw_h_count = ndi.label(healthy)
    raw_s_lbl, raw_s_count = ndi.label(sickle)

    # Final labeling (may be raw = no watershed)
    if use_ws:
        lbl_h, count_h = label_and_count(healthy, True,  min_distance)
        lbl_s, count_s = label_and_count(sickle,  True,  min_distance)
    else:
        lbl_h, count_h = raw_h_lbl, int(raw_h_count)
        lbl_s, count_s = raw_s_lbl, int(raw_s_count)

    total = count_h + count_s
    ratio = (count_s / total) if total > 0 else math.nan

    # Draw montage here (parallelized)
    fig_path = figs_dir / f"frame_{frame_num:04d}.png"
    draw_montage(
        img_gray=img,
        healthy_mask=healthy,
        sickle_mask=sickle,
        lbl_h=lbl_h,
        lbl_s=lbl_s,
        count_h=count_h,
        count_s=count_s,
        frame_num=frame_num,
        out_path=fig_path,
        include_ws_panel=use_ws,
        raw_lbl_h=raw_h_lbl,
        raw_lbl_s=raw_s_lbl,
        raw_count_h=int(raw_h_count),
        raw_count_s=int(raw_s_count),
        display_size=display_size
    )

    return {
        'frame': frame_num,
        'healthy': int(count_h),
        'sickle': int(count_s),
        'total': int(total),
        'sickling_ratio': ratio,
        'fig_path': str(fig_path),
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
    mask_files = [p for p in sorted(args.mask_folder.iterdir()) if p.suffix.lower() == '.png']
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
    # workers: 0 → all cores
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
        futs = [ex.submit(
            process_one,
            p,
            args.image_folder,
            args.frame_zfill,
            args.min_chunk_size,
            args.watershed,
            args.min_distance,
            figs_dir,
            args.display_size,
        ) for p in mask_files]
        for i, fut in enumerate(cf.as_completed(futs), start=1):
            try:
                res = fut.result()
                results.append(res)
                print(f"Processed {int(res['frame']):04d}  (#{i}/{len(mask_files)})")
            except Exception as e:
                print(f"[ERROR] {e}")

    if not results:
        raise SystemExit("No frames processed successfully.")

    # Sort by frame and collect montage paths
    results.sort(key=lambda d: d['frame'])
    montage_paths: List[Path] = [Path(r['fig_path']) for r in results]

    # Save table(s)
    df = pd.DataFrame(results)[['frame','healthy','sickle','total','sickling_ratio']]
    csv_path = out_dir / args.csv_out
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV → {csv_path}")

    if args.excel_out:
        xlsx_path = out_dir / args.excel_out
        try:
            df.to_excel(xlsx_path, index=False)
            print(f"Saved Excel → {xlsx_path}")
        except Exception as e:
            print(f"[WARN] Could not write Excel ({e}); CSV already saved.")

    # Sickling ratio plot
    ratio_fig = out_dir / "sickling_ratio.png"
    plt.figure(figsize=(10, 4))
    plt.plot(df['frame'].values, df['sickling_ratio'].values, marker='o')
    plt.xlabel('Frame')
    plt.ylabel('Sickling ratio (S / (H+S))')
    plt.title('Sickling Ratio per Frame')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(ratio_fig, dpi=150)
    plt.close()
    print(f"Saved sickling ratio plot → {ratio_fig}")

    # Video from montages
    video_path = out_dir / args.video_out
    write_video_from_images(montage_paths, video_path, args.fps)
    print(f"Saved video → {video_path}")

    # Final summary
    print("Done.")
    print(f"Frames processed : {len(results)}")
    print(f"Figures dir      : {figs_dir}")
    print(f"Table CSV        : {csv_path}")
    if args.excel_out:
        print(f"Table Excel      : {out_dir / args.excel_out}")
    print(f"Ratio plot       : {ratio_fig}")
    print(f"Montage video    : {video_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

