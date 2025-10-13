# `count_and_visualize.py`

Count **healthy** and **sickle** cells per frame from nnU-Net masks, (optionally) split touching cells with **watershed**, and generate:
- Per-frame **montage images** (parallelized)
- A **CSV** (and optional **Excel**) table of counts
- A **sickling-ratio** plot (`sickle / (healthy + sickle)`)
- A stitched **montage video**

This is Step **3** of the pipeline:

1) `extract_frames.py` → `Task{ID}_{NAME}/imagesTs/*.png`  
2) `nnunet_infer.py` → `nnunet_masks_out/*.png` (0=bg,1=healthy,2=sickle)  
3) **`sicklecell_count_and_visualize.py`** → counts, plots, video  

---

## Usage

### Basic (no watershed)
```bash
python sicklecell_count_and_visualize.py \
  --mask-folder nnunet_masks_out \
  --image-folder Frames_for_inference/Task101_Experiment1/imagesTs \
  --out-dir Outputs \
  --min-chunk-size 1000 \
  --workers 8 \
  --fps 6
```

### With watershed splitting
```bash
python sicklecell_count_and_visualize.py \
  --mask-folder nnunet_masks_out \
  --image-folder Frames_for_inference/Task101_Experiment1/imagesTs \
  --out-dir Outputs_ws \
  --watershed
```

## Command-line arguments

- ```--mask-folder``` PATH (default: nnunet_masks_out)
Folder with mask PNGs (labels: 0=background, 1=healthy, 2=sickle).

- ```--image-folder``` PATH (default: Frames_for_inference)
The imagesTs folder containing the original frame PNGs used for inference.
Filenames are expected to match the mask frame index, e.g. *_001_0000.png.

- ```--out-dir``` PATH (default: Outputs)
Output directory for figures, tables, and video.

- ```--min-chunk-size``` INT (default: 1000)
Remove connected components smaller than this area (in pixels) before counting.

- ```--workers``` INT (default: 0)
Number of processes used for parallel counting and plotting (0 = use all cores).

- ```--watershed``` (flag)
Enable watershed splitting to separate touching cells before counting.

- ```--min-distance``` INT (default: 22)
Peak spacing in pixels for watershed seed detection (skimage.peak_local_max).

- ```--fps``` INT (default: 5)
Frames-per-second for the stitched montage video.

- ```--csv-out``` STR (default: counts.csv)
Filename for the CSV summary (written inside --out-dir).

- ```--excel-out``` STR (default: empty)
Optional Excel filename (inside --out-dir). If empty, Excel is skipped.

- ```--video-out``` STR (default: montage.mp4)
Filename for the montage video (inside --out-dir).

- ```--display-size``` WIDTHxHEIGHT (optional)
Visualization-only resize for the montage frames (image + masks + labels).
Use this to “un-stretch” visuals if frames were resized at extraction.
Masks/labels use nearest-neighbor to preserve class integers.


