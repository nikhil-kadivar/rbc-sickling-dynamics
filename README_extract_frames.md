## `extract_frames.py`

Extract frames from **one video** or **a directory of videos** into **nnU-Net `imagesTs`** format. Choose a sampling policy (every _N_ seconds, every _N_ frames, or all frames), set a filename prefix, and optionally resize frames to your model’s training resolution.


1) **`extract_frames.py`** → creates `Task{ID}_{NAME}/imagesTs/…`
2) `nnunet_infer.py`→ runs `nnUNetv2_predict` on `imagesTs/`
3) `sicklecell_count_and_visualize.py` → counts cells, makes plots & a montage video

---


### Usage
#### Single Video
```bash
python extract_frames.py \
  --video /path/to/video.mp4 \
  --out-base-dir ./runs \
  --task-id 101 \
  --task-name Experiment1 \
  --prefix Flow \
  --every-sec 10 \
  --target-size 1080x1620 \
  --workers 0Output and naming```

#### Folder of videos
```bash
python extract_frames.py \
  --input-dir /path/to/videos_dir \
  --out-base-dir ./runs \
  --task-id 101 \
  --task-name Experiment1 \
  --prefix Flow \
  --every-n-frames 30 \
  --target-size 1080x1620 \
  --workers 8
```
***Tip:*** `--workers 0` uses all CPU cores for faster saving.

### Command-line arguments

#### Source (mutually exclusive, one required)

- `--input-dir PATH` :
  Directory containing video files.

- `--video PATH` :
   Path to a single video file, this is just to process single video. If you prefer ots of videos to be processed together, please use the input -dir command.

#### Output & naming

- `--out-base-dir PATH` (default: `Frames_for_inference`) :
Base output directory where extracted and renamed frames will be saved.

- `--task-id INT` (default: `101`) :
nnU-Net Task ID (e.g., 101).

- `--task-name STR` (default: `Experiment1`) :
nnU-Net Task Name (e.g., experiment1).

- `--prefix STR` (default: `Flow`) : 
Prefix for saved frames.

### Sampling policy (mutually exclusive, choose one)

- `--every-sec FLOAT` : 
Save one frame every N seconds (timestamp-based; robust to variable FPS).

- `--every-n-frames INT` : 
Save one frame every N frames (index-based).

- `--all-frames` : 
Save all frames.

### Resize & performance

- `--target-size WIDTHxHEIGHT` (default: `1080x1620`) : 
Resize to `WIDTHxHEIGHT` (PIL expects width,height).
The number `1080x1620` is default and we arrived at this since we trained our nnunet model on this width and height. We observed doing inference on the same size at which nnunet is trained gives better accuracy. We recommend using this width and height if you are planning to use the pre-trained nnunet weights from this github repo. If you plan to train from scratch on your own experiment videos or frames. Please adjust this accordingly according to the training dataset.

- `--workers INT` (default: `0`) : 
Number of processes for saving (`0` = use all cores).

### Output structure

Frames are written to a nnU-Net test directory:

```bash
<out-base-dir>/
└─ Task{task-id}_{task-name}/
   └─ imagesTs/
      ├─ Flow_000_0000.png
      ├─ Flow_001_0000.png
      └─ ...

```

- Filenames are zero-padded and end with `_0000.png` for nnU-Net compatibility.

### Notes & tips

- Pick a sampling policy that matches your study design.
Timestamp-based (`--every-sec`) is robust with variable frame rates; frame-based (`--every-n-frames`) is fastest with constant FPS.

- For pretrained weights here, `--target-size 1080x1620` is recommended.

- Provide exactly one source (`--video` or `--input-dir`) and one sampling option.

- Increasing `--workers` speeds up PNG writing on fast disks.
