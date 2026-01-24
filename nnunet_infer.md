## `nnunet_infer.py`

Run **nnU-Net v2** inference on your extracted frames in **`imagesTs`** format and write predicted masks to disk. This script is the middle step of the pipeline:

1) `extract_frames.py` → creates `Task{ID}_{NAME}/imagesTs/…`
2) **`nnunet_infer.py`** → runs `nnUNetv2_predict` on `imagesTs/`
3) `sicklecell_count_and_visualize.py` → counts cells, makes plots & a montage video

---

### Usage

#### CUDA (single GPU)
```bash
python nnunet_infer.py \
  --imagesTs Frames_for_inference/Task101_Experiment1/imagesTs/ \
  --out-dir nnunet_masks_out \
  --nn_trained_dataset-id 102 \
  --device cuda --gpus 0 \
  --npp 3 --nps 3 \
  --enable-progress-bar
```
#### CUDA (multi-GPU)
```bash
python nnunet_infer.py \
  --imagesTs Frames_for_inference/Task101_Experiment1/imagesTs/ \
  --out-dir nnunet_masks_out \
  --nn_trained_dataset-id 102 \
  --device cuda --gpus 0,1,2 \
  --parts 3 \
  --npp 3 --nps 3
```


There is option to check what commands are going to be executed with the flag ```-- dry run ```.

### Command-line arguments
#### Required I/O

- ```--imagesTs``` PATH (default: Frames_for_inference/Task101_Experiment1/imagesTs/)
Path to the nnU-Net imagesTs directory containing input PNGs.

- ```--out-dir``` PATH (default: nnunet_masks_out)
Output directory for predictions (per-frame masks).

#### Model & runtime

- ```--nn_trained_dataset-id``` INT (default: 102)
nnU-Net dataset ID used during training (e.g., 102).
Note: This is not the same as the Task ID you used in extract_frames.py. For the pretrained weights in this repo, use 102.

- ```--checkpoint``` STR (default: checkpoint_best.pth)
Checkpoint name or path from nnU-Net training.This is only required if you train the nnunet on your dataset. For pretrained weights, default option is best!

Worker counts (per process)

- ```--npp``` INT (default: 3)
Number of preprocessing workers (nnU-Net). For small devices like on your personal laptop, use 1.

- ```--nps``` INT (default: 3)
Number of segmentation/export workers (nnU-Net). For small devices like on your personal laptop, use 1.

Parallelism & device

- ```--parts``` INT (default: 0)
Override number of parallel parts/processes. 0 = auto (script picks based on device/GPUs).

- ```--device``` {auto,cuda,cpu,mps} (default: auto)
Compute device for nnUNetv2_predict. Use cuda for NVIDIA GPUs, mps for Apple Silicon.

- ```--gpus LIST``` (default: None)
Comma-separated GPU IDs when using CUDA (e.g., 0,1,2). Ignored for cpu/mps.


#### Logging / UX

- ```--dry-run```
Print the composed nnUNetv2_predict commands without executing.

- ```--enable-progress-bar```
Show tqdm progress bars from nnU-Net (off by default).


