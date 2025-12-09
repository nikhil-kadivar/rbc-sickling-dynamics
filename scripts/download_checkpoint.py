#!/usr/bin/env python

import os
from pathlib import Path
import gdown

# 1) Get nnUNet_results from environment (set by scripts/set_paths.sh)
nnunet_results = os.environ.get("nnUNet_results")
if nnunet_results is None:
    raise RuntimeError(
        "nnUNet_results is not set.\n"
        "Please run `bash scripts/set_paths.sh` in the repo root before "
        "running download_checkpoint.py."
    )

nnunet_results = Path(nnunet_results)

# 2) Where we will store the checkpoint (standardized for this project)
#    >> If your own nnUNet folder uses a different Dataset name, change ONLY "Dataset102_SickleCells"
task_dir = nnunet_results / "Dataset102_RBC" / "nnUNetTrainer__nnUNetPlans__2d"
task_dir.mkdir(parents=True, exist_ok=True)

out_path = task_dir / "checkpoint_best.pth"

# 3) Google Drive share link
#    >>> THIS IS THE ONLY LINE YOU MUST EDIT <<<
GDRIVE_URL = "https://drive.google.com/drive/folders/1FrAzmIM1O82hFJ3d8WIn1UDlTzXGhxW6"

if "PASTE_YOUR_GOOGLE_DRIVE_SHARE_LINK_HERE" in GDRIVE_URL:
    raise RuntimeError(
        "Please edit scripts/download_checkpoint.py and set GDRIVE_URL "
        "to your actual Google Drive share link for checkpoint_best.pth."
    )

print(f"[INFO] nnUNet_results = {nnunet_results}")
print(f"[INFO] Will save checkpoint to:\n  {out_path}")
print(f"[INFO] Downloading from:\n  {GDRIVE_URL}\n")

# gdown can handle the full share URL directly
gdown.download(GDRIVE_URL, str(out_path), quiet=False)

print("\n[INFO] Done.")
print(f"[INFO] Checkpoint available at: {out_path}")

