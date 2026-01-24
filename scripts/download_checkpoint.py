#!/usr/bin/env python

import os
from pathlib import Path
import gdown
import zipfile

# ------------------------------------------------------------
# 1) Get nnUNet_results from environment (set by set_paths.sh)
# ------------------------------------------------------------
nnunet_results = os.environ.get("nnUNet_results")
if nnunet_results is None:
    raise RuntimeError(
        "nnUNet_results is not set.\n"
        "Please run `source scripts/set_paths.sh` in the repo root before "
        "running download_checkpoint.py."
    )

nnunet_results = Path(nnunet_results)
nnunet_results.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2) Google Drive FILE ID for Dataset101_RBC.zip
#    (This is your ZIP: Dataset101_RBC.zip, ~3.21 GB)
# ------------------------------------------------------------
GDRIVE_FILE_ID = "1Xss5CII1Ei8dc1pilkIx48th1ESMOsR-"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

zip_path = nnunet_results / "Dataset101_RBC.zip"

print(f"[INFO] nnUNet_results = {nnunet_results}")
print(f"[INFO] Will download ZIP to:\n  {zip_path}")
print(f"[INFO] Downloading from:\n  {GDRIVE_URL}\n")

# ------------------------------------------------------------
# 3) Download the ZIP
# ------------------------------------------------------------
zip_path = Path(gdown.download(GDRIVE_URL, str(zip_path), quiet=False, fuzzy=True))

if (not zip_path.exists()) or zip_path.stat().st_size < 100_000_000:
    # 100 MB sanity check – your file is ~3.2 GB, so this should be much larger
    raise RuntimeError(
        f"Download looks too small: {zip_path} has size {zip_path.stat().st_size} bytes.\n"
        "Make sure the Google Drive file is the actual Dataset101_RBC.zip."
    )

print(f"[INFO] Download complete. Extracting {zip_path.name} ...")

# ------------------------------------------------------------
# 4) Extract ZIP into nnUNet_results
# ------------------------------------------------------------
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(nnunet_results)

print("[INFO] Extraction done.")

# Optional: remove ZIP after extraction
# zip_path.unlink()

print("\n[INFO] nnUNet results are now available under:")
for p in sorted((nnunet_results / "Dataset101_RBC").glob("**/*")):
    rel = p.relative_to(nnunet_results)
    tag = "[D]" if p.is_dir() else "[F]"
    print(f"  {tag} {rel}")

