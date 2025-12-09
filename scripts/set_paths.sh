#!/usr/bin/env bash
# Set default nnU-Net v2 paths relative to the repo root

export nnUNet_raw="${nnUNet_raw:-$(pwd)/nnunet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$(pwd)/nnunet_data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$(pwd)/nnunet_data/nnUNet_results}"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

echo "nnUNet_raw         = $nnUNet_raw"
echo "nnUNet_preprocessed= $nnUNet_preprocessed"
echo "nnUNet_results     = $nnUNet_results"

