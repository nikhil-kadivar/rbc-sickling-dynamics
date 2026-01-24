# Environment setup

To setup environment, follow this command in your terminal.

### 0. Clone the repository

```bash
git clone https://github.com/nikhil-kadivar/rbc-sickling-dynamics.git
cd rbc-sickling-dynamics
```

### 1. Create Conda environment

```bash

conda create -n sickleflow_gpu python=3.10 -y
conda activate sickleflow_gpu
```

### 2. Install Pytorch
```bash
pip install "torch==2.2.2" --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install all required packages

```bash
pip install -r requirements-gpu.txt
```

### 4. Set the environment paths
```bash
# From the repository root
source scripts/set_paths.sh
```

### 5. Download pretrained model
```bash
python scripts/download_checkpoint.py
```

You are now ready to run the full pipeline.

## Usage

In default mode, to simply run this tool with the pretrained weights, run these three files as shown below.

1) `extract_frames.py` → creates `Task{ID}_{NAME}/imagesTs/…`
2) **`nnunet_infer.py`** → runs `nnUNetv2_predict` on `imagesTs/`
3) `count_and_visualize.py` → counts cells, makes plots & a montage video

---

After creating python environment as suggested in installation step, use command line argument to run with default settings as shown below:

### Step 1
```bash
python extract_frames.py --video /path/to/video.mp4
```
More info on extract_frames.py can be find here: [extract_frames.md](extract_frames.md)


### Step 2
```bash
python nnunet_infer.py
```
More info on extract_frames.py can be find here: [nnunet_infer.md](nnunet_infer.md)

### Step 3
```bash
python count_and_visualize.py --watershed
```
More info on extract_frames.py can be find here: [count_and_visualize.md](count_and_visualize.md)

### Note: 
#### Set required paths (run once per fresh session)

This project relies on environment variables. After starting a fresh session (e.g., a new login, restarted shell, or new compute job), you may need to run:

```bash
# From the repository root
source scripts/set_paths.sh
```

#### Customize weights
If you choose to retrain nnU-Net on your own data, then place the weights in the path "nnunet_data/nnUNet_results/". You can continue using fully automated pipeline now as it is with the new weights!
