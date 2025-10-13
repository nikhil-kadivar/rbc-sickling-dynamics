## Installation


## Usage

In default mode. To simply run this tool, you need to run these three files as shown below.

1) `extract_frames.py` → creates `Task{ID}_{NAME}/imagesTs/…`
2) **`nnunet_infer.py`** → runs `nnUNetv2_predict` on `imagesTs/`
3) `sicklecell_count_and_visualize.py` → counts cells, makes plots & a montage video

---

After creating python environment as suggested in installation step. You can use command line argument to run with default settings is shown below:



### Step 1
```bash
python extract_frames.py \
  --video /path/to/video.mp4 \
```
more info on extract_frames.py can be find here: [extract_frames.md](extract_frames.md)


### Step 2
```bash
python nnunet_infer.py
```
more info on extract_frames.py can be find here: [nnunet_infer.md](nnunet_infer.md)

### Step 3
```bash
python count_and_visualize.py
```
more info on extract_frames.py can be find here: [count_and_visualize.md](count_and_visualize.md)
