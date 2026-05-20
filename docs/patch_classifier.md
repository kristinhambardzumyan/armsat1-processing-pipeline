# Patch Dataset and Classification

This stage builds RGBN image tiles from georeferenced ArmSat-1 scenes and classifies each tile as:

```text
clear
cloud
shadow
snow
```

The classifier operates on 4-band input consisting of:

```text
R
G
B
N
```

---

# Patch Dataset Construction

Patch dataset construction is implemented in:

```text
src/tiling/build_patch_dataset.py
```

The tiling pipeline:

1. Reads georeferenced ArmSat-1 bands.
2. Stacks `R`, `G`, `B`, and `N` into a 4-band RGBN image.
3. Splits each scene into `512 × 512` tiles.
4. Filters invalid tiles using a valid-pixel threshold.
5. Writes tile quality statistics and reports.

For 2024 scenes, the pipeline additionally creates a float32 representation using the original scale and offset values:

```text
float32 = raw_uint16 * scale + offset
```

Masked and NoData pixels are written as:

```text
-9999
```

The float32 representation is used only for classification consistency. The original native-format UInt16 georeferenced outputs remain preserved separately.

---

# Tiling Code Structure

```text
src/tiling/
└── build_patch_dataset.py
```

The tiling pipeline uses:

- GDAL for RGBN stacking and tiling
- Rasterio and NumPy for filtering and float32 conversion
- CSV reports for tile statistics

---

# Scene Lists

Patch dataset construction operates on scene-list files.

Each file contains one scene name per line:

```text
20230307T102907_20230307T102909_ARMSAT1_...
20230308T103120_20230308T103122_ARMSAT1_...
```

Example:

```text
patch_classifier/scene_lists/Armsat-1-2023_all.txt
patch_classifier/scene_lists/Armsat-1-2024_all.txt
```

The script processes only the scenes listed in the provided file.

---

# Example: Build Patch Dataset

For 2023:

```bash
python src/tiling/build_patch_dataset.py \
  patch_classifier/scene_lists/Armsat-1-2023_all.txt
```

For 2024:

```bash
python src/tiling/build_patch_dataset.py \
  patch_classifier/scene_lists/Armsat-1-2024_all.txt
```

Default tiling settings:

```text
tile_size = 512
overlap = 0
min_valid_ratio = 0.99
nodata_f = -9999
```

---

# Patch Classifier

Patch classification is implemented in:

```text
src/patch_classifier/
├── model.py
├── infer_tiles.py
└── make_previews_from_predictions.py
```

| File | Purpose |
|---|---|
| `model.py` | Defines the 4-channel ResNet-50 classifier |
| `infer_tiles.py` | Runs inference on RGBN tiles |
| `make_previews_from_predictions.py` | Creates RGB preview images from predictions |

The classifier is based on ResNet-50 modified to accept 4 input channels instead of 3.

The inference pipeline:

1. Loads RGBN tiles.
2. Fills masked or invalid values.
3. Normalizes the input using fixed training statistics.
4. Runs classification inference.
5. Exports prediction CSV files and summaries.

---

# Model Checkpoint

Download the trained classifier checkpoint from:

[ArmSat-1 Model Checkpoints](https://drive.google.com/drive/folders/1WcZPbszdhc50F6bWuxjOiy8NvIt8Xo5M?usp=drive_link)

After downloading, place the checkpoint here:

```text
patch_classifier/models/resnet50_4ch_classifier.pth
```

Expected inference path:

```bash
--model_path patch_classifier/models/resnet50_4ch_classifier.pth
```

---

# Example: Run Classification

```bash
python -m patch_classifier.infer_tiles \
  --tiles_root outputs/patch_classifier/datasets \
  --datasets Armsat-1-2023 Armsat-1-2024-float32 \
  --model_path patch_classifier/models/resnet50_4ch_classifier.pth \
  --out_dir outputs/patch_classifier/predictions/full_dataset \
  --batch_size 32 \
  --num_workers 4
```

Before running from the repository root:

```bash
export PYTHONPATH=$PWD/src
```

---

# Prediction Outputs

The classifier exports prediction CSV files and summary statistics:

```text
outputs/patch_classifier/predictions/full_dataset/
├── predictions.csv
├── prediction_summary.csv
├── prediction_summary_by_dataset.csv
└── prediction_summary_by_scene.csv
```

The prediction CSV contains:

- tile path
- dataset name
- scene name
- predicted class
- class probabilities

The prediction outputs can later be used for:

- preview generation
- dataset filtering
- quality analysis
- exporting tiles by predicted class

---

# Preview Generation

RGB previews can be generated from prediction CSV files:

```bash
python -m patch_classifier.make_previews_from_predictions \
  --predictions_csv outputs/patch_classifier/predictions/full_dataset/predictions.csv \
  --out_dir outputs/patch_classifier/previews \
  --max_per_class 100
```

The preview pipeline generates stretched RGB JPEG previews grouped by predicted class.

---

# Important Note on 2024 Data

The classifier operates on:

```text
Armsat-1-2024-float32
```

because the original 2024 scenes are stored in native UInt16 format with scale and offset metadata.

The float32 conversion is used only during classification and preprocessing for the classifier pipeline.

The original georeferenced UInt16 outputs remain preserved separately and are not replaced.