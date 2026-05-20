# Matching

The matching stage estimates correspondences between preprocessed ArmSat-1 imagery and Sentinel-2 reference imagery.

The matching pipeline combines dense feature matching with geometric refinement in order to produce reliable control points for georeferencing.

The final method used in this project consists of:

```text
Initial RoMa Feature Matching
    ↓
Global Homography-Based Filtering
    ↓
Tile-Level Local Refinement
    ↓
Match Merging and Deduplication
    ↓
Final TPS-Based Geometric Filtering
```

The final filtered correspondences are later used for TPS-based image warping during georeferencing.

---

# Inputs

The matching stage expects preprocessing outputs for each scene:

```text
<scene_name>/
├── armsat_rgb_utm.tif
└── sentinel2_rgb_utm.tif
```

The matching code automatically locates these files inside each scene directory.

---

# Outputs

The matching stage writes results to:

```text
outputs/matching/roma_homography_tps_2024/
└── aligned/
    └── <scene_name>/
        ├── matches_filtered.json
        ├── matches_raw_*.json
        └── tile_diagnostics*.csv/json
```

The most important output file is:

```text
matches_filtered.json
```

This file contains the final filtered correspondences used during georeferencing.

---

# Code Structure

```text
src/matching/
├── methods/
│   └── roma_homography_tps.py
├── models/
│   ├── base.py
│   └── roma.py
└── utils/
    ├── io_grid.py
    └── matching_helpers.py
```

| File | Purpose |
|---|---|
| `methods/roma_homography_tps.py` | Main matching pipeline |
| `models/base.py` | Shared matcher structures and geometric filtering |
| `models/roma.py` | RoMa inference wrapper |
| `utils/io_grid.py` | Image loading, reprojection, and normalization |
| `utils/matching_helpers.py` | Tiling, diagnostics, duplicate merging, and helper utilities |

---

# Method Overview

## 1. Initial Feature Matching

RoMa is first applied to the complete ArmSat-1 and Sentinel-2 images in order to estimate an initial set of dense correspondences.

## 2. Global Geometric Filtering

The initial correspondences are filtered using a global homography model estimated with RANSAC. This stage removes large-scale outliers and provides a coarse geometric transformation between the images.

## 3. Tile-Level Local Refinement

After coarse filtering, the estimated geometric model is used to predict corresponding Sentinel-2 regions for local refinement.

The images are processed using overlapping tiles, and RoMa is applied again locally in order to improve correspondence estimation in regions affected by local geometric distortions.

## 4. Match Merging and Deduplication

The coarse and tile-level correspondences are merged and duplicate matches are removed before final filtering.

Tile-level matches are additionally checked for geometric consistency with the coarse transformation.

## 5. TPS-Based Geometric Filtering

Final filtering is performed using a thin-plate spline (TPS) consistency model.

The TPS model removes geometrically inconsistent correspondences while preserving smooth local deformations that cannot be represented accurately using a single global transformation.

The final filtered correspondences are exported for the georeferencing stage.

---

# Example Command

The following command demonstrates how the matching pipeline can be executed on a directory of preprocessed scenes.

```bash
python -m matching.methods.roma_homography_tps \
  --scenes_root outputs/preprocessing/2024/scenes \
  --out_dir outputs/matching/roma_homography_tps_2024 \
  --roma_variant outdoor \
  --tile_size 1120 \
  --tile_overlap 64
```

Before running from the repository root:

```bash
export PYTHONPATH=$PWD/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

# Main Arguments

| Argument | Meaning |
|---|---|
| `--scenes_root` | Directory containing preprocessed scene folders |
| `--out_dir` | Output directory for matching results |
| `--roma_variant` | RoMa model variant |
| `--roma_sample_num` | Number of coarse matching samples |
| `--roma_sample_num_tile` | Number of tile-level matching samples |
| `--ransac_reproj_thresh_px` | Homography RANSAC reprojection threshold |
| `--tile_size` | Tile size used for local refinement |
| `--tile_overlap` | Overlap between neighboring tiles |
| `--refine_margin_px` | Margin around predicted reference windows |
| `--final_min_matches` | Minimum matches required for export |
| `--max_export_median_px` | Maximum allowed median residual |
| `--max_export_p90_px` | Maximum allowed 90th-percentile residual |
```