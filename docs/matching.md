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
        ├── reference_candidates.json
        └── tile_diagnostics*.csv/json
```

The most important output file is:

```text
matches_filtered.json
```

This file contains the final filtered correspondences used during georeferencing.

When preprocessing configured multiple references, the existing coarse matching and
geometric verification are applied to each crop. The candidate with the largest
coarse inlier count is selected (the center wins ties), and the existing tile and TPS
stages continue with that crop. `reference_candidates.json` and
`matches_filtered.json` record each direction, crop bounds, initial match count,
inlier count, inlier ratio, and the selected candidate.

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

# Complete Argument Reference

Inputs, reproducibility, and RoMa:

| Argument | Required | Default | Meaning |
|---|---:|---|---|
| `--scenes_root` | yes | — | Root containing preprocessed scene directories |
| `--out_dir` | yes | — | Matching output root |
| `--seed` | no | `42` | Random seed |
| `--no_deterministic_cudnn` | no | off | Disable deterministic cuDNN settings |
| `--roma_variant` | no | `outdoor` | RoMa weights: `outdoor` or `indoor` |
| `--roma_sample_num` | no | `12000` | Coarse-stage sample count |
| `--roma_sample_num_tile` | no | `None` | Tile sample count; reuses the coarse count when omitted |
| `--roma_sample_thresh` | no | `None` | Optional RoMa sampling threshold |
| `--roma_device` | no | `None` | Explicit PyTorch device; automatically selected when omitted |
| `--roma_w_resized` | no | `1120` | RoMa resized input width |
| `--roma_h_resized` | no | `1120` | RoMa resized input height |
| `--roma_upsample_w` | no | `1120` | RoMa upsample width |
| `--roma_upsample_h` | no | `1120` | RoMa upsample height |

Coarse verification and TPS:

| Argument | Default | Meaning |
|---|---|---|
| `--coarse_conf_keep_pct` | `60` | Coarse confidence percentage retained |
| `--coarse_min_conf` | `0.20` | Minimum coarse confidence |
| `--coarse_min_matches` | `80` | Preferred minimum before coarse filtering |
| `--coarse_spatial_cell_px` | `128` | Coarse spatial-balancing cell size |
| `--coarse_spatial_max_per_cell` | `8` | Maximum coarse matches per cell |
| `--min_export_matches` | `12` | Absolute minimum exportable match count |
| `--min_tps_matches_total` | `12` | Minimum matches required for TPS fitting |
| `--min_tps_train_matches_per_fold` | `8` | Minimum TPS training matches per fold |
| `--tps_smoothing` | `1e-3` | TPS smoothing parameter |
| `--tps_abs_thresh_px` | `6.0` | Absolute TPS residual threshold in pixels |
| `--tps_mad_mult` | `3.0` | MAD multiplier for the adaptive TPS threshold |
| `--tps_max_iters` | `5` | Maximum TPS filtering iterations |
| `--tps_max_remove_frac` | `0.20` | Maximum fraction removed per TPS iteration |
| `--cv_block_px` | `256` | Spatial cross-validation block size |
| `--cv_max_folds` | `4` | Maximum cross-validation folds |
| `--ransac_reproj_thresh_px` | `8.0` | Homography RANSAC reprojection threshold |
| `--ransac_confidence` | `0.999` | RANSAC confidence |
| `--ransac_max_iters` | `5000` | Maximum RANSAC iterations |
| `--ransac_refine_iters` | `10` | Homography refinement iterations |

Tile refinement, merging, and export:

| Argument | Default | Meaning |
|---|---|---|
| `--tile_size` | `1120` | Tile width and height in pixels |
| `--tile_overlap` | `64` | Overlap between adjacent tiles |
| `--tile_min_valid_frac` | `0.05` | Minimum valid-data fraction per tile |
| `--tile_min_side_px` | `256` | Minimum accepted tile side length |
| `--refine_margin_px` | `256` | Margin around the predicted reference window |
| `--tile_conf_keep_pct` | `60` | Tile confidence percentage retained |
| `--tile_min_conf` | `0.20` | Minimum tile confidence |
| `--tile_spatial_cell_px` | `128` | Tile-stage spatial-balancing cell size |
| `--tile_spatial_max_per_cell` | `4` | Maximum tile matches per cell |
| `--tile_prior_consistency_px` | `64.0` | Maximum deviation from the coarse prediction |
| `--tile_worker_batch_size` | `4` | Tiles handled by each short-lived worker |
| `--merge_src_round_px` | `0.5` | Source-coordinate rounding for deduplication |
| `--merge_dst_round_px` | `0.5` | Reference-coordinate rounding for deduplication |
| `--final_conf_keep_pct` | `60` | Final confidence percentage retained |
| `--final_min_conf` | `0.20` | Minimum final confidence |
| `--final_min_matches` | `80` | Preferred minimum before final filtering |
| `--final_spatial_cell_px` | `128` | Final spatial-balancing cell size |
| `--final_spatial_max_per_cell` | `8` | Maximum final matches per cell |
| `--max_export_median_px` | `10.0` | Maximum accepted median residual |
| `--max_export_p90_px` | `30.0` | Maximum accepted 90th-percentile residual |
| `--save_all` | off | Save results that exceed residual quality limits |

---

# Multi-Reference Matching

There is no additional matching command-line argument. Matching discovers the
candidate list written by preprocessing in each scene's
`scene_prep_summary.json`. A scene without that metadata follows the original
single-reference path.

For every configured candidate, the existing RoMa coarse matcher and existing
geometric filter compute:

- candidate direction and crop bounds;
- initial match count;
- geometrically verified inlier count;
- inlier ratio and rejection status.

The viable candidate with the largest inlier count is marked as selected. Only
that candidate proceeds through the existing tile refinement, merge, final
filtering, and TPS path, avoiding duplicate refinement logic and unnecessary work
for candidates that were not selected.

Candidate evaluation is written to:

```text
<out_dir>/aligned/<scene>/reference_candidates.json
```

The same candidate diagnostics and a `selected_reference_candidate` object are
also included in `matches_filtered.json`. The selected object contains
`direction`, `crop_bounds`, and `reference_path`; georeferencing consumes it
automatically.

Both `armsat_rgb_utm*.tif` and an already-UTM `armsat_rgb_native.tif` are accepted
as the ArmSat matching input. This fallback is important for preprocessing runs
where reprojection was unnecessary and therefore no separate `_utm` file was
created.
