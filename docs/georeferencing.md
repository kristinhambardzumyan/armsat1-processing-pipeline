# Georeferencing

The georeferencing stage transforms the original ArmSat-1 imagery into geographically aligned scenes using the filtered correspondences produced during the matching stage.

The final transformation is performed using GDAL-based thin-plate spline (TPS) warping.

The georeferencing pipeline consists of:

```text
Filtered Correspondence Loading
    ↓
Spatially Balanced GCP Selection
    ↓
Coordinate Transformation
    ↓
GCP Attachment
    ↓
GDAL TPS-Based Warping
    ↓
Georeferenced ArmSat-1 Output
```

The resulting georeferenced scenes are aligned to the Sentinel-2 reference imagery.

---

# Inputs

The georeferencing stage expects:

```text
Preprocessing outputs:
    armsat_rgb_utm.tif

Matching outputs:
    matches_filtered.json

Original ArmSat-1 bands:
    *_R.tif
    *_G.tif
    *_B.tif
    *_N.tif
```

The matching correspondences are used as ground control points (GCPs) for TPS-based image warping.

---

# Outputs

The georeferencing stage writes:

```text
outputs/georeferencing/
└── <scene_name>/
    ├── <scene>_R_georef_tps.tif
    ├── <scene>_G_georef_tps.tif
    ├── <scene>_B_georef_tps.tif
    ├── <scene>_N_georef_tps.tif
    ├── georef_metadata.json
    └── *_gcps.vrt
```

A global CSV/JSON report is additionally written:

```text
georef_report.csv
georef_report.json
```

Parallel jobs also retain uniquely named per-job reports and merge their scene rows
into these stable report files under a file lock.

---

# Code Structure

```text
src/georeferencing/
├── georeferencing.py
```

The georeferencing module:

- loads filtered correspondences
- selects spatially balanced GCPs
- converts coordinates between reference and source CRS
- attaches GCPs using GDAL
- applies TPS warping
- exports georeferenced ArmSat-1 imagery
- writes metadata and reports

---

# Method Overview

## 1. Correspondence Loading

The filtered correspondences produced during the matching stage are loaded from:

```text
matches_filtered.json
```

These matches represent corresponding locations between ArmSat-1 and Sentinel-2 imagery.

## 2. Spatially Balanced GCP Selection

The full set of correspondences is spatially balanced using a grid-based selection strategy.

This prevents GCPs from becoming concentrated in only a small part of the image and improves TPS stability across the scene.

## 3. Coordinate Transformation

The selected correspondence locations are converted between:

- reference image coordinates
- geographic coordinates
- original ArmSat-1 image coordinates

This step allows the final TPS transformation to be applied directly to the original ArmSat-1 bands.

## 4. GCP Attachment

The selected GCPs are attached to the original ArmSat-1 imagery using GDAL virtual raster (VRT) files.

## 5. TPS-Based Warping

GDAL TPS warping is then applied in order to produce the final georeferenced output aligned to the Sentinel-2 reference imagery.

The original ArmSat-1 bands are transformed independently and exported as georeferenced GeoTIFF files.

---

# Example Command

The following command demonstrates how georeferencing can be executed.

```bash
python -m georeferencing.georeferencing \
  --template_scenes_root outputs/preprocessing/2023/scenes \
  --original_scenes_root /path/to/Armsat-1 \
  --aligned_root outputs/matching/roma_homography_tps_2023/aligned \
  --out_dir outputs/georeferencing/2023
```

Before running from the repository root:

```bash
export PYTHONPATH=$PWD/src
```

---

# Complete Georeferencing Argument Reference

| Argument | Required | Default | Meaning |
|---|---:|---|---|
| `--template_scenes_root` | yes | — | Preprocessed scene root containing reference rasters |
| `--original_scenes_root` | yes | — | Root containing original ArmSat bands |
| `--aligned_root` | yes | — | Matching output root or its `aligned` directory |
| `--out_dir` | yes | — | Georeferencing output directory |
| `--scene` | no | `None` | Process one exact scene |
| `--scene_list` | no | `None` | Process scenes from a text file; ignored when `--scene` is used |
| `--min_gcps` | no | `12` | Minimum GCP count required to warp a scene |
| `--max_gcps` | no | `1200` | Maximum spatially balanced GCPs selected |
| `--grid_size` | no | `16` | Grid dimension used for spatial balancing |
| `--min_conf_eval` | no | `0.0` | Minimum match confidence used for GCP selection |
| `--resampling` | no | `bilinear` | GDAL resampling: `near`, `bilinear`, or `cubic` |
| `--quiet` | no | off | Suppress verbose GDAL subprocess output |

---

# Multi-Reference Selection and Reports

Multi-reference selection does not require a separate georeferencing argument.
Georeferencing reads `selected_reference_candidate` from the scene's
`matches_filtered.json` and uses that candidate GeoTIFF as the reference grid.
When the field is absent, the original center-reference behavior remains in
effect.

The template lookup accepts `armsat_rgb_utm*.tif` and falls back to
`armsat_rgb_native.tif` when preprocessing determined that the native raster was
already in the requested UTM CRS. Scene-list entries may be either scene names or
complete scene-directory paths.

Parallel georeferencing processes write uniquely named per-process CSV and JSON
reports. The stable `georef_report.csv` and `georef_report.json` are merged under
a file lock and replaced atomically, so one process cannot erase another
process's scene rows.

After georeferencing, selected-reference versus output previews can be generated
with:

```bash
python scripts/make_georef_side_by_side.py \
  --preprocessing_scenes_root outputs/preprocessing/scenes \
  --matching_aligned_root outputs/matching/aligned \
  --georeferencing_root outputs/georeferencing \
  --out_dir outputs/georeferencing_side_by_side \
  --scene_list /path/to/scenes.txt \
  --max_side 900
```

Preview arguments:

| Argument | Required | Default | Meaning |
|---|---:|---|---|
| `--preprocessing_scenes_root` | yes | — | Prepared scene directories containing reference candidates |
| `--matching_aligned_root` | yes | — | Matching `aligned` directory containing `matches_filtered.json` |
| `--georeferencing_root` | yes | — | Final georeferencing output directory |
| `--out_dir` | yes | — | Destination for JPEGs and `side_by_side_report.json` |
| `--scene_list` | no | `None` | Optional scene-name/path list |
| `--max_side` | no | `900` | Maximum panel dimension |
