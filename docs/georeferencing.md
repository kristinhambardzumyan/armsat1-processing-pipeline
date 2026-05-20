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

A global report is additionally written:

```text
georef_report.csv
```

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

# Main Arguments

| Argument | Meaning |
|---|---|
| `--template_scenes_root` | Preprocessing scene directory |
| `--original_scenes_root` | Original ArmSat-1 scenes |
| `--aligned_root` | Matching output directory |
| `--out_dir` | Georeferencing output directory |
| `--min_gcps` | Minimum required GCPs |
| `--max_gcps` | Maximum selected GCPs |
| `--grid_size` | Grid size used for spatial balancing |
| `--resampling` | GDAL resampling method |

```