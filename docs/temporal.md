# Temporal Tile Extraction

The temporal extraction stage builds multi-date ArmSat-1 tile sequences from georeferenced scenes.

Its purpose is to identify geographic areas observed by ArmSat-1 on multiple acquisition dates and extract aligned RGBN observations for those areas.

---

# Method Overview

The block-based temporal pipeline consists of:

```text
Georeferenced ArmSat-1 Scenes
    ↓
Scene Index Construction
    ↓
Fixed Spatial Block Generation
    ↓
Date-Count Coverage Raster Creation
    ↓
Repeated-Area Candidate Search
    ↓
Temporal Tile Extraction
    ↓
Multi-Date RGBN Tile Sequences
```

---

# Inputs

The temporal pipeline expects georeferenced ArmSat-1 outputs from both years:

```text
outputs/georeferencing/2023
outputs/georeferencing/2024
```

Each scene folder should contain georeferenced bands:

```text
<scene>_R.tif
<scene>_G.tif
<scene>_B.tif
<scene>_N.tif
```

---

# Code Structure

```text
src/temporal/
├── scene_index.py
├── blocks.py
├── coverage.py
├── extract_tiles.py
└── run_block_pipeline.py
```

| File | Purpose |
|---|---|
| `scene_index.py` | Builds a geospatial index of scenes, dates, bounds, band paths, and metadata |
| `blocks.py` | Builds fixed spatial processing blocks over the full study area |
| `coverage.py` | Creates date-count rasters and repeated-area masks |
| `extract_tiles.py` | Extracts temporal RGBN tile sequences |
| `run_block_pipeline.py` | Runs the full block-based temporal pipeline |

The scene index extracts scene IDs, acquisition dates, band paths, bounds, resolution, NoData values, and footprints. 

The block builder creates fixed spatial blocks with an expanded margin around each block.

The coverage step builds a date-count raster where each pixel stores the number of distinct acquisition dates covering that location.

The extraction step selects repeated tile candidates and writes per-date RGBN observations. 

---

# Recommended Pipeline

The recommended implementation is the block-based pipeline:

```text
src/temporal/run_block_pipeline.py
```

This avoids creating one very large coverage raster for the entire study area. Instead, it processes the region block by block.

---

# Example Command

```bash
python -u src/temporal/run_block_pipeline.py \
  --roots \
    outputs/georeferencing/2023 \
    outputs/georeferencing/2024 \
  --out_root outputs/temporal/block_outputs \
  --target_crs EPSG:32638 \
  --block_size_m 20000 \
  --block_margin_m 2048 \
  --res_m 2.0 \
  --tile_size_px 512 \
  --candidate_step_px 128 \
  --min_distinct_dates 2 \
  --min_repeated_fraction 0.99 \
  --valid_fraction_threshold 0.99 \
  --apply_scale_offset
```

Before running from the repository root:

```bash
export PYTHONPATH=$PWD/src
export PYTHONUNBUFFERED=1
```

---

# Main Arguments

| Argument | Meaning |
|---|---|
| `--roots` | Georeferenced scene roots |
| `--out_root` | Temporal output directory |
| `--target_crs` | Common CRS used for temporal extraction |
| `--block_size_m` | Size of each fixed processing block in meters |
| `--block_margin_m` | Extra margin around each block |
| `--res_m` | Output spatial resolution in meters |
| `--tile_size_px` | Temporal tile size in pixels |
| `--candidate_step_px` | Step size for scanning candidate repeated tiles |
| `--min_distinct_dates` | Minimum number of acquisition dates required |
| `--min_repeated_fraction` | Required fraction of tile area covered by repeated observations |
| `--valid_fraction_threshold` | Minimum valid-pixel fraction for each observation |
| `--apply_scale_offset` | Applies scale/offset metadata when reading bands |

---

# Output Structure

The block pipeline writes:

```text
outputs/temporal/block_outputs/
├── scene_index.csv
├── scene_footprints.gpkg
├── processing_blocks.gpkg
├── block_processing_summary.csv
├── config.json
└── BLK_000001/
    ├── coverage/
    │   ├── BLK_000001_date_count.tif
    │   └── BLK_000001_repeated_mask.tif
    ├── metadata/
    │   ├── BLK_000001_tiles.csv
    │   ├── BLK_000001_observations.csv
    │   └── BLK_000001_tile_footprints.gpkg
    └── tiles/
        └── BLK_000001_TS_000001/
            └── observations/
                ├── BLK_000001_TS_000001_20230307_ARMSAT1_RGBN_512_2m.tif
                └── BLK_000001_TS_000001_20240412_ARMSAT1_RGBN_512_2m.tif
```

Each temporal tile contains multiple observations of the same geographic area on different acquisition dates.

---

# Output Files

## `scene_index.csv`

Contains one row per georeferenced scene, including:

- scene ID
- acquisition date
- band paths
- bounds
- resolution
- NoData value

## `processing_blocks.gpkg`

Contains fixed spatial blocks used to divide the study area.

## `*_date_count.tif`

A raster where each pixel stores how many distinct acquisition dates cover that location.

## `*_repeated_mask.tif`

A binary raster showing where the date count is at least `min_distinct_dates`.

## `*_tiles.csv`

Metadata for selected temporal tiles.

## `*_observations.csv`

Metadata for all per-date observations written for each temporal tile.

---

# Notes

The temporal pipeline writes float32 RGBN outputs.

When `--apply_scale_offset` is enabled, source scale and offset metadata are applied while extracting observations.