# Preprocessing

The preprocessing stage prepares ArmSat-1 scenes for later matching and georeferencing.

Its main purpose is to create a geometrically comparable ArmSat-1 / Sentinel-2 image pair for each scene.

For every ArmSat-1 scene, this step:

1. Finds the original ArmSat-1 band files (`R`, `G`, `B`, `N`).
2. Builds an RGB ArmSat image from the `R`, `G`, and `B` bands.
3. Extracts the acquisition date from the ArmSat filename.
4. Computes the scene bounding box in WGS84 coordinates.
5. Expands the bounding box by a configurable buffer.
6. Downloads a Sentinel-2 RGB reference image from Google Earth Engine.
7. Reprojects the ArmSat RGB image to the target UTM CRS.
8. Warps the Sentinel-2 image onto the exact ArmSat grid.
9. Saves per-scene metadata and global preprocessing reports.

The output of this stage is later used for feature matching and georeferencing.

---

# Inputs

The preprocessing pipeline expects original ArmSat-1 band files:

```text
*_R.tif
*_G.tif
*_B.tif
*_N.tif
```

Quicklook (`1QK`) files are ignored automatically.

---

# Outputs

For each scene, preprocessing produces:

- ArmSat RGB image
- Reprojected ArmSat RGB image
- Sentinel-2 RGB reference image
- Sentinel-2 image warped to the ArmSat grid
- Scene metadata summary
- Global CSV and JSON reports

---

# Code Structure

```text
src/preprocessing/
├── run.py
├── geo.py
├── s2_gee.py
└── report.py
```

| File | Purpose |
|---|---|
| `run.py` | Main preprocessing entry point |
| `geo.py` | Geospatial utilities, RGB construction, reprojection, GDAL warping |
| `s2_gee.py` | Google Earth Engine Sentinel-2 download/export logic |
| `report.py` | CSV and JSON preprocessing reports |

---

# Google Earth Engine Setup

This stage requires access to Google Earth Engine.

Before running preprocessing:

1. Create a Google Earth Engine account.
2. Create or register a GEE project.
3. Install the Earth Engine Python API.
4. Authenticate on the machine where the code will run.

Authenticate once:

```bash
earthengine authenticate
```

Test initialization:

```bash
python -c "import ee; ee.Initialize(project='project-name'); print('EE OK')"
```

Replace `project-name` with your own Earth Engine project ID.

---

# Preprocessing Arguments

| Argument | Required | Default | Meaning |
|---|---:|---|---|
| `--input_dir` | yes | — | Directory containing original ArmSat band GeoTIFFs |
| `--out_dir` | yes | — | Preprocessing output directory |
| `--utm_epsg` | no | `32638` | Target UTM EPSG code |
| `--buffer_deg` | no | `0.04` | Extra WGS84 margin around the required reference extent |
| `--window_days` | no | `60` | Temporal search window before and after the ArmSat date |
| `--resolution` | no | `10.0` | Sentinel-2 download/export resolution in metres |
| `--prefer_band` | no | `R` | Preferred driver band: `R`, `G`, `B`, or `N` |
| `--gee_project` | no | `None` | Google Earth Engine project ID |
| `--max_cloud_pct` | no | `100.0` | Maximum Sentinel-2 cloud percentage |
| `--gee_composite` | no | `least_cloudy_mosaic` | Sentinel-2 compositing strategy |
| `--gee_download_mode` | no | `local` | `local` direct download or `drive` export |
| `--drive_folder` | no | `armsat_sentinel_exports` | Drive folder used for Earth Engine exports |
| `--start_idx` | no | `0` | First sorted scene index when no scene list is supplied |
| `--end_idx` | no | `None` | Exclusive final sorted scene index |
| `--scene_list` | no | `None` | File containing selected scene names or paths |
| `--multi_reference_config` | no | `None` | Per-scene reference configuration; omit for center-only processing |
| `--multi_reference_all` | no | off | Apply one multi-reference configuration to every scene processed in the run |
| `--multi_reference_overlap` | no | `None` | Shared overlap; required with `--multi_reference_all` |
| `--multi_reference_directions` | no | all eight | Shared directions used with `--multi_reference_all` |

---

# Example Command

The following command demonstrates how preprocessing can be executed.

Paths and the Earth Engine project ID depend on the local environment.

```bash
python src/preprocessing/run.py \
  --input_dir /path/to/Armsat-1 \
  --out_dir outputs/preprocessing/2023 \
  --utm_epsg 32638 \
  --buffer_deg 0.02 \
  --window_days 60 \
  --resolution 10 \
  --gee_project project-name \
  --max_cloud_pct 10 \
  --gee_composite least_cloudy_mosaic \
  --prefer_band R
```

---

# Output Structure

```text
outputs/preprocessing/2023/
├── scenes/
│   └── <scene_name>/
│       ├── armsat_rgb_native.tif
│       ├── armsat_rgb_utm.tif
│       ├── sentinel2_rgb_raw.tif
│       ├── sentinel2_rgb_utm.tif
│       └── scene_prep_summary.json
├── logs/
│   └── preprocessing_<timestamp>.log
├── run_report_preprocessing_local.csv
├── run_report_preprocessing_local.json
└── run_report_preprocessing_local_<unique_job_id>.csv/json
```

Parallel jobs write immutable, uniquely named per-job reports and update the stable
CSV/JSON pair under a file lock. The stable reports merge all scene results, so a
finishing job cannot overwrite rows written by another job.

## Reference modes and neighboring crops

Pass `--multi_reference_config /path/to/multi_reference.json` to create additional
Sentinel-2 crops around the original center reference. Every candidate has the
same pixel dimensions, resolution, and CRS as the center crop; only its spatial
bounds are shifted.

The candidate positions form a 3 × 3 neighborhood:

```text
NW   N   NE
 W   C    E
SW   S   SE
```

`C` is the original center reference and is always included automatically.
Do not add `C` to `directions`.

The configuration is a JSON object keyed by exact scene name:

```json
{
  "image_001": {"overlap": 0.5},
  "image_057": {
    "overlap": 0.5,
    "directions": ["W", "NW", "N"]
  }
}
```

Configuration fields:

| Field | Required | Accepted values | Meaning |
|---|---:|---|---|
| `overlap` | yes | Number from `0` up to but not including `1` | Fraction of the center crop shared with each neighboring crop |
| `directions` | no | Any subset of `NW`, `N`, `NE`, `W`, `E`, `SW`, `S`, `SE` | Neighbor directions to create; omitting it creates all eight |

### Overlap

The neighbor shift is:

```text
shift = crop size × (1 - overlap)
```

For example:

- `overlap: 0.5` shifts the neighbor by half a crop width or height. A cardinal
  neighbor shares 50% of the crop area with `C`; a diagonal neighbor shares
  50% of the width and 50% of the height, or 25% of the total area.
- `overlap: 0.75` shifts it by one quarter of the crop size.
- `overlap: 0` produces adjacent crops with no shared area.

Diagonal candidates are shifted in both horizontal and vertical directions.

### Directions

When `directions` is omitted, the result is:

```text
C, NW, N, NE, W, E, SW, S, SE
```

When directions are supplied, only those neighbors are added. For example:

```json
{
  "image_057": {
    "overlap": 0.5,
    "directions": ["W", "NW", "N"]
  }
}
```

creates `C`, `W`, `NW`, and `N`. Direction names are case-insensitive,
duplicates are removed, and the configured order is preserved.

### Which scenes use multiple references

- A scene absent from the JSON uses only `C`.
- A listed scene uses `C` plus its configured neighbors.
- `--multi_reference_all` applies one overlap/direction configuration to every
  scene processed in the run without listing scene names in JSON.
- Omitting `--multi_reference_config` preserves the center-only workflow for
  every scene unless `--multi_reference_all` is used.

For local GEE downloads, candidates are generated immediately. For Drive mode,
pass the same `--multi_reference_config` during export submission so Earth
Engine exports a source region large enough for every requested neighbor. Pass
the configuration again to `preprocessing.finalize_drive_exports` after
downloading the exported GeoTIFFs. The finalizer creates `C` and its neighbors
and records their directions, paths, and crop bounds in
`scene_prep_summary.json`.

---

# Multi-Reference Commands and Arguments

The multi-reference configuration is keyed by the exact scene directory name.
`overlap` is required for a listed scene and must satisfy `0 <= overlap < 1`.
`directions` is optional; omitting it selects all eight directions. The center
(`C`) is always produced.

The preprocessing entry point provides two mutually exclusive modes:

| Argument | Mode | Meaning |
|---|---|---|
| `--multi_reference_config` | Selected scenes | JSON configuration keyed by exact scene name |
| `--multi_reference_all` | All processed scenes | Enable one shared configuration for every scene in the run |
| `--multi_reference_overlap` | All processed scenes | Required overlap value for `--multi_reference_all` |
| `--multi_reference_directions` | All processed scenes | Optional directions; omitting them creates all eight neighbors |

`--multi_reference_config` and `--multi_reference_all` cannot be combined.
`--scene_list` independently controls which scenes are processed. If both
`--scene_list` and `--multi_reference_config` are omitted while
`--multi_reference_all` is enabled, every scene discovered in `--input_dir`
uses the shared settings.

For example, apply `C`, `W`, `NW`, and `N` to every discovered scene:

```bash
python -m preprocessing.run \
  --input_dir /path/to/armsat_scenes \
  --out_dir outputs/preprocessing \
  --gee_project your-gee-project \
  --multi_reference_all \
  --multi_reference_overlap 0.5 \
  --multi_reference_directions W NW N
```

For a local Earth Engine download, pass the chosen mode arguments directly to
`preprocessing.run`. For Drive mode, pass the same mode and settings when
submitting the exports and when finalizing them. Export submission determines
the enlarged download extent, while finalization creates the individual grids.

```bash
python -m preprocessing.finalize_drive_exports \
  --export_dirs /path/to/drive_exports_1 /path/to/drive_exports_2 \
  --scenes_root outputs/preprocessing/scenes \
  --utm_epsg 32638 \
  --scene_list /path/to/scenes.txt \
  --multi_reference_config configs/multi_reference.json
```

Drive finalizer arguments:

| Argument | Required | Default | Meaning |
|---|---:|---|---|
| `--export_dirs` | yes | — | One or more directories containing downloaded Drive GeoTIFFs |
| `--scenes_root` | yes | — | Preprocessing `scenes` directory |
| `--utm_epsg` | yes | — | Target EPSG code |
| `--scene_list` | no | `None` | Limit finalization to the listed scenes |
| `--multi_reference_config` | no | `None` | Same configuration used during export submission |
| `--multi_reference_all` | no | off | Apply shared settings to every scene being finalized |
| `--multi_reference_overlap` | no | `None` | Shared overlap; required with `--multi_reference_all` |
| `--multi_reference_directions` | no | all eight | Shared directions; omit for all eight |

Create previews of the ArmSat image against every available reference candidate:

```bash
python -m preprocessing.make_reference_previews \
  --scenes_root outputs/preprocessing/scenes \
  --out_dir outputs/reference_previews \
  --scene_list /path/to/scenes.txt \
  --max_side 600
```

Reference preview arguments:

| Argument | Required | Default | Meaning |
|---|---:|---|---|
| `--scenes_root` | yes | — | Prepared scene root |
| `--out_dir` | yes | — | Preview output directory |
| `--max_side` | no | `1600` | Maximum preview width or height |
| `--scene_list` | no | `None` | Optional scene-name/path list |

Validation accepts either `--config` or `--multi_reference_all`:

| Argument | Required | Default | Meaning |
|---|---:|---|---|
| `--scenes_root` | yes | — | Prepared scene root |
| `--config` | one mode required | `None` | Per-scene reference configuration |
| `--multi_reference_all` | one mode required | off | Validate shared settings for every listed scene |
| `--multi_reference_overlap` | with all-scenes mode | `None` | Shared overlap |
| `--multi_reference_directions` | no | all eight | Shared directions |
| `--scene_list` | no | `None` | Optional validation subset; all scene directories are used in all-scenes mode, while configured JSON keys are used in per-scene mode |
| `--previews_dir` | yes | — | Directory containing generated previews |
| `--report` | yes | — | JSON validation-report destination |

The validator checks candidate presence, direction order, grid geometry, spatial
shifts, and preview existence without running matching or inference.
