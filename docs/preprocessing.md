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

# Important Arguments

| Argument | Meaning |
|---|---|
| `--input_dir` | Directory containing original ArmSat-1 scenes |
| `--out_dir` | Output directory |
| `--utm_epsg` | Target UTM CRS |
| `--buffer_deg` | Extra WGS84 bbox margin |
| `--window_days` | Sentinel-2 temporal search window |
| `--resolution` | Sentinel-2 download resolution |
| `--gee_project` | Google Earth Engine project ID |
| `--max_cloud_pct` | Maximum Sentinel-2 cloud percentage |
| `--gee_composite` | Sentinel-2 compositing strategy |
| `--prefer_band` | Driver band used for bbox/date processing |

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
├── run_report_preprocessing_local_<timestamp>.csv
└── run_report_preprocessing_local_<timestamp>.json
```