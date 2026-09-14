# ArmSat-1 Processing Pipeline

Automated processing pipeline for ArmSat-1 satellite imagery.

The pipeline includes:

- preprocessing and Sentinel-2 reference retrieval
- feature matching and geometric refinement
- georeferencing and image warping
- georeferencing evaluation
- RGBN patch dataset construction
- clear/cloud/shadow/snow tile classification
- multi-temporal tile extraction

---

# Pipeline Overview

```text
Raw ArmSat-1 Scenes
    ↓
Sentinel-2 Reference Retrieval + Preprocessing
    ↓
Feature Matching
    ↓
Geometric Filtering and Refinement
    ↓
Georeferencing
    ↓
Patch Dataset Construction
    ↓
Patch Classification
    ↓
Temporal Tile Extraction
```

---

# Repository Structure

```text
armsat1-processing-pipeline/
├── src/
│   ├── preprocessing/
│   ├── matching/
│   ├── georeferencing/
│   ├── evaluation/
│   ├── tiling/
│   ├── patch_classifier/
│   └── temporal/
├── docs/
├── outputs/
├── requirements.txt
├── environment.yml
└── README.md
```

---

# Main Components

| Stage | Description |
|---|---|
| Preprocessing | Builds comparable ArmSat-1 and Sentinel-2 image pairs and downloads Sentinel-2 reference imagery from Google Earth Engine |
| Matching | Estimates correspondences using RoMa and geometric filtering |
| Georeferencing | Warps original ArmSat-1 imagery into Sentinel-2 reference coordinates |
| Evaluation | Measures alignment accuracy using manually selected reference points |
| Tiling | Builds RGBN patch datasets from georeferenced scenes |
| Patch Classification | Predicts clear/cloud/shadow/snow labels for RGBN tiles |
| Temporal Extraction | Builds multi-date temporal RGBN tile sequences |

---

# Documentation

Detailed documentation for each stage is available in:

```text
docs/
├── preprocessing.md
├── matching.md
├── georeferencing.md
├── evaluation.md
├── patch_classifier.md
└── temporal.md
```

Recommended reading order:

1. `preprocessing.md`
2. `matching.md`
3. `georeferencing.md`
4. `evaluation.md`
5. `patch_classifier.md`
6. `temporal.md`

---

# Installation

Python 3.10 is recommended. Install the dependencies with either
`requirements.txt` or `environment.yml`.

### Using requirements.txt

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Using environment.yml (optional)

This option requires Conda or Miniconda to be installed already. The file creates
an environment named `armsat-pipeline`; this is only a local environment name
and may be changed by the user.

```bash
conda env create -f environment.yml
conda activate armsat-pipeline
```

GDAL command-line tools must be installed and available on `PATH`:

```bash
gdalinfo --version
gdalwarp --version
```

Run pipeline commands from the repository root with `src` on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/src"
```

Google Earth Engine access is required for Sentinel-2 retrieval. PyTorch and RoMa
are required for matching; a compatible accelerator is recommended for practical
matching speed.

Detailed arguments and stage-specific setup are documented in:

- [Preprocessing](docs/preprocessing.md)
- [Matching](docs/matching.md)
- [Georeferencing](docs/georeferencing.md)
- [Patch classification](docs/patch_classifier.md)
- [Temporal processing](docs/temporal.md)

---

# Google Earth Engine

The preprocessing stage requires Google Earth Engine access.

Authenticate once:

```bash
earthengine authenticate
```

Test initialization:

```bash
python -c "import ee; ee.Initialize(project='your-project'); print('EE OK')"
```

---

# Model Checkpoints

Download the classifier checkpoint from:

[ArmSat-1 Model Checkpoints](https://drive.google.com/drive/folders/1WcZPbszdhc50F6bWuxjOiy8NvIt8Xo5M?usp=drive_link)

Place the checkpoint here:

```text
patch_classifier/models/resnet50_4ch_classifier.pth
```

---

# Outputs

Large raster outputs, intermediate files, and model checkpoints are not included in the repository.

The repository keeps only the folder structure required for pipeline execution.

---

# Notes

- Sentinel-2 imagery is used as the geographic reference source.
- Sentinel-2 RGB bands have native 10 m spatial resolution.
- ArmSat-1 imagery has approximately 2 m spatial resolution.
- Final alignment precision is therefore limited by the Sentinel-2 reference resolution.
- The preprocessing stage automatically retrieves close-date Sentinel-2 imagery with low cloud coverage.
- The pipeline supports both 2023 and 2024 ArmSat-1 datasets.

---

# Reference Processing Workflow

The default workflow uses one center Sentinel-2 reference for each scene.
Selected scenes can instead use the center crop plus overlapping neighboring
crops.

The available reference positions are:

```text
NW   N   NE
 W   C    E
SW   S   SE
```

`C` is the original center reference. Every multi-reference scene keeps `C`
and adds the configured neighbors around it.

## 1. Create a scene list

A scene list contains one exact scene name or scene-directory path per line:

```text
EXACT_SCENE_NAME_001
EXACT_SCENE_NAME_057
```

Blank lines and lines beginning with `#` are ignored.

## 2. Configure reference candidates

Create `configs/multi_reference.json`:

```json
{
  "EXACT_SCENE_NAME_001": {
    "overlap": 0.5
  },
  "EXACT_SCENE_NAME_057": {
    "overlap": 0.5,
    "directions": ["W", "NW", "N"]
  }
}
```

- `overlap` is the fraction shared with the center crop. It must satisfy
  `0 <= overlap < 1`.
- Omitting `directions` creates all eight neighbors.
- The center candidate `C` is always included.
- Scenes absent from the file use only the center reference.

Omit `--multi_reference_config` entirely when all scenes should use one
reference.

Giving a scene only `overlap` creates `C` plus all eight neighbors. Giving it
`overlap` and `directions` creates `C` plus only those directions.

To apply one configuration to every scene processed in the run, use:

```bash
--multi_reference_all \
--multi_reference_overlap 0.5 \
--multi_reference_directions W NW N
```

Omit `--multi_reference_directions` to use all eight neighbors. This all-scenes
mode does not require a JSON configuration and cannot be combined with
`--multi_reference_config`. The scene list still determines which scenes run;
when `--scene_list` is also omitted, every scene discovered under
`--input_dir` is processed with the shared multi-reference settings.

## 3. Run preprocessing

Set the source path first:

```bash
export PYTHONPATH="$PWD/src"
```

For a direct Earth Engine download:

```bash
python -u -m preprocessing.run \
  --input_dir /path/to/original_armsat_scenes \
  --out_dir outputs/preprocessing \
  --utm_epsg 32638 \
  --buffer_deg 0.02 \
  --window_days 60 \
  --resolution 10 \
  --gee_project your-gee-project \
  --max_cloud_pct 10 \
  --gee_download_mode local \
  --scene_list /path/to/scenes.txt \
  --multi_reference_config configs/multi_reference.json
```

For Earth Engine Drive export, change the download mode and provide a Drive
folder:

```bash
python -u -m preprocessing.run \
  --input_dir /path/to/original_armsat_scenes \
  --out_dir outputs/preprocessing \
  --utm_epsg 32638 \
  --buffer_deg 0.02 \
  --window_days 60 \
  --resolution 10 \
  --gee_project your-gee-project \
  --max_cloud_pct 10 \
  --gee_download_mode drive \
  --drive_folder armsat_sentinel_exports \
  --scene_list /path/to/scenes.txt \
  --multi_reference_config configs/multi_reference.json
```

After the Drive tasks finish, download the GeoTIFFs and finalize them:

```bash
python -u -m preprocessing.finalize_drive_exports \
  --export_dirs /path/to/downloaded_drive_exports \
  --scenes_root outputs/preprocessing/scenes \
  --utm_epsg 32638 \
  --scene_list /path/to/scenes.txt \
  --multi_reference_config configs/multi_reference.json
```

The same reference configuration must be used for Drive submission and
finalization. For all-scenes mode, replace `--multi_reference_config` with
`--multi_reference_all`, `--multi_reference_overlap`, and optional
`--multi_reference_directions` in both commands.

## 4. Preview and validate references

This preview step shows the ArmSat image beside each available reference crop,
including `C` and all configured neighbors. It is intended for checking the
reference coverage before matching.

```bash
python -u -m preprocessing.make_reference_previews \
  --scenes_root outputs/preprocessing/scenes \
  --out_dir outputs/reference_previews \
  --scene_list /path/to/scenes.txt \
  --max_side 600

python -u scripts/validate_multi_reference_preprocessing.py \
  --scenes_root outputs/preprocessing/scenes \
  --config configs/multi_reference.json \
  --scene_list /path/to/scenes.txt \
  --previews_dir outputs/reference_previews \
  --report outputs/preprocessing_validation.json
```

For all-scenes mode, validate with the shared settings instead of `--config`:

```bash
python -u scripts/validate_multi_reference_preprocessing.py \
  --scenes_root outputs/preprocessing/scenes \
  --multi_reference_all \
  --multi_reference_overlap 0.5 \
  --multi_reference_directions W NW N \
  --previews_dir outputs/reference_previews \
  --report outputs/preprocessing_validation.json
```

## 5. Run matching

Matching reads the candidate information from `scene_prep_summary.json`. It
evaluates every candidate, selects the one with the highest verified inlier
count, and performs tile refinement and final TPS filtering with the selected
candidate.

```bash
python -u -m matching.methods.roma_homography_tps \
  --scenes_root outputs/preprocessing/scenes \
  --out_dir outputs/matching
```

Candidate metrics and selection are saved in
`outputs/matching/aligned/<scene>/reference_candidates.json` and
`matches_filtered.json`.

## 6. Run georeferencing

Georeferencing automatically uses the selected reference recorded by matching:

```bash
python -u -m georeferencing.georeferencing \
  --template_scenes_root outputs/preprocessing/scenes \
  --original_scenes_root /path/to/original_armsat_scenes \
  --aligned_root outputs/matching/aligned \
  --out_dir outputs/georeferencing \
  --scene_list /path/to/scenes.txt \
  --max_gcps 300 \
  --grid_size 16
```

## 7. Create final comparison previews

This preview is different from the reference-coverage preview above. It shows
the reference candidate selected by matching beside the final georeferenced
ArmSat RGB image.

```bash
python -u scripts/make_georef_side_by_side.py \
  --preprocessing_scenes_root outputs/preprocessing/scenes \
  --matching_aligned_root outputs/matching/aligned \
  --georeferencing_root outputs/georeferencing \
  --out_dir outputs/georeferencing_side_by_side \
  --scene_list /path/to/scenes.txt \
  --max_side 900
```

## Parallel processing

Preprocessing, Drive finalization, and georeferencing keep unique per-process
reports and update their merged reports with file locking and atomic replacement.
Parallel processes must still receive disjoint scene sets; the same scene should
not be processed by two processes simultaneously.

## Command reference

Every command-line argument, default value, and stage-specific explanation is
listed in:

- [Preprocessing arguments](docs/preprocessing.md)
- [Matching arguments](docs/matching.md)
- [Georeferencing and preview arguments](docs/georeferencing.md)
- [Patch-classifier workflow and arguments](docs/patch_classifier.md)
- [Temporal-processing workflow and arguments](docs/temporal.md)
