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

Clone the repository:

```bash
git clone https://github.com/kristinhambardzumyan/armsat1-processing-pipeline.git
cd armsat1-processing-pipeline
```

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate roma
```

Verify GDAL installation:

```bash
gdalinfo --version
```

Verify PyTorch CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

The Conda environment is recommended because GDAL, Rasterio, GeoPandas, CUDA-enabled PyTorch, and RoMa/romatch can be difficult to install reliably with pip only.

A lightweight `requirements.txt` is also included for reference, but the recommended installation method is:

```bash
conda env create -f environment.yml
```

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
