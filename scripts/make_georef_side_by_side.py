from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.enums import Resampling


def scene_names(path: Path | None, scenes_root: Path) -> list[str]:
    if path is None:
        return sorted(item.name for item in scenes_root.iterdir() if item.is_dir())
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            names.append(Path(value.rstrip("/")).name)
    return list(dict.fromkeys(names))


def preview_shape(width: int, height: int, max_side: int) -> tuple[int, int]:
    scale = min(1.0, float(max_side) / max(width, height))
    return max(1, round(height * scale)), max(1, round(width * scale))


def valid_mask(array: np.ndarray, nodata: float | None) -> np.ndarray:
    valid = np.isfinite(array).all(axis=2)
    if nodata is not None:
        valid &= (array != float(nodata)).all(axis=2)
    valid &= (array > -1e6).all(axis=2)
    valid &= (array < 1e20).all(axis=2)
    return valid


def stretch_rgb(array: np.ndarray, nodata: float | None) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    valid = valid_mask(array, nodata)
    if int(valid.sum()) < 100:
        raise RuntimeError("Too few valid pixels for RGB preview")

    output = np.zeros(array.shape, dtype=np.uint8)
    for channel in range(3):
        band = array[..., channel]
        values = band[valid]
        low, high = np.percentile(values, [2, 98])
        if high <= low:
            low, high = float(values.min()), float(values.max())
        scaled = np.clip((band - low) / (high - low + 1e-8), 0.0, 1.0)
        output[..., channel] = np.round(scaled * 255.0).astype(np.uint8)
    output[~valid] = 0
    return output


def read_reference_rgb(path: Path, max_side: int) -> Image.Image:
    with rasterio.open(path) as dataset:
        if dataset.count < 3:
            raise RuntimeError(f"Reference must have at least three bands: {path}")
        out_height, out_width = preview_shape(dataset.width, dataset.height, max_side)
        array = dataset.read(
            [1, 2, 3],
            out_shape=(3, out_height, out_width),
            resampling=Resampling.average,
        )
        nodata = dataset.nodata
    return Image.fromarray(stretch_rgb(np.moveaxis(array, 0, 2), nodata))


def georef_band(scene_dir: Path, scene: str, band: str) -> Path:
    exact = scene_dir / f"{scene}_{band}_georef_tps.tif"
    if exact.exists():
        return exact
    candidates = sorted(scene_dir.glob(f"*_{band}_georef_tps.tif"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one georeferenced {band} band in {scene_dir}, found {len(candidates)}"
        )
    return candidates[0]


def read_georeferenced_rgb(scene_dir: Path, scene: str, max_side: int) -> Image.Image:
    paths = [georef_band(scene_dir, scene, band) for band in ("R", "G", "B")]
    arrays = []
    expected_grid = None
    nodata = None
    for path in paths:
        with rasterio.open(path) as dataset:
            grid = (dataset.width, dataset.height, dataset.crs, dataset.transform)
            if expected_grid is None:
                expected_grid = grid
                out_height, out_width = preview_shape(dataset.width, dataset.height, max_side)
                nodata = dataset.nodata
            elif grid != expected_grid:
                raise RuntimeError(f"Georeferenced RGB bands do not share one grid: {scene}")
            arrays.append(
                dataset.read(
                    1,
                    out_shape=(out_height, out_width),
                    resampling=Resampling.average,
                )
            )
    return Image.fromarray(stretch_rgb(np.stack(arrays, axis=2), nodata))


def selected_reference(
    scene: str,
    preprocessing_root: Path,
    matching_root: Path,
    georef_root: Path,
) -> tuple[str, Path]:
    metadata_path = georef_root / scene / "georef_metadata.json"
    matches_path = matching_root / scene / "matches_filtered.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected = metadata.get("selected_reference_candidate")
        recorded_path = metadata.get("ref_template")
    else:
        selected = None
        recorded_path = None
    if not selected and matches_path.exists():
        matches = json.loads(matches_path.read_text(encoding="utf-8"))
        selected = matches.get("selected_reference_candidate")
    if not selected:
        raise RuntimeError(f"No selected reference metadata for {scene}")

    direction = str(selected["direction"]).upper()
    candidate_path = Path(recorded_path or selected["reference_path"])
    if not candidate_path.exists():
        candidate_path = preprocessing_root / scene / candidate_path.name
    if not candidate_path.exists():
        raise FileNotFoundError(f"Selected reference does not exist: {candidate_path}")
    return direction, candidate_path


def labeled_side_by_side(
    reference: Image.Image,
    georeferenced: Image.Image,
    scene: str,
    direction: str,
) -> Image.Image:
    panel_height = max(reference.height, georeferenced.height)
    header_height = 54
    canvas = Image.new(
        "RGB",
        (reference.width + georeferenced.width, panel_height + header_height),
        color=(20, 20, 20),
    )
    canvas.paste(reference, (0, header_height))
    canvas.paste(georeferenced, (reference.width, header_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"Selected Sentinel-2 reference ({direction})", fill="white")
    draw.text((reference.width + 12, 8), "Georeferenced ArmSat RGB", fill="white")
    draw.text((12, 29), scene, fill=(190, 190, 190))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create selected-reference versus georeferenced ArmSat RGB previews."
    )
    parser.add_argument("--preprocessing_scenes_root", required=True)
    parser.add_argument("--matching_aligned_root", required=True)
    parser.add_argument("--georeferencing_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--scene_list", default=None)
    parser.add_argument("--max_side", type=int, default=900)
    args = parser.parse_args()

    preprocessing_root = Path(args.preprocessing_scenes_root)
    matching_root = Path(args.matching_aligned_root)
    georef_root = Path(args.georeferencing_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = scene_names(Path(args.scene_list) if args.scene_list else None, georef_root)
    report = []
    failures = []

    for scene in scenes:
        try:
            direction, reference_path = selected_reference(
                scene, preprocessing_root, matching_root, georef_root
            )
            reference = read_reference_rgb(reference_path, args.max_side)
            georeferenced = read_georeferenced_rgb(georef_root / scene, scene, args.max_side)
            preview = labeled_side_by_side(reference, georeferenced, scene, direction)
            output_path = out_dir / f"{scene}__selected_{direction}__reference_vs_georeferenced.jpg"
            preview.save(output_path, quality=95)
            report.append(
                {
                    "scene": scene,
                    "status": "PASS",
                    "selected_reference": direction,
                    "reference_path": str(reference_path),
                    "preview_path": str(output_path),
                }
            )
            print(f"[OK] {scene}: selected_reference={direction} -> {output_path}")
        except Exception as error:
            failures.append(f"{scene}: {error!r}")
            report.append({"scene": scene, "status": "FAIL", "error": repr(error)})
            print(f"[FAIL] {scene}: {error!r}")

    report_path = out_dir / "side_by_side_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")
    if failures:
        raise SystemExit(f"Side-by-side preview failed for {len(failures)} scene(s)")


if __name__ == "__main__":
    main()
