from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import rasterio

from preprocessing.geo import (
    REFERENCE_DIRECTION_OFFSETS,
    all_scene_reference_config,
    reference_candidate_directions,
)


def load_scene_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            names.append(Path(value.rstrip("/")).name)
    return list(dict.fromkeys(names))


def close_enough(actual: float, expected: float, tolerance: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=tolerance)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate multi-reference raster grids and previews without reading scene pixels."
    )
    parser.add_argument("--scenes_root", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config")
    mode.add_argument("--multi_reference_all", action="store_true")
    parser.add_argument("--multi_reference_overlap", type=float, default=None)
    parser.add_argument("--multi_reference_directions", nargs="+", default=None)
    parser.add_argument("--scene_list", default=None)
    parser.add_argument("--previews_dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    scenes_root = Path(args.scenes_root)
    previews_dir = Path(args.previews_dir)
    report_path = Path(args.report)
    config = {}
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("--config must contain a JSON object keyed by scene name")
    global_settings = all_scene_reference_config(
        args.multi_reference_all,
        args.multi_reference_overlap,
        args.multi_reference_directions,
    )
    if args.scene_list:
        scenes = load_scene_names(Path(args.scene_list))
    elif global_settings:
        scenes = sorted(path.name for path in scenes_root.iterdir() if path.is_dir())
    else:
        scenes = list(config)
    results = []
    errors = []

    for scene in scenes:
        scene_errors = []
        scene_dir = scenes_root / scene
        settings = global_settings or config.get(scene)
        if not settings:
            scene_errors.append("scene is missing from multi-reference config")
            expected_directions = []
        else:
            expected_directions = reference_candidate_directions(settings)

        summary_path = scene_dir / "scene_prep_summary.json"
        if not summary_path.exists():
            scene_errors.append(f"missing {summary_path.name}")
            candidates = []
        else:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            candidates = summary.get("multi_reference", {}).get("candidates", [])

        actual_directions = [str(item.get("direction", "")).upper() for item in candidates]
        if actual_directions != expected_directions:
            scene_errors.append(
                f"candidate directions {actual_directions} != expected {expected_directions}"
            )

        center_path = scene_dir / "sentinel2_rgb_utm.tif"
        center_info = None
        if not center_path.exists():
            scene_errors.append("missing center reference sentinel2_rgb_utm.tif")
        else:
            with rasterio.open(center_path) as dataset:
                center_info = {
                    "width": dataset.width,
                    "height": dataset.height,
                    "crs": dataset.crs,
                    "xres": abs(dataset.transform.a),
                    "yres": abs(dataset.transform.e),
                    "bounds": dataset.bounds,
                }

        if center_info and settings:
            overlap = float(settings["overlap"])
            center_bounds = center_info["bounds"]
            width_m = center_bounds.right - center_bounds.left
            height_m = center_bounds.top - center_bounds.bottom
            tolerance = max(center_info["xres"], center_info["yres"], 1.0) * 0.01

            for item in candidates:
                direction = str(item["direction"]).upper()
                candidate_path = Path(item["path"])
                if not candidate_path.is_absolute():
                    candidate_path = scene_dir / candidate_path
                if not candidate_path.exists():
                    scene_errors.append(f"missing reference {direction}: {candidate_path.name}")
                    continue

                with rasterio.open(candidate_path) as dataset:
                    if (dataset.width, dataset.height) != (
                        center_info["width"], center_info["height"]
                    ):
                        scene_errors.append(f"{direction}: raster dimensions differ from center")
                    if dataset.crs != center_info["crs"]:
                        scene_errors.append(f"{direction}: CRS differs from center")
                    if not close_enough(abs(dataset.transform.a), center_info["xres"], tolerance):
                        scene_errors.append(f"{direction}: x resolution differs from center")
                    if not close_enough(abs(dataset.transform.e), center_info["yres"], tolerance):
                        scene_errors.append(f"{direction}: y resolution differs from center")

                    dx, dy = REFERENCE_DIRECTION_OFFSETS.get(direction, (0, 0))
                    expected_left = center_bounds.left + dx * width_m * (1.0 - overlap)
                    expected_bottom = center_bounds.bottom + dy * height_m * (1.0 - overlap)
                    if not close_enough(dataset.bounds.left, expected_left, tolerance):
                        scene_errors.append(f"{direction}: unexpected horizontal crop shift")
                    if not close_enough(dataset.bounds.bottom, expected_bottom, tolerance):
                        scene_errors.append(f"{direction}: unexpected vertical crop shift")

                suffix = "" if direction == "C" else f"__ref_{direction}"
                preview_path = previews_dir / f"{scene}{suffix}.jpg"
                if not preview_path.exists():
                    scene_errors.append(f"missing preview {preview_path.name}")

        status = "PASS" if not scene_errors else "FAIL"
        results.append(
            {
                "scene": scene,
                "status": status,
                "expected_directions": expected_directions,
                "actual_directions": actual_directions,
                "errors": scene_errors,
            }
        )
        errors.extend(f"{scene}: {error}" for error in scene_errors)
        print(f"[{status}] {scene}: references={actual_directions}", flush=True)
        for error in scene_errors:
            print(f"  - {error}", flush=True)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Validation report: {report_path}", flush=True)
    if errors:
        raise SystemExit(f"Multi-reference preprocessing validation failed with {len(errors)} error(s)")


if __name__ == "__main__":
    main()
