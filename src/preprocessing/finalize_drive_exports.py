from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from preprocessing.geo import (
    all_scene_reference_config,
    gdal_warp_reference_candidates,
    gdal_warp_to_template_grid,
)
from preprocessing.report import RunReport


def load_multi_reference_config(path: str | None) -> dict[str, dict]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("multi_reference_config must be a JSON object keyed by scene name")
    return payload


def load_scene_names(scenes_root: Path, scene_list: str | None) -> list[str]:
    if not scene_list:
        return sorted(path.name for path in scenes_root.iterdir() if path.is_dir())
    names = []
    for line in Path(scene_list).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            names.append(Path(value.rstrip("/")).name)
    return list(dict.fromkeys(names))


def find_export(scene: str, export_dirs: list[Path]) -> Path:
    exact_names = [f"{scene}_sentinel2_rgb_raw.tif", f"{scene}_sentinel2_rgb_raw.tiff"]
    exact = [directory / name for directory in export_dirs for name in exact_names]
    existing_exact = [path for path in exact if path.exists()]
    if len(existing_exact) == 1:
        return existing_exact[0]
    if len(existing_exact) > 1:
        raise RuntimeError(f"Multiple exact Drive exports found for {scene}: {existing_exact}")

    candidates = []
    for directory in export_dirs:
        candidates.extend(directory.glob(f"{scene}_sentinel2_rgb_raw*.tif"))
        candidates.extend(directory.glob(f"{scene}_sentinel2_rgb_raw*.tiff"))
    candidates = sorted(set(candidates))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one Drive export for {scene}, found {len(candidates)}: {candidates}"
        )
    return candidates[0]


def find_armsat_template(scene_dir: Path) -> Path:
    candidates = sorted(scene_dir.glob("armsat_rgb_utm*.tif"))
    if not candidates:
        native = scene_dir / "armsat_rgb_native.tif"
        if native.exists():
            candidates = [native]
    if not candidates:
        raise FileNotFoundError(f"Missing ArmSat RGB template in {scene_dir}")
    return candidates[0]


def finalize_scene(
    scene: str,
    scenes_root: Path,
    export_dirs: list[Path],
    utm_epsg: int,
    multi_reference: dict | None,
) -> dict:
    scene_dir = scenes_root / scene
    if not scene_dir.is_dir():
        raise FileNotFoundError(f"Missing preprocessing scene directory: {scene_dir}")

    export_path = find_export(scene, export_dirs)
    raw_path = scene_dir / "sentinel2_rgb_raw.tif"
    if export_path.resolve() != raw_path.resolve():
        shutil.copy2(export_path, raw_path)

    armsat_template = find_armsat_template(scene_dir)
    center_path = scene_dir / "sentinel2_rgb_utm.tif"
    gdal_warp_to_template_grid(
        raw_path,
        center_path,
        armsat_template,
        utm_epsg,
        dst_nodata=0,
    )

    reference_candidates = None
    if multi_reference:
        reference_candidates = gdal_warp_reference_candidates(
            in_path=raw_path,
            out_dir=scene_dir,
            template_tif=center_path,
            epsg=utm_epsg,
            config=multi_reference,
            dst_nodata=0,
        )

    summary_path = scene_dir / "scene_prep_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary.update({
        "scene": scene,
        "status": "OK_GEE_DRIVE_EXPORT_FINALIZED",
        "armsat_rgb_utm": str(armsat_template),
        "gee_local_raw": str(raw_path),
        "sentinel_clipped": [str(center_path)],
        "utm_epsg": int(utm_epsg),
        "dstnodata": 0,
        "grid_strategy": "gee_drive_sentinel2_rgb_then_warp_to_native_armsat_grid",
    })
    if multi_reference:
        summary["multi_reference"] = {
            "overlap": float(multi_reference["overlap"]),
            "directions": [
                item["direction"] for item in reference_candidates
                if item["direction"] != "C"
            ],
            "candidates": reference_candidates,
        }
    else:
        summary.pop("multi_reference", None)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "scene": scene,
        "status": "OK",
        "drive_export": str(export_path),
        "raw_path": str(raw_path),
        "center_reference": str(center_path),
        "reference_candidates": len(reference_candidates) if reference_candidates else 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export_dirs", nargs="+", required=True)
    parser.add_argument("--scenes_root", required=True)
    parser.add_argument("--utm_epsg", type=int, required=True)
    parser.add_argument("--scene_list", default=None)
    multi_reference_mode = parser.add_mutually_exclusive_group()
    multi_reference_mode.add_argument("--multi_reference_config", default=None)
    multi_reference_mode.add_argument("--multi_reference_all", action="store_true")
    parser.add_argument("--multi_reference_overlap", type=float, default=None)
    parser.add_argument("--multi_reference_directions", nargs="+", default=None)
    args = parser.parse_args()

    export_dirs = [Path(path) for path in args.export_dirs]
    missing_dirs = [path for path in export_dirs if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"Missing Drive export directories: {missing_dirs}")
    scenes_root = Path(args.scenes_root)
    config = load_multi_reference_config(args.multi_reference_config)
    global_multi_reference = all_scene_reference_config(
        args.multi_reference_all,
        args.multi_reference_overlap,
        args.multi_reference_directions,
    )
    scenes = load_scene_names(scenes_root, args.scene_list)
    reporter = RunReport(scenes_root.parent, "run_report_finalize_drive")

    for scene in scenes:
        try:
            row = finalize_scene(
                scene=scene,
                scenes_root=scenes_root,
                export_dirs=export_dirs,
                utm_epsg=args.utm_epsg,
                multi_reference=global_multi_reference or config.get(scene),
            )
            print(f"[OK] {scene}: references={row['reference_candidates']}", flush=True)
        except Exception as error:
            row = {"scene": scene, "status": "FAILED", "error": repr(error)}
            print(f"[FAIL] {scene}: {error!r}", flush=True)
        reporter.add(row)

    csv_path, json_path = reporter.write()
    print(f"CSV report: {csv_path}")
    print(f"JSON report: {json_path}")


if __name__ == "__main__":
    main()
