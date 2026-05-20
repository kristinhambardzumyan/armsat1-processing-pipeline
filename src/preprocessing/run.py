import json
import argparse
from pathlib import Path
from tqdm import tqdm
import rasterio
import logging
from datetime import datetime

from preprocessing.geo import (
    run, is_band_tif, scene_key, pick_driver, parse_armsat_date,
    bbox_wgs84, expand_bbox, time_window,
    gdal_reproject, gdal_warp_to_template_grid,
    build_armsat_rgb
)

from preprocessing.s2_gee import (
    download_mosaic_rgb_to_local,
    export_mosaic_rgb_to_drive,
    init_gee,
)
from preprocessing.report import RunReport

def process_one_scene(
    scene: str,
    scene_paths: list[Path],
    driver: Path,
    out_scene: Path,
    utm_epsg: int,
    buffer_deg: float,
    window_days: int,
    resolution: float,
    gee_project: str | None,
    max_cloud_pct: float,
    gee_composite: str,
    gee_download_mode: str,
    drive_folder: str,
) -> dict:
    out_scene.mkdir(parents=True, exist_ok=True)

    armsat_dt = parse_armsat_date(scene)
    if not armsat_dt:
        raise RuntimeError(f"Cannot parse date from scene: {scene}")

    # Build ArmSat RGB native
    armsat_rgb_native = out_scene / "armsat_rgb_native.tif"
    armsat_rgb_native, rgb_info = build_armsat_rgb(scene_paths, armsat_rgb_native)

    # bbox + time window
    b = bbox_wgs84(driver)
    b_exp = expand_bbox(b, buffer_deg)
    start, end = time_window(armsat_dt, window_days)

    # ArmSat RGB -> UTM
    with rasterio.open(armsat_rgb_native) as ds:
        src_epsg = ds.crs.to_epsg() if ds.crs else None

    if src_epsg == utm_epsg:
        print(f"ArmSat RGB is already EPSG:{utm_epsg}; skipping reprojection.")
        armsat_rgb_utm = armsat_rgb_native
    else:
        armsat_rgb_utm = out_scene / "armsat_rgb_utm.tif"
        gdal_reproject(armsat_rgb_native, armsat_rgb_utm, utm_epsg)
    with rasterio.open(armsat_rgb_utm) as ds:
        native_xres = abs(ds.transform.a)
        native_yres = abs(ds.transform.e)

    # Sentinel-2 RGB mosaic from GEE: local direct download or Drive export
    gee_tif = out_scene / "sentinel2_rgb_raw.tif"

    if gee_download_mode == "local":
        download_mosaic_rgb_to_local(
            scene=scene,
            bbox_wgs84=b_exp,
            time_interval=(start, end),
            out_path=gee_tif,
            resolution=resolution,
            max_cloud_pct=max_cloud_pct,
            composite=gee_composite,
        )

    elif gee_download_mode == "drive":
        task_id = export_mosaic_rgb_to_drive(
            scene=scene,
            bbox_wgs84=b_exp,
            time_interval=(start, end),
            resolution=resolution,
            max_cloud_pct=max_cloud_pct,
            composite=gee_composite,
            drive_folder=drive_folder,
        )

        return {
            "scene": scene,
            "status": "STARTED_GEE_DRIVE_EXPORT",
            "armsat_date": armsat_dt.strftime("%Y-%m-%d"),
            "armsat_driver": str(driver),
            "armsat_rgb_native": str(armsat_rgb_native),
            "armsat_rgb_utm": str(armsat_rgb_utm),
            "gee_drive_task_id": task_id,
            "drive_folder": drive_folder,
            "expected_drive_file": f"{scene}_sentinel2_rgb_raw.tif",
            "scene_dir": str(out_scene),
        }

    # Warp Sentinel-2 RGB to exact ArmSat UTM grid
    out_clip = out_scene / "sentinel2_rgb_utm.tif"
    gdal_warp_to_template_grid(
        gee_tif,
        out_clip,
        armsat_rgb_utm,
        utm_epsg,
        dst_nodata=0
    )

    summary = {
        "scene": scene,
        "status": "OK_GEE_DOWNLOADED_AND_WARPED",
        "armsat_driver": str(driver),
        "armsat_date": armsat_dt.strftime("%Y-%m-%d"),
        "search_window": {
            "start": start,
            "end": end,
        },
        "bbox_wgs84": {
            "minLon": b[0],
            "minLat": b[1],
            "maxLon": b[2],
            "maxLat": b[3],
        },
        "bbox_wgs84_expanded": {
            "minLon": b_exp[0],
            "minLat": b_exp[1],
            "maxLon": b_exp[2],
            "maxLat": b_exp[3],
        },
        "utm_epsg": utm_epsg,
        "armsat_rgb_native": str(armsat_rgb_native),
        "armsat_rgb_bands": rgb_info["armsat_rgb_bands"],
        "armsat_rgb_utm": str(armsat_rgb_utm),
        "armsat_native_resolution_m": {
            "xres": native_xres,
            "yres": native_yres,
        },
        "gee_local_raw": str(gee_tif),
        "sentinel_clipped": [str(out_clip)],
        "gee_project": gee_project,
        "ref_band_for_matching": "RGB(B04,B03,B02)",
        "gee_composite": gee_composite,
        "max_cloud_pct": max_cloud_pct,
        "resolution": resolution,
        "dstnodata": 0,
        "grid_strategy": "gee_sentinel2_rgb_10m_direct_download_then_warp_to_native_armsat_grid",
    }

    (out_scene / "scene_prep_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--utm_epsg", type=int, default=32638)
    ap.add_argument("--buffer_deg", type=float, default=0.04)
    ap.add_argument("--window_days", type=int, default=60)
    ap.add_argument("--resolution", type=float, default=10.0)
    ap.add_argument("--prefer_band", choices=["R", "G", "B", "N"], default="R")

    # GEE arguments
    ap.add_argument("--gee_project", default=None)
    ap.add_argument("--max_cloud_pct", type=float, default=100.0)
    ap.add_argument(
        "--gee_composite",
        choices=["least_cloudy_mosaic"],
        default="least_cloudy_mosaic",
    )
    ap.add_argument(
        "--gee_download_mode",
        choices=["local", "drive"],
        default="local",
    )

    ap.add_argument(
        "--drive_folder",
        default="armsat_sentinel_exports",
    )

    # Batching arguments
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--end_idx", type=int, default=None)
    ap.add_argument(
        "--scene_list",
        default=None,
        help="Optional txt file with one scene name per line. If provided, only these scenes are processed.",
    )
    args = ap.parse_args()

    run(["gdalwarp", "--version"], check=True)

    init_gee(project=args.gee_project)

    in_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"preprocessing_{log_ts}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting preprocessing pipeline")
    scenes_root = out_dir / "scenes"
    scenes_root.mkdir(exist_ok=True)

    tifs = [p for p in in_dir.rglob("*.tif") if is_band_tif(p)] + \
           [p for p in in_dir.rglob("*.tiff") if is_band_tif(p)]

    if not tifs:
        raise RuntimeError("No ARMSAT band tifs found (_R/_G/_B/_N). 1QK is ignored.")

    groups = {}
    for p in tifs:
        groups.setdefault(scene_key(p), []).append(p)

    reporter = RunReport(out_dir=out_dir, name_prefix=f"run_report_preprocessing_{args.gee_download_mode}")

    all_scenes = sorted(groups)

    if args.scene_list:
        scene_list_path = Path(args.scene_list)
        wanted_scenes = [
            line.strip()
            for line in scene_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        missing = [s for s in wanted_scenes if s not in groups]
        if missing:
            raise RuntimeError(
                f"{len(missing)} scenes from scene_list were not found in input_dir. "
                f"First missing scenes: {missing[:5]}"
            )

        batch_scenes = wanted_scenes
    else:
        batch_scenes = all_scenes[args.start_idx:args.end_idx]
    logger.info(f"Total scenes: {len(all_scenes)}")
    logger.info(f"Running batch: start_idx={args.start_idx}, end_idx={args.end_idx}")
    logger.info(f"Scenes in this batch: {len(batch_scenes)}")

    for scene in tqdm(batch_scenes, desc="Downloading GEE mosaics"):
        out_scene = scenes_root / scene
        try:
            scene_paths = groups[scene]
            driver = pick_driver(scene_paths, prefer=args.prefer_band)

            summary = process_one_scene(
                scene=scene,
                scene_paths=scene_paths,
                driver=driver,
                out_scene=out_scene,
                utm_epsg=args.utm_epsg,
                buffer_deg=args.buffer_deg,
                window_days=args.window_days,
                resolution=args.resolution,
                gee_project=args.gee_project,
                max_cloud_pct=args.max_cloud_pct,
                gee_composite=args.gee_composite,
                gee_download_mode=args.gee_download_mode,
                drive_folder=args.drive_folder,
            )

            reporter.add({
                "scene": scene,
                "status": summary.get("status"),
                "armsat_date": summary.get("armsat_date"),
                "armsat_driver": summary.get("armsat_driver"),
                "armsat_rgb_native": summary.get("armsat_rgb_native"),
                "armsat_rgb_utm": summary.get("armsat_rgb_utm"),
                "gee_local_raw": summary.get("gee_local_raw"),
                "ref_clip": summary.get("sentinel_clipped", [None])[0],
                "gee_drive_task_id": summary.get("gee_drive_task_id"),
                "drive_folder": summary.get("drive_folder"),
                "expected_drive_file": summary.get("expected_drive_file"),
                "scene_dir": str(out_scene),
            })

        except Exception as e:
            reporter.add({
                "scene": scene,
                "status": "FAILED",
                "error": str(e),
                "scene_dir": str(out_scene),
            })
            logger.exception(f"FAILED scene: {scene}")

    csv_path, json_path = reporter.write()

    logger.info("GLOBAL REPORT")
    logger.info(f"CSV: {csv_path}")
    logger.info(f"JSON: {json_path}")
    logger.info("GEE images downloaded locally and warped to ArmSat grid.")

if __name__ == "__main__":
    main()