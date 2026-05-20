from __future__ import annotations
import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import List, Tuple
import numpy as np
import rasterio
from pyproj import Transformer

GCP = Tuple[float, float, float, float, float]

def run(cmd: list[str], check: bool = True, quiet: bool = False) -> str:
    if not quiet:
        print("\n$ " + " ".join(map(str, cmd)), flush=True)
    p = subprocess.run(
        list(map(str, cmd)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not quiet:
        print(p.stdout, flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stdout}"
        )
    return p.stdout

def read_matches(matches_json: Path):
    data = json.loads(matches_json.read_text(encoding="utf-8"))
    pts_mov = np.asarray(data["pts_mov_on_ref"], dtype=np.float64)
    pts_ref = np.asarray(data["pts_ref"], dtype=np.float64)
    conf = np.asarray(data.get("confidence", []), dtype=np.float64)
    if conf.size == 0:
        conf = np.ones(len(pts_mov), dtype=np.float64)
    if len(pts_mov) != len(pts_ref) or len(pts_mov) != len(conf):
        raise RuntimeError(f"Inconsistent match arrays in {matches_json}")
    return pts_mov, pts_ref, conf, data

def px_to_map(ds: rasterio.io.DatasetReader, x: float, y: float) -> Tuple[float, float]:
    X, Y = ds.transform * (x, y)
    return float(X), float(Y)

def map_to_px(ds: rasterio.io.DatasetReader, X: float, Y: float) -> Tuple[float, float]:
    x, y = (~ds.transform) * (X, Y)
    return float(x), float(y)

def crs_transform(src_crs, dst_crs, xs, ys):
    tr = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    return tr.transform(xs, ys)

def dedup_gcps(gcps: List[GCP], src_round_px: float = 0.25) -> List[GCP]:
    best: dict[tuple[int, int], GCP] = {}
    for sx, sy, X, Y, c in gcps:
        key = (
            round(sx / src_round_px),
            round(sy / src_round_px),
        )
        old = best.get(key)
        if old is None or c > old[-1]:
            best[key] = (sx, sy, X, Y, c)
    return list(best.values())


def select_gcps_spatial_balanced(
    pts_mov: np.ndarray,
    pts_ref: np.ndarray,
    conf: np.ndarray,
    *,
    image_shape: tuple[int, int],
    max_gcps: int = 1200,
    grid_size: int = 16,
    min_conf: float = 0.0,
):
    keep = conf >= float(min_conf)
    pts_mov = pts_mov[keep]
    pts_ref = pts_ref[keep]
    conf = conf[keep]
    if len(conf) == 0:
        return pts_mov, pts_ref, conf
    if max_gcps <= 0 or len(conf) <= max_gcps:
        return pts_mov, pts_ref, conf
    h, w = image_shape
    order = np.argsort(-conf)
    pts_mov = pts_mov[order]
    pts_ref = pts_ref[order]
    conf = conf[order]
    cell_w = w / grid_size
    cell_h = h / grid_size
    cells: dict[tuple[int, int], list[int]] = {}
    for i, p in enumerate(pts_ref):
        x, y = float(p[0]), float(p[1])

        if not (0 <= x < w and 0 <= y < h):
            continue

        cx = min(grid_size - 1, int(x / cell_w))
        cy = min(grid_size - 1, int(y / cell_h))
        cells.setdefault((cy, cx), []).append(i)
    selected: list[int] = []

    while len(selected) < max_gcps:
        added = False

        for key in sorted(cells.keys()):
            if cells[key]:
                selected.append(cells[key].pop(0))
                added = True

                if len(selected) >= max_gcps:
                    break

        if not added:
            break

    idx = np.asarray(selected, dtype=np.int64)
    return pts_mov[idx], pts_ref[idx], conf[idx]

def build_gcps_for_original_band(
    orig_band: Path,
    ref_template: Path,
    pts_mov_on_ref: np.ndarray,
    pts_ref: np.ndarray,
    conf: np.ndarray,
) -> List[GCP]:
    with rasterio.open(ref_template) as ref, rasterio.open(orig_band) as src:
        ref_crs = ref.crs
        src_crs = src.crs
        if ref_crs is None:
            raise RuntimeError(f"Reference template has no CRS: {ref_template}")
        if src_crs is None:
            raise RuntimeError(f"Source band has no CRS: {orig_band}")
        gcps: List[GCP] = []
        for (mx, my), (rx, ry), c in zip(pts_mov_on_ref, pts_ref, conf):
            src_X_ref, src_Y_ref = px_to_map(ref, float(mx), float(my))

            if src_crs != ref_crs:
                src_X_arr, src_Y_arr = crs_transform(
                    ref_crs,
                    src_crs,
                    [src_X_ref],
                    [src_Y_ref],
                )
                src_X = float(src_X_arr[0])
                src_Y = float(src_Y_arr[0])
            else:
                src_X, src_Y = src_X_ref, src_Y_ref

            sx, sy = map_to_px(src, src_X, src_Y)
            dst_X, dst_Y = px_to_map(ref, float(rx), float(ry))
            gcps.append((float(sx), float(sy), float(dst_X), float(dst_Y), float(c)))

        return gcps

def attach_gcps_vrt(src_tif: Path, vrt_out: Path, gcps: List[GCP], quiet: bool = False) -> None:
    cmd = ["gdal_translate", "-of", "VRT"]
    for sx, sy, X, Y, _ in gcps:
        cmd += ["-gcp", sx, sy, X, Y]
    cmd += [src_tif, vrt_out]
    run(cmd, quiet=quiet)

def warp_with_tps(
    src_with_gcps: Path,
    out_tif: Path,
    out_crs,
    te: Tuple[float, float, float, float],
    xres: float,
    yres: float,
    *,
    src_nodata=None,
    dst_nodata=None,
    resampling: str = "bilinear",
    quiet: bool = False,
) -> None:
    xmin, ymin, xmax, ymax = te
    cmd = [
        "gdalwarp",
        "-overwrite",
        "-tps",
        "-t_srs", str(out_crs),
        "-tr", xres, yres,
        "-r", resampling,
    ]
    if src_nodata is not None:
        cmd += ["-srcnodata", str(src_nodata)]
    if dst_nodata is not None:
        cmd += ["-dstnodata", str(dst_nodata)]
    cmd += [
        "-of", "GTiff",
        # "-co", "COMPRESS=DEFLATE",
        "-co", "TILED=YES",
        "-co", "BIGTIFF=IF_SAFER",
        src_with_gcps,
        out_tif,
    ]
    run(cmd, quiet=quiet)

def read_src_nodata(src_tif: Path):
    with rasterio.open(src_tif) as src:
        return src.nodata

def find_original_bands(scene_dir: Path) -> List[Path]:
    out: List[Path] = []
    for suf in [
        "_R.tif", "_G.tif", "_B.tif", "_N.tif",
        "_R.tiff", "_G.tiff", "_B.tiff", "_N.tiff",
    ]:
        out.extend(scene_dir.glob(f"*{suf}"))
    return sorted(out)

def resolve_aligned_root(path: Path) -> Path:
    if (path / "aligned").is_dir():
        return path / "aligned"
    return path

def resolve_scene_list(args, template_scenes_root: Path) -> list[str]:
    if args.scene:
        return [args.scene]

    if args.scene_list:
        return [
            x.strip()
            for x in Path(args.scene_list).read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
    return sorted([p.name for p in template_scenes_root.iterdir() if p.is_dir()])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template_scenes_root", required=True)
    ap.add_argument("--original_scenes_root", required=True)
    ap.add_argument("--aligned_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--scene_list", default=None)
    ap.add_argument("--min_gcps", type=int, default=12)
    ap.add_argument("--max_gcps", type=int, default=1200)
    ap.add_argument("--grid_size", type=int, default=16)
    ap.add_argument("--min_conf_eval", type=float, default=0.0)
    ap.add_argument("--resampling", default="bilinear", choices=["near", "bilinear", "cubic"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    template_scenes_root = Path(args.template_scenes_root)
    original_scenes_root = Path(args.original_scenes_root)
    aligned_root = resolve_aligned_root(Path(args.aligned_root))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not template_scenes_root.exists():
        raise RuntimeError(f"template_scenes_root does not exist: {template_scenes_root}")

    if not original_scenes_root.exists():
        raise RuntimeError(f"original_scenes_root does not exist: {original_scenes_root}")

    if not aligned_root.exists():
        raise RuntimeError(f"aligned_root does not exist: {aligned_root}")

    scenes = resolve_scene_list(args, template_scenes_root)
    report_rows = []
    print(f"[INFO] Scenes to process: {len(scenes)}", flush=True)
    print(f"[INFO] max_gcps: {args.max_gcps}", flush=True)
    print(f"[INFO] grid_size: {args.grid_size}", flush=True)

    for scene in scenes:
        print(f"\n========== {scene} ==========", flush=True)

        template_scene_dir = template_scenes_root / scene
        original_scene_dir = original_scenes_root / scene
        aligned_scene_dir = aligned_root / scene

        row = {
            "scene": scene,
            "status": "unknown",
            "reason": "",
            "matches_total": None,
            "gcps_selected": None,
            "gcps_after_dedup_min": None,
            "bands_found": 0,
            "bands_written": 0,
        }

        try:
            if not template_scene_dir.exists():
                row["status"] = "skipped"
                row["reason"] = "missing_template_scene_dir"
                print(f"[SKIP] {scene}: missing template scene dir")
                report_rows.append(row)
                continue

            if not original_scene_dir.exists():
                row["status"] = "skipped"
                row["reason"] = "missing_original_scene_dir"
                print(f"[SKIP] {scene}: missing original scene dir")
                report_rows.append(row)
                continue

            matches_json = aligned_scene_dir / "matches_filtered.json"

            if not matches_json.exists():
                row["status"] = "skipped"
                row["reason"] = "missing_matches_filtered"
                print(f"[SKIP] {scene}: missing matches_filtered.json")
                report_rows.append(row)
                continue

            ref_template_candidates = sorted(template_scene_dir.glob("armsat_rgb_utm*.tif"))

            if not ref_template_candidates:
                row["status"] = "skipped"
                row["reason"] = "missing_armsat_rgb_utm_template"
                print(f"[SKIP] {scene}: no armsat_rgb_utm*.tif")
                report_rows.append(row)
                continue

            ref_template = ref_template_candidates[0]
            pts_mov, pts_ref, conf, match_data = read_matches(matches_json)
            row["matches_total"] = int(len(conf))

            if len(conf) < args.min_gcps:
                row["status"] = "skipped"
                row["reason"] = f"too_few_matches({len(conf)})"
                print(f"[SKIP] {scene}: too few matches")
                report_rows.append(row)
                continue

            with rasterio.open(ref_template) as ref:
                out_crs = ref.crs

                if out_crs is None:
                    row["status"] = "skipped"
                    row["reason"] = "template_has_no_crs"
                    print(f"[SKIP] {scene}: template has no CRS")
                    report_rows.append(row)
                    continue

                te = (ref.bounds.left, ref.bounds.bottom, ref.bounds.right, ref.bounds.top)
                xres = abs(float(ref.transform.a))
                yres = abs(float(ref.transform.e))
                ref_shape_hw = (ref.height, ref.width)

            pts_mov, pts_ref, conf = select_gcps_spatial_balanced(
                pts_mov,
                pts_ref,
                conf,
                image_shape=ref_shape_hw,
                max_gcps=args.max_gcps,
                grid_size=args.grid_size,
                min_conf=args.min_conf_eval,
            )

            row["gcps_selected"] = int(len(conf))

            if len(conf) < args.min_gcps:
                row["status"] = "skipped"
                row["reason"] = f"too_few_gcps_after_selection({len(conf)})"
                print(f"[SKIP] {scene}: too few GCPs after selection")
                report_rows.append(row)
                continue

            orig_bands = find_original_bands(original_scene_dir)
            row["bands_found"] = int(len(orig_bands))

            if not orig_bands:
                row["status"] = "skipped"
                row["reason"] = "no_original_bands_found"
                print(f"[SKIP] {scene}: no original R/G/B/N bands")
                report_rows.append(row)
                continue

            scene_out = out_dir / scene
            scene_out.mkdir(parents=True, exist_ok=True)
            print(f"[INFO] Template scene dir : {template_scene_dir}")
            print(f"[INFO] Original scene dir : {original_scene_dir}")
            print(f"[INFO] Matches JSON       : {matches_json}")
            print(f"[INFO] Matches total      : {row['matches_total']}")
            print(f"[INFO] GCPs selected      : {row['gcps_selected']}")
            print(f"[INFO] Found bands        : {len(orig_bands)}")

            written = 0
            dedup_counts = []
            for orig in orig_bands:
                print(f"\n[Band] {orig.name}", flush=True)

                gcps = build_gcps_for_original_band(
                    orig_band=orig,
                    ref_template=ref_template,
                    pts_mov_on_ref=pts_mov,
                    pts_ref=pts_ref,
                    conf=conf,
                )

                gcps = dedup_gcps(gcps, src_round_px=0.25)
                dedup_counts.append(len(gcps))

                if len(gcps) < args.min_gcps:
                    print(f"[SKIP] {orig.name}: too few GCPs after dedup ({len(gcps)})")
                    continue

                src_nodata = read_src_nodata(orig)
                dst_nodata = src_nodata

                band_name = orig.stem.split("_")[-1]
                vrt_path = scene_out / f"{scene}_{band_name}_gcps.vrt"
                out_tif = scene_out / f"{scene}_{band_name}_georef_tps.tif"

                attach_gcps_vrt(orig, vrt_path, gcps, quiet=args.quiet)

                warp_with_tps(
                    src_with_gcps=vrt_path,
                    out_tif=out_tif,
                    out_crs=out_crs,
                    te=te,
                    xres=xres,
                    yres=yres,
                    src_nodata=src_nodata,
                    dst_nodata=dst_nodata,
                    resampling=args.resampling,
                    quiet=args.quiet,
                )

                written += 1
                print(f"[OK] Wrote {out_tif}", flush=True)

            row["bands_written"] = int(written)
            row["gcps_after_dedup_min"] = int(min(dedup_counts)) if dedup_counts else None

            metadata = {
                "scene": scene,
                "template_scene_dir": str(template_scene_dir),
                "original_scene_dir": str(original_scene_dir),
                "matches_json": str(matches_json),
                "ref_template": str(ref_template),
                "output_crs": str(out_crs),
                "output_bounds": list(te),
                "output_resolution": [xres, yres],
                "matches_total": row["matches_total"],
                "gcps_selected": row["gcps_selected"],
                "gcps_after_dedup_min": row["gcps_after_dedup_min"],
                "max_gcps": args.max_gcps,
                "grid_size": args.grid_size,
                "selection_policy": "spatial_balanced_round_robin_by_confidence",
                "resampling": args.resampling,
                "matching_residual_median_px": match_data.get("residual_median_px"),
                "matching_residual_p90_px": match_data.get("residual_p90_px"),
                "matching_final_inliers": match_data.get("n_matches_inliers"),
            }

            (scene_out / "georef_metadata.json").write_text(
                json.dumps(metadata, indent=2),
                encoding="utf-8",
            )

            if written > 0:
                row["status"] = "success"
                row["reason"] = ""
            else:
                row["status"] = "failed"
                row["reason"] = "no_bands_written"

        except Exception as e:
            row["status"] = "failed"
            row["reason"] = repr(e)
            print(f"[FAIL] {scene}: {repr(e)}", flush=True)

        report_rows.append(row)

    report_path = out_dir / "georef_report.csv"

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n[OUTPUT]")
    print(f"Report: {report_path}")
    print(f"Successful scenes: {sum(r['status'] == 'success' for r in report_rows)}")
    print(f"Failed/skipped scenes: {sum(r['status'] != 'success' for r in report_rows)}")

if __name__ == "__main__":
    main()