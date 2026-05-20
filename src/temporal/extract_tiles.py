from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import reproject, Resampling
from shapely.geometry import box

BANDS = ["R", "G", "B", "N"]

def candidate_bounds(row, col, transform, tile_size_px):
    return array_bounds(
        tile_size_px,
        tile_size_px,
        transform * rasterio.Affine.translation(col, row),
    )

def inside_core(bounds, core_bounds):
    w, s, e, n = bounds
    cw, cs, ce, cn = core_bounds
    return w >= cw and e <= ce and s >= cs and n <= cn

def generate_candidates(date_count, transform, core_bounds, args):
    h, w = date_count.shape
    candidates = []

    for row in range(0, h - args.tile_size_px + 1, args.candidate_step_px):
        for col in range(0, w - args.tile_size_px + 1, args.candidate_step_px):
            patch = date_count[row:row + args.tile_size_px, col:col + args.tile_size_px]
            repeated = patch >= args.min_distinct_dates
            repeated_fraction = float(repeated.mean())

            if repeated_fraction < args.min_repeated_fraction:
                continue

            bounds = candidate_bounds(row, col, transform, args.tile_size_px)

            if not inside_core(bounds, core_bounds):
                continue

            candidates.append({
                "row": row,
                "col": col,
                "bounds": bounds,
                "repeated_fraction": repeated_fraction,
                "mean_date_count": float(patch[repeated].mean()) if repeated.any() else 0.0,
                "max_date_count": int(patch.max()),
            })

    return candidates

def select_nonoverlap(candidates, shape, tile_size_px):
    candidates = sorted(
        candidates,
        key=lambda c: (c["repeated_fraction"], c["mean_date_count"], c["max_date_count"]),
        reverse=True,
    )

    occupied = np.zeros(shape, dtype=np.uint8)
    selected = []

    for c in candidates:
        r, col = c["row"], c["col"]

        if occupied[r:r + tile_size_px, col:col + tile_size_px].any():
            continue

        selected.append(c)
        occupied[r:r + tile_size_px, col:col + tile_size_px] = 1

    return selected

def read_band_to_tile(path, bounds, args):
    west, south, east, north = bounds
    dst_transform = from_origin(west, north, args.res_m, args.res_m)

    dst = np.full(
        (args.tile_size_px, args.tile_size_px),
        args.output_nodata,
        dtype=np.float32,
    )

    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=args.target_crs,
            dst_nodata=args.output_nodata,
            resampling=Resampling.bilinear,
        )

        scale = src.scales[0] if src.scales and src.scales[0] is not None else 1.0
        offset = src.offsets[0] if src.offsets and src.offsets[0] is not None else 0.0

    valid = (dst != args.output_nodata) & np.isfinite(dst)

    if args.apply_scale_offset:
        dst[valid] = dst[valid] * scale + offset

    return dst, valid, dst_transform

def mosaic_day(day_df, bounds, args):
    arrays = {
        b: np.full((args.tile_size_px, args.tile_size_px), args.output_nodata, dtype=np.float32)
        for b in BANDS
    }

    valid_any = np.zeros((args.tile_size_px, args.tile_size_px), dtype=bool)
    used_scenes = []
    dst_transform = None

    for _, row in day_df.iterrows():
        scene_valid = None

        for band in BANDS:
            arr, valid, dst_transform = read_band_to_tile(Path(row[f"path_{band}"]), bounds, args)

            fill = valid & (arrays[band] == args.output_nodata)
            arrays[band][fill] = arr[fill]

            if band == "R":
                scene_valid = valid

        if scene_valid is not None and scene_valid.any():
            valid_any |= scene_valid
            used_scenes.append(row["scene_id"])

    stack = np.stack([arrays[b] for b in BANDS], axis=0)
    return stack, valid_any, dst_transform, used_scenes

def extract_tiles_for_block(block_id, block_scenes, date_count, transform, core_bounds, out_dir, args):
    tile_dir_root = out_dir / "tiles"
    meta_dir = out_dir / "metadata"

    tile_dir_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    candidates = generate_candidates(date_count, transform, core_bounds, args)
    selected = select_nonoverlap(candidates, date_count.shape, args.tile_size_px)

    tile_rows = []
    obs_rows = []
    footprint_rows = []

    tile_idx = 0

    for cand in selected:
        by_day = []
        bounds = cand["bounds"]

        for day, day_df in block_scenes.groupby("acq_day"):
            stack, valid, dst_transform, used_scenes = mosaic_day(day_df, bounds, args)
            vf = float(valid.mean())

            if vf >= args.valid_fraction_threshold:
                by_day.append((day, stack, dst_transform, used_scenes, vf))

        if len(by_day) < args.min_distinct_dates:
            continue

        tile_idx += 1
        tile_id = f"{block_id}_TS_{tile_idx:06d}"
        obs_dir = tile_dir_root / tile_id / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)

        west, south, east, north = bounds

        tile_rows.append({
            "block_id": block_id,
            "tile_id": tile_id,
            "west": west,
            "south": south,
            "east": east,
            "north": north,
            "distinct_date_count": len(by_day),
            "dates": " | ".join([x[0] for x in by_day]),
            "repeated_fraction": cand["repeated_fraction"],
            "mean_date_count": cand["mean_date_count"],
            "max_date_count": cand["max_date_count"],
        })

        footprint_rows.append({
            "block_id": block_id,
            "tile_id": tile_id,
            "distinct_date_count": len(by_day),
            "dates": " | ".join([x[0] for x in by_day]),
            "geometry": box(west, south, east, north),
        })

        for day, stack, dst_transform, used_scenes, vf in by_day:
            out_tif = obs_dir / f"{tile_id}_{day}_ARMSAT1_RGBN_{args.tile_size_px}_{args.res_m:g}m.tif"

            profile = {
                "driver": "GTiff",
                "height": args.tile_size_px,
                "width": args.tile_size_px,
                "count": 4,
                "dtype": "float32",
                "crs": args.target_crs,
                "transform": dst_transform,
                "nodata": args.output_nodata,
                "compress": "deflate",
                "interleave": "band",
            }

            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(stack)
                dst.set_band_description(1, "Red")
                dst.set_band_description(2, "Green")
                dst.set_band_description(3, "Blue")
                dst.set_band_description(4, "NIR")

            obs_rows.append({
                "block_id": block_id,
                "tile_id": tile_id,
                "acq_day": day,
                "valid_fraction": vf,
                "source_scene_ids": " | ".join(used_scenes),
                "observation_path": str(out_tif),
            })

    pd.DataFrame(tile_rows).to_csv(meta_dir / f"{block_id}_tiles.csv", index=False)
    pd.DataFrame(obs_rows).to_csv(meta_dir / f"{block_id}_observations.csv", index=False)

    if footprint_rows:
        gpd.GeoDataFrame(footprint_rows, geometry="geometry", crs=args.target_crs).to_file(
            meta_dir / f"{block_id}_tile_footprints.gpkg",
            driver="GPKG",
        )

    return len(tile_rows), len(obs_rows)