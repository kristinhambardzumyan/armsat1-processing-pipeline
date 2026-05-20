from pathlib import Path
import argparse
import json
import pandas as pd
from src.temporal.scene_index import build_scene_index
from src.temporal.blocks import build_blocks
from src.temporal.coverage import build_date_count, save_count_and_mask
from src.temporal.extract_tiles import extract_tiles_for_block

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--out_root", required=True)
    p.add_argument("--target_crs", default="EPSG:32638")
    p.add_argument("--block_size_m", type=float, default=20000)
    p.add_argument("--block_margin_m", type=float, default=2048)
    p.add_argument("--res_m", type=float, default=2.0)
    p.add_argument("--tile_size_px", type=int, default=512)
    p.add_argument("--candidate_step_px", type=int, default=128)
    p.add_argument("--min_distinct_dates", type=int, default=2)
    p.add_argument("--min_repeated_fraction", type=float, default=0.99)
    p.add_argument("--valid_fraction_threshold", type=float, default=0.99)
    p.add_argument("--output_nodata", type=float, default=-9999.0)
    p.add_argument("--apply_scale_offset", action="store_true")
    p.add_argument("--only_block_id", default=None)
    return p.parse_args()

def main():
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print("[STEP 1] Build scene index")
    scene_gdf = build_scene_index(args.roots, args.target_crs)
    scene_gdf.to_file(out_root / "scene_footprints.gpkg", driver="GPKG")
    scene_gdf.drop(columns=["geometry"]).to_csv(out_root / "scene_index.csv", index=False)
    print(f"[INFO] Scenes found: {len(scene_gdf)}")

    print("[STEP 2] Build fixed blocks")
    blocks = build_blocks(scene_gdf, args.block_size_m, args.block_margin_m, args.target_crs)
    blocks.to_file(out_root / "processing_blocks.gpkg", driver="GPKG")
    if args.only_block_id:
        blocks = blocks[blocks["block_id"] == args.only_block_id].copy()
    results = []

    print("[STEP 3] Process blocks")
    for _, block in blocks.iterrows():
        block_id = block["block_id"]
        expanded_geom = block["expanded_geometry"]
        block_scenes = scene_gdf[scene_gdf.geometry.intersects(expanded_geom)].copy()
        dates = sorted(block_scenes["acq_day"].unique())

        if len(dates) < args.min_distinct_dates:
            continue

        print(f"\n[INFO] {block_id}: scenes={len(block_scenes)}, dates={len(dates)}")

        block_dir = out_root / block_id
        coverage_dir = block_dir / "coverage"
        coverage_dir.mkdir(parents=True, exist_ok=True)
        expanded_bounds = (
            block["expanded_minx"],
            block["expanded_miny"],
            block["expanded_maxx"],
            block["expanded_maxy"],
        )

        core_bounds = (
            block["core_minx"],
            block["core_miny"],
            block["core_maxx"],
            block["core_maxy"],
        )

        date_count, transform = build_date_count(
            block_scenes,
            expanded_bounds,
            args.target_crs,
            args.res_m,
        )

        count_path, mask_path = save_count_and_mask(
            date_count,
            transform,
            coverage_dir,
            block_id,
            args.target_crs,
            args.min_distinct_dates,
        )

        n_tiles, n_obs = extract_tiles_for_block(
            block_id,
            block_scenes,
            date_count,
            transform,
            core_bounds,
            block_dir,
            args,
        )

        results.append({
            "block_id": block_id,
            "n_scenes": len(block_scenes),
            "n_dates": len(dates),
            "selected_tiles": n_tiles,
            "observations": n_obs,
            "count_raster": str(count_path),
            "mask_raster": str(mask_path),
        })

    pd.DataFrame(results).to_csv(out_root / "block_processing_summary.csv", index=False)
    with open(out_root / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n[DONE]")
    print(f"Saved to: {out_root}")

if __name__ == "__main__":
    main()
