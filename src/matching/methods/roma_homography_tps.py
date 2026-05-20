from __future__ import annotations
import argparse
import gc
import json
import os
import pickle
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from tqdm import tqdm

from matching.utils.io_grid import (
    load_armsat_and_buffered_sentinel_for_matching_rgb as load_armsat_and_buffered_sentinel_for_matching,
)
from matching.models.roma import create_roma_matcher
from matching.models.base import RawMatchSet
from matching.utils.matching_helpers import (
    add_tps_args,
    filter_kwargs_from_args,
    find_scene_inputs,
    fit_coarse_prediction_model,
    generate_tiles,
    is_bad_geometry,
    merge_duplicate_matches_global,
    minimal_tile_filter,
    predict_points_with_coarse_model,
    predict_ref_window_from_coarse_model,
    save_matches_json,
    save_tile_diagnostics,
    save_raw_matches_json,
)

def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

def log_cuda_mem(tag: str) -> None:
    if not torch.cuda.is_available():
        print(f"[CUDA] {tag} | CUDA not available", flush=True)
        return

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    max_reserved = torch.cuda.max_memory_reserved() / 1024**3
    print(
        f"[CUDA] {tag} | allocated={allocated:.2f} GB | reserved={reserved:.2f} GB | "
        f"max_allocated={max_allocated:.2f} GB | max_reserved={max_reserved:.2f} GB",
        flush=True,
    )

def set_reproducibility(seed: int, *, deterministic_cudnn: bool = True) -> None:
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic_cudnn:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def _as_numpy_float32(x: Any) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)

def raw_matchset_to_cpu(raw: RawMatchSet) -> RawMatchSet:
    return RawMatchSet(
        pts0=_as_numpy_float32(raw.pts0),
        pts1=_as_numpy_float32(raw.pts1),
        conf=_as_numpy_float32(raw.conf),
    )

def add_roma_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--roma_variant", default="outdoor", choices=["outdoor", "indoor"])
    ap.add_argument("--roma_sample_num", type=int, default=12000)
    ap.add_argument("--roma_sample_num_tile", type=int, default=None)
    ap.add_argument("--roma_sample_thresh", type=float, default=None)
    ap.add_argument("--roma_device", default=None)
    ap.add_argument("--roma_w_resized", type=int, default=1120)
    ap.add_argument("--roma_h_resized", type=int, default=1120)
    ap.add_argument("--roma_upsample_w", type=int, default=1120)
    ap.add_argument("--roma_upsample_h", type=int, default=1120)

def make_roma_matcher(args: argparse.Namespace, *, sample_num: int):
    return create_roma_matcher(
        roma_variant=args.roma_variant,
        roma_sample_num=sample_num,
        roma_sample_thresh=args.roma_sample_thresh,
        roma_device=args.roma_device,
        roma_w_resized=args.roma_w_resized,
        roma_h_resized=args.roma_h_resized,
        roma_upsample_res=(
            (args.roma_upsample_h, args.roma_upsample_w)
            if args.roma_upsample_h is not None and args.roma_upsample_w is not None
            else None
        ),
    )

def tile_worker_main(payload_path: str) -> None:
    """Short-lived tile RoMa worker.
    The worker only performs raw RoMa inference for a batch of tiles and returns
    local tile coordinates. The parent process keeps the exact same tile filter,
    coordinate offset, prior-consistency gate, merge and final TPS logic.
    """
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    args = argparse.Namespace(**payload["args"])
    scene = str(payload["scene"])
    scene_dir = Path(payload["scene_dir"])
    result_path = Path(payload["result_path"])
    tile_sample_num = int(payload["tile_sample_num"])
    tile_specs = payload["tiles"]

    set_reproducibility(int(getattr(args, "seed", 42)), deterministic_cudnn=bool(getattr(args, "deterministic_cudnn", True)))

    matcher = None
    gp = None
    results: list[dict[str, Any]] = []

    try:
        inp = find_scene_inputs(scene_dir)
        gp = load_armsat_and_buffered_sentinel_for_matching(
            armsat_path=inp["armsat"],
            s2_path=inp["s2"],
            out_nodata=-9999.0,
        )
        matcher = make_roma_matcher(args, sample_num=tile_sample_num)

        for spec in tile_specs:
            tile_id = int(spec["tile_id"])
            y0, y1, x0, x1 = [int(v) for v in spec["tile_window_yx"]]
            ry0, ry1, rx0, rx1 = [int(v) for v in spec["pred_ref_window_yx"]]
            row = dict(spec["row"])

            mov_tile = ref_tile = mov_valid_tile = ref_valid_tile = None
            raw_tile = None
            try:
                mov_tile = gp.mov_img01[y0:y1, x0:x1]
                mov_valid_tile = gp.mov_valid[y0:y1, x0:x1]
                ref_tile = gp.ref_img01[ry0:ry1, rx0:rx1]
                ref_valid_tile = gp.ref_valid[ry0:ry1, rx0:rx1]

                raw_tile = matcher.infer_raw(
                    mov_tile,
                    ref_tile,
                    mov_valid_tile,
                    ref_valid_tile,
                    sample_num=tile_sample_num,
                )
                raw_tile = raw_matchset_to_cpu(raw_tile)
                row["n_raw_matches"] = int(len(raw_tile.conf))

                if len(raw_tile.conf) == 0:
                    row["status"] = "zero_raw"
                    results.append({"ok": False, "kind": "zero_raw", "row": row})
                    continue

                results.append(
                    {
                        "ok": True,
                        "kind": "raw_local",
                        "row": row,
                        "pts0": raw_tile.pts0.astype(np.float32, copy=False),
                        "pts1": raw_tile.pts1.astype(np.float32, copy=False),
                        "conf": raw_tile.conf.astype(np.float32, copy=False),
                    }
                )

            except Exception as e:
                row["status"] = "worker_error"
                row["error"] = repr(e)
                results.append({"ok": False, "kind": "error", "row": row})
            finally:
                try:
                    del mov_tile, ref_tile, mov_valid_tile, ref_valid_tile, raw_tile
                except Exception:
                    pass
                cleanup_cuda()

    finally:
        try:
            del matcher, gp
        except Exception:
            pass
        cleanup_cuda()

    with open(result_path, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

def run_tile_worker_batch(
    *,
    args: argparse.Namespace,
    scene: str,
    scene_dir: Path,
    tile_specs: list[dict[str, Any]],
    tile_sample_num: int,
    tmp_dir: Path,
    batch_idx: int,
) -> list[dict[str, Any]]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    payload_path = tmp_dir / f"{scene}_tile_batch_{batch_idx:04d}.json"
    result_path = tmp_dir / f"{scene}_tile_batch_{batch_idx:04d}.pkl"

    payload = {
        "args": vars(args),
        "scene": scene,
        "scene_dir": str(scene_dir),
        "tile_sample_num": int(tile_sample_num),
        "tiles": tile_specs,
        "result_path": str(result_path),
    }

    with open(payload_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env["PYTHONPATH"] = os.environ.get("PYTHONPATH", str(Path.cwd()))

    cmd = [sys.executable, str(Path(__file__).resolve()), "--tile_worker_payload", str(payload_path)]
    subprocess.run(cmd, check=True, env=env)

    with open(result_path, "rb") as f:
        results = pickle.load(f)

    try:
        payload_path.unlink(missing_ok=True)
        result_path.unlink(missing_ok=True)
    except Exception:
        pass

    return results

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Memory-safe final coarse-to-tile RoMa pipeline: "
            "homography RANSAC for the coarse stage, tile RoMa in subprocess batches, "
            "then TPS for the final tile-refined filter."
        )
    )
    ap.add_argument("--scenes_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no_deterministic_cudnn",
        action="store_false",
        dest="deterministic_cudnn",
        help="Disable deterministic cuDNN settings.",
    )
    ap.set_defaults(deterministic_cudnn=True)

    add_roma_args(ap)
    ap.add_argument("--coarse_conf_keep_pct", type=int, default=60)
    ap.add_argument("--coarse_min_conf", type=float, default=0.20)
    ap.add_argument("--coarse_min_matches", type=int, default=80)
    ap.add_argument("--coarse_spatial_cell_px", type=int, default=128)
    ap.add_argument("--coarse_spatial_max_per_cell", type=int, default=8)
    ap.add_argument("--min_export_matches", type=int, default=12)
    add_tps_args(ap)
    ap.add_argument("--ransac_reproj_thresh_px", type=float, default=8.0)
    ap.add_argument("--ransac_confidence", type=float, default=0.999)
    ap.add_argument("--ransac_max_iters", type=int, default=5000)
    ap.add_argument("--ransac_refine_iters", type=int, default=10)
    ap.add_argument("--tile_size", type=int, default=1120)
    ap.add_argument("--tile_overlap", type=int, default=64)
    ap.add_argument("--tile_min_valid_frac", type=float, default=0.05)
    ap.add_argument("--tile_min_side_px", type=int, default=256)
    ap.add_argument("--refine_margin_px", type=int, default=256)
    ap.add_argument("--tile_conf_keep_pct", type=int, default=60)
    ap.add_argument("--tile_min_conf", type=float, default=0.20)
    ap.add_argument("--tile_spatial_cell_px", type=int, default=128)
    ap.add_argument("--tile_spatial_max_per_cell", type=int, default=4)
    ap.add_argument("--tile_prior_consistency_px", type=float, default=64.0)
    ap.add_argument("--tile_worker_batch_size", type=int, default=4)
    ap.add_argument("--merge_src_round_px", type=float, default=0.5)
    ap.add_argument("--merge_dst_round_px", type=float, default=0.5)
    ap.add_argument("--final_conf_keep_pct", type=int, default=60)
    ap.add_argument("--final_min_conf", type=float, default=0.20)
    ap.add_argument("--final_min_matches", type=int, default=80)
    ap.add_argument("--final_spatial_cell_px", type=int, default=128)
    ap.add_argument("--final_spatial_max_per_cell", type=int, default=8)
    ap.add_argument("--max_export_median_px", type=float, default=10.0)
    ap.add_argument("--max_export_p90_px", type=float, default=30.0)
    ap.add_argument("--save_all", action="store_true")
    return ap

def main() -> None:
    args = build_parser().parse_args()
    set_reproducibility(args.seed, deterministic_cudnn=args.deterministic_cudnn)

    # Final production choice: fixed homography coarse model and final TPS filter.
    args.coarse_filter_model = "homography_ransac"
    args.final_filter_model = "tps"
    final_filter_models = ["tps"]

    scenes_root = Path(args.scenes_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_aligned_dir = out_dir / "aligned"
    out_aligned_dir.mkdir(exist_ok=True)

    print("[INFO] Creating parent RoMa matcher for coarse scene matching", flush=True)
    matcher = make_roma_matcher(args, sample_num=args.roma_sample_num)
    tile_sample_num = args.roma_sample_num_tile if args.roma_sample_num_tile is not None else args.roma_sample_num

    scene_dirs = sorted([p for p in scenes_root.iterdir() if p.is_dir()])
    n_saved = 0
    n_failed = 0
    desc = "Coarse-to-tile RoMa | coarse=homography_ransac | final=tps | memory-safe"

    for sd in tqdm(scene_dirs, desc=desc):
        scene = sd.name
        gp = None
        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            inp = find_scene_inputs(sd)
            gp = load_armsat_and_buffered_sentinel_for_matching(
                armsat_path=inp["armsat"],
                s2_path=inp["s2"],
                out_nodata=-9999.0,
            )

            coarse_raw = matcher.infer_raw(
                gp.mov_img01,
                gp.ref_img01,
                gp.mov_valid,
                gp.ref_valid,
                sample_num=args.roma_sample_num,
            )
            coarse_raw = raw_matchset_to_cpu(coarse_raw)
            cleanup_cuda()

            save_raw_matches_json(
                out_aligned_dir,
                scene,
                "roma",
                coarse_raw,
                gp.ref_img01.shape[:2],
                stage_name="coarse",
                grid_strategy="roma_homography_tps_coarse_raw_before_filtering",
            )

            coarse_mr = matcher.filter_matches(
                coarse_raw,
                image_shape=gp.ref_img01.shape[:2],
                conf_keep_pct=args.coarse_conf_keep_pct,
                min_conf=args.coarse_min_conf,
                min_matches=args.coarse_min_matches,
                spatial_cell_px=args.coarse_spatial_cell_px,
                spatial_max_per_cell=args.coarse_spatial_max_per_cell,
                min_export_matches=args.min_export_matches,
                **filter_kwargs_from_args(args, filter_model=args.coarse_filter_model),
            )

            if not (
                coarse_mr.exportable
                and coarse_mr.pts0 is not None
                and coarse_mr.pts1 is not None
                and coarse_mr.conf is not None
            ):
                n_failed += 1
                print(f"[SKIP] {scene}: coarse stage failed: {coarse_mr.reason}", flush=True)
                continue

            coarse_model_type, coarse_model, coord_scale = fit_coarse_prediction_model(
                coarse_mr.pts0,
                coarse_mr.pts1,
                image_shape=gp.ref_img01.shape[:2],
                filter_model=args.coarse_filter_model,
                tps_smoothing=args.tps_smoothing,
                ransac_reproj_thresh_px=args.ransac_reproj_thresh_px,
                ransac_confidence=args.ransac_confidence,
                ransac_max_iters=args.ransac_max_iters,
                ransac_refine_iters=args.ransac_refine_iters,
            )

            h, w = gp.ref_img01.shape[:2]
            tiles = generate_tiles(h=h, w=w, tile_size=args.tile_size, overlap=args.tile_overlap)

            refine_pts0: list[np.ndarray] = []
            refine_pts1: list[np.ndarray] = []
            refine_conf: list[np.ndarray] = []
            tile_rows: list[dict[str, Any]] = []
            tile_specs: list[dict[str, Any]] = []

            n_tiles_total = 0
            n_tiles_used = 0
            n_tiles_skip_small = 0
            n_tiles_skip_valid = 0
            n_tiles_skip_empty_ref = 0
            n_tiles_zero_raw = 0
            n_tiles_rejected_filter = 0
            n_tiles_rejected_prior = 0

            for tile_id, (y0, y1, x0, x1) in enumerate(tiles):
                n_tiles_total += 1
                th = y1 - y0
                tw = x1 - x0
                row: dict[str, Any] = {
                    "tile_id": int(tile_id),
                    "tile_window_yx": [int(y0), int(y1), int(x0), int(x1)],
                    "tile_shape_hw": [int(th), int(tw)],
                }

                if th < args.tile_min_side_px or tw < args.tile_min_side_px:
                    n_tiles_skip_small += 1
                    row["status"] = "skip_small"
                    tile_rows.append(row)
                    continue

                mov_valid_tile = gp.mov_valid[y0:y1, x0:x1]
                mov_valid_frac = float(mov_valid_tile.mean())
                row["mov_valid_frac"] = mov_valid_frac
                if mov_valid_frac < args.tile_min_valid_frac:
                    n_tiles_skip_valid += 1
                    row["status"] = "skip_low_valid"
                    tile_rows.append(row)
                    continue

                ry0, ry1, rx0, rx1 = predict_ref_window_from_coarse_model(
                    model_type=coarse_model_type,
                    model=coarse_model,
                    coord_scale=coord_scale,
                    tile_window=(y0, y1, x0, x1),
                    h_ref=h,
                    w_ref=w,
                    refine_margin_px=args.refine_margin_px,
                )

                if ry1 <= ry0 or rx1 <= rx0:
                    n_tiles_skip_empty_ref += 1
                    row["status"] = "skip_empty_pred_window"
                    row["pred_ref_window_yx"] = [int(ry0), int(ry1), int(rx0), int(rx1)]
                    tile_rows.append(row)
                    continue

                ref_valid_tile = gp.ref_valid[ry0:ry1, rx0:rx1]
                if ref_valid_tile.size == 0:
                    n_tiles_skip_empty_ref += 1
                    row["status"] = "skip_empty_ref"
                    row["pred_ref_window_yx"] = [int(ry0), int(ry1), int(rx0), int(rx1)]
                    tile_rows.append(row)
                    continue

                ref_valid_frac = float(ref_valid_tile.mean())
                row["pred_ref_window_yx"] = [int(ry0), int(ry1), int(rx0), int(rx1)]
                row["pred_ref_shape_hw"] = [int(ry1 - ry0), int(rx1 - rx0)]
                row["ref_valid_frac"] = ref_valid_frac
                if ref_valid_frac <= 0.0:
                    n_tiles_skip_empty_ref += 1
                    row["status"] = "skip_empty_ref"
                    tile_rows.append(row)
                    continue

                tile_specs.append(
                    {
                        "tile_id": int(tile_id),
                        "tile_window_yx": [int(y0), int(y1), int(x0), int(x1)],
                        "pred_ref_window_yx": [int(ry0), int(ry1), int(rx0), int(rx1)],
                        "row": row,
                    }
                )

            cleanup_cuda()

            tmp_worker_dir = out_dir / "_tile_worker_tmp"
            batch_size = max(1, int(args.tile_worker_batch_size))

            for batch_idx, start_idx in enumerate(range(0, len(tile_specs), batch_size)):
                batch_specs = tile_specs[start_idx:start_idx + batch_size]
                print(
                    f"[INFO] {scene} | tile worker batch {batch_idx} | "
                    f"tiles={batch_specs[0]['tile_id']}..{batch_specs[-1]['tile_id']}",
                    flush=True,
                )

                worker_results = run_tile_worker_batch(
                    args=args,
                    scene=scene,
                    scene_dir=sd,
                    tile_specs=batch_specs,
                    tile_sample_num=int(tile_sample_num),
                    tmp_dir=tmp_worker_dir,
                    batch_idx=batch_idx,
                )
                cleanup_cuda()

                for result in worker_results:
                    row = dict(result["row"])
                    raw_tile = filtered_tile = None
                    pts0_global = pts1_global = conf_global = None
                    try:
                        if not result.get("ok", False):
                            kind = result.get("kind", "error")
                            if kind == "zero_raw":
                                n_tiles_zero_raw += 1
                            else:
                                n_tiles_rejected_filter += 1
                            tile_rows.append(row)
                            continue

                        raw_tile = RawMatchSet(
                            pts0=np.asarray(result["pts0"], dtype=np.float32),
                            pts1=np.asarray(result["pts1"], dtype=np.float32),
                            conf=np.asarray(result["conf"], dtype=np.float32),
                        )
                        row["n_raw_matches"] = int(len(raw_tile.conf))

                        ry0, ry1, rx0, rx1 = row["pred_ref_window_yx"]
                        ref_tile_shape = (int(ry1) - int(ry0), int(rx1) - int(rx0))

                        filtered_tile, tile_stats = minimal_tile_filter(
                            raw_tile,
                            ref_tile_shape=ref_tile_shape,
                            conf_keep_pct=args.tile_conf_keep_pct,
                            min_conf=args.tile_min_conf,
                            spatial_cell_px=args.tile_spatial_cell_px,
                            spatial_max_per_cell=args.tile_spatial_max_per_cell,
                        )
                        row["tile_filter"] = tile_stats

                        if filtered_tile is None:
                            n_tiles_rejected_filter += 1
                            row["status"] = "rejected_by_tile_filter"
                            tile_rows.append(row)
                            continue

                        y0, y1, x0, x1 = row["tile_window_yx"]
                        mov_offset = np.asarray([x0, y0], dtype=np.float32)
                        ref_offset = np.asarray([rx0, ry0], dtype=np.float32)
                        pts0_global = (filtered_tile.pts0 + mov_offset).astype(np.float32)
                        pts1_global = (filtered_tile.pts1 + ref_offset).astype(np.float32)
                        conf_global = filtered_tile.conf.astype(np.float32)

                        pred_ref = predict_points_with_coarse_model(
                            coarse_model_type,
                            coarse_model,
                            pts0_global,
                            coord_scale=coord_scale,
                        )
                        prior_res = np.linalg.norm(pred_ref - pts1_global, axis=1)
                        row["prior_model_type"] = coarse_model_type
                        row["prior_residual_median_px"] = float(np.median(prior_res))
                        row["prior_residual_p90_px"] = float(np.percentile(prior_res, 90))

                        keep_prior = prior_res <= float(args.tile_prior_consistency_px)
                        pts0_global = pts0_global[keep_prior].astype(np.float32, copy=True)
                        pts1_global = pts1_global[keep_prior].astype(np.float32, copy=True)
                        conf_global = conf_global[keep_prior].astype(np.float32, copy=True)
                        row["n_after_prior_gate"] = int(len(conf_global))

                        if len(conf_global) == 0:
                            n_tiles_rejected_prior += 1
                            row["status"] = "rejected_by_prior_gate"
                            tile_rows.append(row)
                            continue

                        refine_pts0.append(pts0_global)
                        refine_pts1.append(pts1_global)
                        refine_conf.append(conf_global)
                        row["status"] = "used_refined"
                        n_tiles_used += 1
                        tile_rows.append(row)

                    finally:
                        try:
                            del raw_tile, filtered_tile, pts0_global, pts1_global, conf_global
                        except Exception:
                            pass
                        cleanup_cuda()

                try:
                    del worker_results, batch_specs
                except Exception:
                    pass
                cleanup_cuda()

            try:
                del tile_specs
            except Exception:
                pass
            cleanup_cuda()

            coarse_pts0 = coarse_mr.pts0.astype(np.float32)
            coarse_pts1 = coarse_mr.pts1.astype(np.float32)
            coarse_conf = coarse_mr.conf.astype(np.float32)
            if refine_pts0:
                pts0_all = np.concatenate([coarse_pts0] + refine_pts0, axis=0)
                pts1_all = np.concatenate([coarse_pts1] + refine_pts1, axis=0)
                conf_all = np.concatenate([coarse_conf] + refine_conf, axis=0)
            else:
                pts0_all = coarse_pts0
                pts1_all = coarse_pts1
                conf_all = coarse_conf

            premerge_match_count = int(len(conf_all))
            all_raw = RawMatchSet(pts0=pts0_all, pts1=pts1_all, conf=conf_all)
            save_raw_matches_json(
                out_aligned_dir,
                scene,
                "roma",
                all_raw,
                gp.ref_img01.shape[:2],
                stage_name="refined",
                grid_strategy="roma_homography_tps_refined_matches_before_merge",
            )

            pts0_merged, pts1_merged, conf_merged, merge_stats = merge_duplicate_matches_global(
                pts0_all,
                pts1_all,
                conf_all,
                src_round_px=args.merge_src_round_px,
                dst_round_px=args.merge_dst_round_px,
            )
            merged_match_count = int(len(conf_merged))
            merged_raw = RawMatchSet(pts0=pts0_merged, pts1=pts1_merged, conf=conf_merged)
            save_raw_matches_json(
                out_aligned_dir,
                scene,
                "roma",
                merged_raw,
                gp.ref_img01.shape[:2],
                stage_name="merged",
                grid_strategy="roma_homography_tps_merged_matches_before_final_filtering",
                extra={"merge_duplicates": merge_stats},
            )
            out_tile_json = save_tile_diagnostics(out_aligned_dir, scene, tile_rows)

            coarse_config = {
                "coarse_conf_keep_pct": int(args.coarse_conf_keep_pct),
                "coarse_min_conf": float(args.coarse_min_conf),
                "coarse_min_matches": int(args.coarse_min_matches),
                "coarse_spatial_cell_px": int(args.coarse_spatial_cell_px),
                "coarse_spatial_max_per_cell": int(args.coarse_spatial_max_per_cell),
                "coarse_filter_model": str(args.coarse_filter_model),
                "roma_variant": str(args.roma_variant),
                "roma_sample_num": int(args.roma_sample_num),
                "roma_sample_num_tile": int(tile_sample_num),
                "roma_sample_thresh": float(args.roma_sample_thresh) if args.roma_sample_thresh is not None else None,
                "roma_w_resized": args.roma_w_resized,
                "roma_h_resized": args.roma_h_resized,
                "roma_upsample_w": args.roma_upsample_w,
                "roma_upsample_h": args.roma_upsample_h,
                "ransac_reproj_thresh_px": float(args.ransac_reproj_thresh_px),
                "ransac_confidence": float(args.ransac_confidence),
                "ransac_max_iters": int(args.ransac_max_iters),
                "ransac_refine_iters": int(args.ransac_refine_iters),
                "seed": int(args.seed),
                "deterministic_cudnn": bool(args.deterministic_cudnn),
                "tile_worker_batch_size": int(args.tile_worker_batch_size),
            }
            base_refine_config = {
                "tile_size": int(args.tile_size),
                "tile_overlap": int(args.tile_overlap),
                "tile_min_valid_frac": float(args.tile_min_valid_frac),
                "tile_min_side_px": int(args.tile_min_side_px),
                "refine_margin_px": int(args.refine_margin_px),
                "tile_conf_keep_pct": int(args.tile_conf_keep_pct),
                "tile_min_conf": float(args.tile_min_conf),
                "tile_spatial_cell_px": int(args.tile_spatial_cell_px),
                "tile_spatial_max_per_cell": int(args.tile_spatial_max_per_cell),
                "tile_prior_consistency_px": float(args.tile_prior_consistency_px),
                "merge_src_round_px": float(args.merge_src_round_px),
                "merge_dst_round_px": float(args.merge_dst_round_px),
            }

            saved_any_filter = False
            for fm in final_filter_models:
                final_mr = matcher.filter_matches(
                    merged_raw,
                    image_shape=gp.ref_img01.shape[:2],
                    conf_keep_pct=args.final_conf_keep_pct,
                    min_conf=args.final_min_conf,
                    min_matches=args.final_min_matches,
                    spatial_cell_px=args.final_spatial_cell_px,
                    spatial_max_per_cell=args.final_spatial_max_per_cell,
                    min_export_matches=args.min_export_matches,
                    **filter_kwargs_from_args(args, filter_model=fm),
                )

                if not (
                    final_mr.exportable
                    and final_mr.pts0 is not None
                    and final_mr.pts1 is not None
                    and final_mr.conf is not None
                ):
                    print(f"[SKIP] {scene}: final stage failed | final_filter={fm}: {final_mr.reason}", flush=True)
                    continue

                bad_geometry = is_bad_geometry(
                    final_mr,
                    max_median_px=args.max_export_median_px,
                    max_p90_px=args.max_export_p90_px,
                )
                if bad_geometry and not args.save_all:
                    print(
                        f"[REJECT] {scene}: bad geometry | final_filter={fm} | "
                        f"med={final_mr.residual_median_px:.3f} | p90={final_mr.residual_p90_px:.3f}",
                        flush=True,
                    )
                    continue

                scene_diagnostics = {
                    "coarse_stage": {
                        "coarse_filter_model": str(args.coarse_filter_model),
                        "coarse_prediction_model_type": str(coarse_model_type),
                        "n_coarse_matches": int(len(coarse_conf)),
                        "coarse_residual_median_px": float(coarse_mr.residual_median_px) if coarse_mr.residual_median_px is not None else None,
                        "coarse_residual_p90_px": float(coarse_mr.residual_p90_px) if coarse_mr.residual_p90_px is not None else None,
                    },
                    "tile_refine_stage": {
                        "n_tiles_total": int(n_tiles_total),
                        "n_tiles_used": int(n_tiles_used),
                        "n_tiles_skip_small": int(n_tiles_skip_small),
                        "n_tiles_skip_valid": int(n_tiles_skip_valid),
                        "n_tiles_skip_empty_ref": int(n_tiles_skip_empty_ref),
                        "n_tiles_zero_raw": int(n_tiles_zero_raw),
                        "n_tiles_rejected_filter": int(n_tiles_rejected_filter),
                        "n_tiles_rejected_prior": int(n_tiles_rejected_prior),
                        "n_refine_matches_total": int(sum(len(x) for x in refine_conf)) if refine_conf else 0,
                    },
                    "final_stage": {
                        "requested_final_filter_model": str(args.final_filter_model),
                        "final_filter_model": str(fm),
                        "premerge_match_count": int(premerge_match_count),
                        "merged_match_count": int(merged_match_count),
                        "final_match_count": int(len(final_mr.conf)),
                    },
                    "merge_duplicates": merge_stats,
                    "bad_geometry_saved": bool(bad_geometry),
                }
                refine_config = {
                    **base_refine_config,
                    "requested_final_filter_model": str(args.final_filter_model),
                    "final_filter_model": str(fm),
                }

                out_json = save_matches_json(
                    out_aligned_dir,
                    scene,
                    "roma",
                    final_mr,
                    gp.ref_img01.shape[:2],
                    grid_strategy="roma_homography_tps",
                    extra={
                        "coarse_config": coarse_config,
                        "refine_config": refine_config,
                        "scene_diagnostics": scene_diagnostics,
                    },
                )

                n_saved += 1
                saved_any_filter = True
                prefix = "[WARN-SAVED]" if bad_geometry else "[OK]"
                print(
                    f"{prefix} {scene} | coarse_filter={args.coarse_filter_model} | final_filter={fm} | "
                    f"coarse={len(coarse_conf)} | tiles_used={n_tiles_used}/{n_tiles_total} | "
                    f"refine_matches={scene_diagnostics['tile_refine_stage']['n_refine_matches_total']} | "
                    f"premerge={premerge_match_count} | merged={merged_match_count} | final={len(final_mr.conf)} | "
                    f"med={final_mr.residual_median_px:.3f} | p90={final_mr.residual_p90_px:.3f} | "
                    f"out={out_json.name} | tiles={out_tile_json.name}",
                    flush=True,
                )

            if not saved_any_filter:
                n_failed += 1
                print(f"[SKIP] {scene}: no final filters were saved", flush=True)

        except Exception as e:
            n_failed += 1
            print(f"[FAIL] {scene}: {repr(e)}", flush=True)
        finally:
            try:
                del gp, coarse_raw, coarse_mr, coarse_model, coord_scale
            except Exception:
                pass
            try:
                del tiles, tile_specs, refine_pts0, refine_pts1, refine_conf, tile_rows
            except Exception:
                pass
            try:
                del pts0_all, pts1_all, conf_all, all_raw, pts0_merged, pts1_merged, conf_merged, merged_raw
            except Exception:
                pass
            try:
                del final_mr, scene_diagnostics, coarse_config, refine_config
            except Exception:
                pass
            cleanup_cuda()

    print("\n[OUTPUT]")
    print(out_aligned_dir, "contains matches_raw_coarse.json, matches_raw_refined.json, matches_raw_merged.json, matches_filtered.json, and tile_diagnostics.json per scene")
    print(f"Saved scenes: {n_saved}")
    print(f"Skipped/failed scenes: {n_failed}")

if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--tile_worker_payload":
        tile_worker_main(sys.argv[2])
    else:
        main()