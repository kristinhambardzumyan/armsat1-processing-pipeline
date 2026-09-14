from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Optional
import cv2
import numpy as np
import rasterio
from matching.models.base import RawMatchSet, MatchResult, _fit_tps_rbf, _predict_tps

def find_scene_inputs(scene_dir: Path) -> dict[str, Path]:
    armsat_candidates = sorted(scene_dir.glob("armsat_rgb_utm*.tif"))
    if not armsat_candidates:
        armsat_native = scene_dir / "armsat_rgb_native.tif"
        if armsat_native.exists():
            armsat_candidates = [armsat_native]
    if not armsat_candidates:
        raise FileNotFoundError(f"Missing ARMSAT RGB tif in {scene_dir}")
    armsat = armsat_candidates[0]

    s2_candidates = sorted(scene_dir.glob("sentinel2_rgb_utm*.tif"))

    if not s2_candidates:
        s2_dir = scene_dir / "s2_clipped_utm"
        s2_candidates = sorted(s2_dir.glob("s2_rgb_clip_utm*.tif"))

    if not s2_candidates:
        raise FileNotFoundError(f"Missing Sentinel UTM tif in {scene_dir}")

    s2 = s2_candidates[0]

    return {"armsat": armsat, "s2": s2}

def find_scene_reference_candidates(scene_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    """Return the moving image and configured reference crops (center first)."""
    inputs = find_scene_inputs(scene_dir)
    summary_path = scene_dir / "scene_prep_summary.json"
    configured = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        configured = summary.get("multi_reference", {}).get("candidates")

    if not configured:
        with rasterio.open(inputs["s2"]) as ds:
            bounds = [ds.bounds.left, ds.bounds.bottom, ds.bounds.right, ds.bounds.top]
        return inputs["armsat"].resolve(), [{
            "direction": "C",
            "path": inputs["s2"].resolve(),
            "crop_bounds": bounds,
        }]

    candidates: list[dict[str, Any]] = []
    for item in configured:
        path = Path(item["path"])
        if not path.is_absolute():
            path = scene_dir / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing configured reference candidate: {path}")
        candidates.append({
            "direction": str(item["direction"]).upper(),
            "path": path,
            "crop_bounds": [float(x) for x in item["crop_bounds"]],
        })
    return inputs["armsat"].resolve(), candidates

def axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_overlap must satisfy 0 <= overlap < tile_size")

    stride = tile_size - overlap
    starts = list(range(0, max(length - tile_size, 0) + 1, stride))

    if not starts:
        return [0]

    end_start = max(0, length - tile_size)
    if starts[-1] != end_start:
        starts.append(end_start)

    return sorted(set(starts))

def generate_tiles(
    h: int,
    w: int,
    tile_size: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    ys = axis_starts(h, tile_size, overlap)
    xs = axis_starts(w, tile_size, overlap)

    tiles: list[tuple[int, int, int, int]] = []
    for y0 in ys:
        y1 = min(h, y0 + tile_size)
        for x0 in xs:
            x1 = min(w, x0 + tile_size)
            tiles.append((y0, y1, x0, x1))

    return tiles

def clamp_window(
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    h: int,
    w: int,
) -> tuple[int, int, int, int]:
    y0 = max(0, min(y0, h))
    y1 = max(0, min(y1, h))
    x0 = max(0, min(x0, w))
    x1 = max(0, min(x1, w))
    return y0, y1, x0, x1

def build_large_context_ref_window(
    tile_window: tuple[int, int, int, int],
    h_ref: int,
    w_ref: int,
    context_margin_px: int,
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = tile_window
    return clamp_window(
        y0 - context_margin_px,
        y1 + context_margin_px,
        x0 - context_margin_px,
        x1 + context_margin_px,
        h_ref,
        w_ref,
    )

def spatial_thin_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    image_shape: tuple[int, int],
    cell_px: int,
    max_per_cell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_shape

    if len(conf) == 0:
        return pts0, pts1, conf

    order = np.argsort(-conf)
    pts0 = pts0[order]
    pts1 = pts1[order]
    conf = conf[order]

    buckets: dict[tuple[int, int], int] = {}
    keep_idx: list[int] = []

    for i, p in enumerate(pts1):
        x, y = float(p[0]), float(p[1])

        if not (0 <= x < w and 0 <= y < h):
            continue

        cell = (int(x // cell_px), int(y // cell_px))
        cnt = buckets.get(cell, 0)

        if cnt >= max_per_cell:
            continue

        buckets[cell] = cnt + 1
        keep_idx.append(i)

    if not keep_idx:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
        )

    keep_idx_arr = np.asarray(keep_idx, dtype=np.int64)
    return pts0[keep_idx_arr], pts1[keep_idx_arr], conf[keep_idx_arr]

def minimal_tile_filter(
    raw: RawMatchSet,
    *,
    ref_tile_shape: tuple[int, int],
    conf_keep_pct: int,
    min_conf: float,
    spatial_cell_px: int,
    spatial_max_per_cell: int,
) -> tuple[Optional[RawMatchSet], dict[str, Any]]:
    pts0 = raw.pts0
    pts1 = raw.pts1
    conf = raw.conf
    n_raw = int(len(conf))
    if n_raw == 0:
        return None, {"reason": "no_raw_matches", "n_raw": 0}
    pct_thr = np.percentile(conf, 100 - conf_keep_pct)
    keep = conf >= max(float(min_conf), float(pct_thr))
    pts0 = pts0[keep]
    pts1 = pts1[keep]
    conf = conf[keep]
    n_after_conf = int(len(conf))
    if n_after_conf == 0:
        return None, {
            "reason": "no_matches_after_conf",
            "n_raw": n_raw,
            "n_after_conf": 0,
        }
    pts0, pts1, conf = spatial_thin_matches(
        pts0,
        pts1,
        conf,
        image_shape=ref_tile_shape,
        cell_px=spatial_cell_px,
        max_per_cell=spatial_max_per_cell,
    )
    n_after_spatial = int(len(conf))
    if n_after_spatial == 0:
        return None, {
            "reason": "no_matches_after_spatial",
            "n_raw": n_raw,
            "n_after_conf": n_after_conf,
            "n_after_spatial": 0,
        }
    return RawMatchSet(
        pts0=pts0.astype(np.float32),
        pts1=pts1.astype(np.float32),
        conf=conf.astype(np.float32),
    ), {
        "reason": "ok",
        "n_raw": n_raw,
        "n_after_conf": n_after_conf,
        "n_after_spatial": n_after_spatial,
    }

def merge_duplicate_matches_global(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    *,
    src_round_px: float,
    dst_round_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if len(conf) == 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            {
                "n_input": 0,
                "n_merged": 0,
                "n_removed_as_duplicates": 0,
                "src_round_px": float(src_round_px),
                "dst_round_px": float(dst_round_px),
            },
        )

    best: dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}
    for p0, p1, c in zip(pts0, pts1, conf):
        key = (
            round(float(p0[0]) / src_round_px),
            round(float(p0[1]) / src_round_px),
            round(float(p1[0]) / dst_round_px),
            round(float(p1[1]) / dst_round_px),
        )
        old = best.get(key)
        if old is None or float(c) > old[2]:
            best[key] = (p0, p1, float(c))
    vals = list(best.values())
    pts0_out = np.asarray([v[0] for v in vals], dtype=np.float32)
    pts1_out = np.asarray([v[1] for v in vals], dtype=np.float32)
    conf_out = np.asarray([v[2] for v in vals], dtype=np.float32)
    stats = {
        "n_input": int(len(conf)),
        "n_merged": int(len(conf_out)),
        "n_removed_as_duplicates": int(len(conf) - len(conf_out)),
        "src_round_px": float(src_round_px),
        "dst_round_px": float(dst_round_px),
    }
    return pts0_out, pts1_out, conf_out, stats

def coverage_ratio(
    pts: np.ndarray,
    image_shape: tuple[int, int],
    cell_px: int,
) -> float:
    h, w = image_shape
    if len(pts) == 0:
        return 0.0
    occupied = set()
    for x, y in pts:
        if 0 <= x < w and 0 <= y < h:
            occupied.add((int(x // cell_px), int(y // cell_px)))
    n_cells_x = max(1, int(np.ceil(w / cell_px)))
    n_cells_y = max(1, int(np.ceil(h / cell_px)))
    return float(len(occupied)) / float(n_cells_x * n_cells_y)

def fit_coarse_prediction_model(
    pts0: np.ndarray,
    pts1: np.ndarray,
    *,
    image_shape: tuple[int, int],
    filter_model: str,
    tps_smoothing: float,
    ransac_reproj_thresh_px: float,
    ransac_confidence: float,
    ransac_max_iters: int,
    ransac_refine_iters: int,
) -> tuple[str, Any, Optional[float]]:
    filter_model = str(filter_model).lower()

    if filter_model == "tps":
        coord_scale = float(max(image_shape))
        model = _fit_tps_rbf(
            pts0.astype(np.float32),
            pts1.astype(np.float32),
            coord_scale=coord_scale,
            smoothing=float(tps_smoothing),
        )
        return "tps", model, coord_scale

    if filter_model == "affine_ransac":
        model, _ = cv2.estimateAffine2D(
            pts0.astype(np.float32),
            pts1.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_thresh_px),
            maxIters=int(ransac_max_iters),
            confidence=float(ransac_confidence),
            refineIters=int(ransac_refine_iters),
        )
        if model is None:
            raise RuntimeError("Could not fit affine model for coarse tile-window prediction.")
        return "affine_ransac", model.astype(np.float32), None

    if filter_model == "homography_ransac":
        model, _ = cv2.findHomography(
            pts0.astype(np.float32),
            pts1.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=float(ransac_reproj_thresh_px),
            maxIters=int(ransac_max_iters),
            confidence=float(ransac_confidence),
        )
        if model is None:
            raise RuntimeError("Could not fit homography model for coarse tile-window prediction.")
        return "homography_ransac", model.astype(np.float32), None

    raise ValueError(
        f"Unknown filter_model={filter_model}. "
        "Use one of: tps, affine_ransac, homography_ransac."
    )

def predict_points_with_coarse_model(
    model_type: str,
    model: Any,
    pts_xy: np.ndarray,
    *,
    coord_scale: Optional[float],
) -> np.ndarray:
    model_type = str(model_type).lower()

    if model_type == "tps":
        if coord_scale is None:
            raise RuntimeError("coord_scale is required for TPS prediction.")
        return _predict_tps(model, pts_xy.astype(np.float32), coord_scale=coord_scale)

    if model_type == "affine_ransac":
        pred = cv2.transform(
            pts_xy.reshape(-1, 1, 2).astype(np.float32),
            model,
        )
        return pred.reshape(-1, 2).astype(np.float32)

    if model_type == "homography_ransac":
        pred = cv2.perspectiveTransform(
            pts_xy.reshape(-1, 1, 2).astype(np.float32),
            model,
        )
        return pred.reshape(-1, 2).astype(np.float32)
    raise ValueError(f"Unknown coarse model type: {model_type}")

def predict_ref_window_from_coarse_model(
    *,
    model_type: str,
    model: Any,
    coord_scale: Optional[float],
    tile_window: tuple[int, int, int, int],
    h_ref: int,
    w_ref: int,
    refine_margin_px: int,
) -> tuple[int, int, int, int]:
    y0, y1, x0, x1 = tile_window

    corners = np.asarray(
        [
            [x0, y0],
            [x1 - 1, y0],
            [x0, y1 - 1],
            [x1 - 1, y1 - 1],
            [(x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0],
        ],
        dtype=np.float32,
    )
    pred = predict_points_with_coarse_model(
        model_type,
        model,
        corners,
        coord_scale=coord_scale,
    )
    xs = pred[:, 0]
    ys = pred[:, 1]
    rx0 = int(np.floor(xs.min())) - int(refine_margin_px)
    rx1 = int(np.ceil(xs.max())) + int(refine_margin_px) + 1
    ry0 = int(np.floor(ys.min())) - int(refine_margin_px)
    ry1 = int(np.ceil(ys.max())) + int(refine_margin_px) + 1
    return clamp_window(ry0, ry1, rx0, rx1, h_ref, w_ref)

def save_matches_json(
    out_aligned_dir: Path,
    scene: str,
    matcher_name: str,
    mr: MatchResult,
    ref_shape: tuple[int, int],
    *,
    grid_strategy: str,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    p = out_aligned_dir / scene / "matches_filtered.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scene": scene,
        "matcher": matcher_name,
        "method": mr.method,
        "pts_mov_on_ref": mr.pts0.tolist(),
        "pts_ref": mr.pts1.tolist(),
        "confidence": mr.conf.tolist() if mr.conf is not None else [],
        "ref_shape_hw": [int(ref_shape[0]), int(ref_shape[1])],
        "grid_strategy": grid_strategy,
        "filter_model": mr.filter_model,
        "match_stage": mr.match_stage,
        "n_matches_raw": mr.n_matches_raw,
        "n_matches_after_conf": mr.n_matches_after_conf,
        "n_matches_after_spatial": mr.n_matches_after_spatial,
        "n_matches_inliers": mr.n_matches_inliers,
        "inlier_ratio": mr.inlier_ratio,
        "n_filter_iters": mr.n_filter_iters,
        "residual_median_px": mr.residual_median_px,
        "residual_p90_px": mr.residual_p90_px,
    }

    if extra:
        payload.update(extra)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p

def save_reference_candidate_diagnostics(
    out_aligned_dir: Path,
    scene: str,
    rows: list[dict[str, Any]],
) -> Path:
    p = out_aligned_dir / scene / "reference_candidates.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return p

def save_raw_matches_json(
    out_aligned_dir: Path,
    scene: str,
    matcher_name: str,
    raw: RawMatchSet,
    ref_shape: tuple[int, int],
    *,
    stage_name: str = "raw",
    grid_strategy: str,
    extra: Optional[dict[str, Any]] = None,
) -> Path:
    p = out_aligned_dir / scene / f"matches_raw_{stage_name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "scene": scene,
        "matcher": matcher_name,
        "stage": stage_name,
        "pts_mov_on_ref": raw.pts0.tolist(),
        "pts_ref": raw.pts1.tolist(),
        "confidence": raw.conf.tolist(),
        "ref_shape_hw": [int(ref_shape[0]), int(ref_shape[1])],
        "n_matches": int(len(raw.conf)),
        "grid_strategy": grid_strategy,
    }

    if extra:
        payload.update(extra)

    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p

def save_tile_diagnostics(
    out_aligned_dir: Path,
    scene: str,
    rows: list[dict[str, Any]],
) -> Path:
    p = out_aligned_dir / scene / "tile_diagnostics.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return p

def add_tps_args(ap) -> None:
    ap.add_argument("--min_tps_matches_total", type=int, default=12)
    ap.add_argument("--min_tps_train_matches_per_fold", type=int, default=8)
    ap.add_argument("--tps_smoothing", type=float, default=1e-3)
    ap.add_argument("--tps_abs_thresh_px", type=float, default=6.0)
    ap.add_argument("--tps_mad_mult", type=float, default=3.0)
    ap.add_argument("--tps_max_iters", type=int, default=5)
    ap.add_argument("--tps_max_remove_frac", type=float, default=0.20)
    ap.add_argument("--cv_block_px", type=int, default=256)
    ap.add_argument("--cv_max_folds", type=int, default=4)

def filter_kwargs_from_args(args, *, filter_model: str) -> dict[str, Any]:
    return {
        "filter_model": filter_model,
        "min_tps_matches_total": args.min_tps_matches_total,
        "min_tps_train_matches_per_fold": args.min_tps_train_matches_per_fold,
        "tps_smoothing": args.tps_smoothing,
        "tps_abs_thresh_px": args.tps_abs_thresh_px,
        "tps_mad_mult": args.tps_mad_mult,
        "tps_max_iters": args.tps_max_iters,
        "tps_max_remove_frac": args.tps_max_remove_frac,
        "cv_block_px": args.cv_block_px,
        "cv_max_folds": args.cv_max_folds,
        "ransac_reproj_thresh_px": args.ransac_reproj_thresh_px,
        "ransac_confidence": args.ransac_confidence,
        "ransac_max_iters": args.ransac_max_iters,
        "ransac_refine_iters": args.ransac_refine_iters,
    }

def is_bad_geometry(
    mr: MatchResult,
    *,
    max_median_px: float,
    max_p90_px: float,
) -> bool:
    return (
        mr.residual_median_px is not None
        and mr.residual_p90_px is not None
        and (
            mr.residual_median_px > max_median_px
            or mr.residual_p90_px > max_p90_px
        )
    )
