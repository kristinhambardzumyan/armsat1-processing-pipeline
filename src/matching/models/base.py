from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple
import cv2
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator

@dataclass
class RawMatchSet:
    pts0: np.ndarray
    pts1: np.ndarray
    conf: np.ndarray

@dataclass
class MatchResult:
    method: str
    ok: bool
    exportable: bool
    reason: Optional[str]
    match_stage: Optional[str]
    n_matches_raw: int
    n_matches_after_conf: int
    n_matches_after_spatial: int
    n_matches_inliers: int
    inlier_ratio: Optional[float]
    pts0: Optional[np.ndarray]
    pts1: Optional[np.ndarray]
    conf: Optional[np.ndarray]
    filter_model: Optional[str] = None
    n_filter_iters: Optional[int] = None
    residual_median_px: Optional[float] = None
    residual_p90_px: Optional[float] = None

class SupportsRawInference(Protocol):
    method_name: str
    def infer_raw(
        self,
        mov_img01: np.ndarray,
        ref_img01: np.ndarray,
        mov_valid: np.ndarray,
        ref_valid: np.ndarray,
        **kwargs,
    ) -> RawMatchSet:
        ...

def resize_keep_aspect(
    img: np.ndarray,
    max_side: int,
    interpolation: int,
) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    s = min(1.0, float(max_side) / float(max(h, w)))
    if s == 1.0:
        return img, 1.0
    out = cv2.resize(
        img,
        (int(round(w * s)), int(round(h * s))),
        interpolation=interpolation,
    )
    return out, float(s)

def _spatial_thin_matches(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    image_shape: Tuple[int, int],
    cell_px: int,
    max_per_cell: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = image_shape
    if pts1.shape[0] == 0:
        return pts0, pts1, conf

    order = np.argsort(-conf)
    pts0 = pts0[order]
    pts1 = pts1[order]
    conf = conf[order]
    buckets: Dict[tuple[int, int], int] = {}
    keep_idx = []

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

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    return pts0[keep_idx], pts1[keep_idx], conf[keep_idx]

def _dedup_by_moving_point_only(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    src_round_px: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(conf) == 0:
        return pts0, pts1, conf

    best: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray, float]] = {}

    for p0, p1, c in zip(pts0, pts1, conf):
        key = (
            round(float(p0[0]) / src_round_px),
            round(float(p0[1]) / src_round_px),
        )
        old = best.get(key)
        if old is None or float(c) > old[2]:
            best[key] = (p0, p1, float(c))

    vals = list(best.values())
    return (
        np.asarray([v[0] for v in vals], dtype=np.float32),
        np.asarray([v[1] for v in vals], dtype=np.float32),
        np.asarray([v[2] for v in vals], dtype=np.float32),
    )

def _dedup_match_pairs(
    pts0: np.ndarray,
    pts1: np.ndarray,
    conf: np.ndarray,
    src_round_px: float = 0.5,
    dst_round_px: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(conf) == 0:
        return pts0, pts1, conf

    best: Dict[tuple[int, int, int, int], tuple[np.ndarray, np.ndarray, float]] = {}

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
    return (
        np.asarray([v[0] for v in vals], dtype=np.float32),
        np.asarray([v[1] for v in vals], dtype=np.float32),
        np.asarray([v[2] for v in vals], dtype=np.float32),
    )

def _fit_tps_rbf(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    coord_scale: float,
    smoothing: float,
) -> RBFInterpolator:
    src_n = src_pts.astype(np.float64) / float(coord_scale)
    dst_n = dst_pts.astype(np.float64) / float(coord_scale)

    return RBFInterpolator(
        y=src_n,
        d=dst_n,
        kernel="thin_plate_spline",
        degree=1,
        smoothing=float(smoothing),
    )

def _predict_tps(
    model: RBFInterpolator,
    pts: np.ndarray,
    coord_scale: float,
) -> np.ndarray:
    pred_n = model(pts.astype(np.float64) / float(coord_scale))
    return (pred_n * float(coord_scale)).astype(np.float32)

def _make_spatial_fold_ids(
    pts_ref: np.ndarray,
    image_shape: Tuple[int, int],
    block_px: int,
    max_folds: int,
) -> tuple[Optional[np.ndarray], int]:
    h, w = image_shape

    if len(pts_ref) == 0:
        return None, 0

    block_px = max(1, int(block_px))
    max_folds = max(2, int(max_folds))

    bx = np.clip(
        (pts_ref[:, 0] // block_px).astype(np.int64),
        0,
        max(0, (w - 1) // block_px),
    )
    by = np.clip(
        (pts_ref[:, 1] // block_px).astype(np.int64),
        0,
        max(0, (h - 1) // block_px),
    )

    block_keys = np.stack([bx, by], axis=1)
    uniq, inverse, counts = np.unique(
        block_keys,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    n_blocks = int(len(uniq))

    if n_blocks < 2:
        return None, n_blocks

    n_folds = min(max_folds, n_blocks)
    order = np.argsort(counts)[::-1]

    block_to_fold = np.full(n_blocks, -1, dtype=np.int64)
    fold_sizes = np.zeros(n_folds, dtype=np.int64)

    for blk in order:
        f = int(np.argmin(fold_sizes))
        block_to_fold[blk] = f
        fold_sizes[f] += counts[blk]

    fold_ids = block_to_fold[inverse]
    return fold_ids.astype(np.int64), n_folds

def _crossfit_tps_bidirectional_residuals(
    pts0: np.ndarray,
    pts1: np.ndarray,
    *,
    image_shape: Tuple[int, int],
    block_px: int,
    max_folds: int,
    min_train_matches_per_fold: int,
    smoothing: float,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    fold_ids, n_folds = _make_spatial_fold_ids(
        pts1,
        image_shape=image_shape,
        block_px=block_px,
        max_folds=max_folds,
    )
    if fold_ids is None:
        return None, f"not_enough_spatial_blocks({n_folds})"

    coord_scale = float(max(image_shape))
    residuals = np.full(len(pts0), np.nan, dtype=np.float32)

    for f in range(n_folds):
        test = fold_ids == f
        train = ~test

        n_train = int(train.sum())
        n_test = int(test.sum())

        if n_test == 0:
            continue
        if n_train < int(min_train_matches_per_fold):
            return None, f"too_few_train_matches_in_fold({n_train})"

        fwd = _fit_tps_rbf(
            pts0[train],
            pts1[train],
            coord_scale=coord_scale,
            smoothing=smoothing,
        )
        bwd = _fit_tps_rbf(
            pts1[train],
            pts0[train],
            coord_scale=coord_scale,
            smoothing=smoothing,
        )

        pred1 = _predict_tps(fwd, pts0[test], coord_scale=coord_scale)
        pred0 = _predict_tps(bwd, pts1[test], coord_scale=coord_scale)

        res_fwd = np.linalg.norm(pred1 - pts1[test], axis=1)
        res_bwd = np.linalg.norm(pred0 - pts0[test], axis=1)
        residuals[test] = (0.5 * (res_fwd + res_bwd)).astype(np.float32)

    if not np.isfinite(residuals).all():
        return None, "non_finite_crossfit_residuals"

    return residuals, None

def _robust_sigma_from_mad(values: np.ndarray) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return 1.4826 * mad

def _robust_residual_threshold(
    residuals: np.ndarray,
    abs_thresh_px: float,
    mad_mult: float,
) -> float:
    med = float(np.median(residuals))
    sigma_hat = _robust_sigma_from_mad(residuals)
    adaptive = med + float(mad_mult) * sigma_hat
    return float(max(abs_thresh_px, adaptive))

def _point_residuals_affine(
    pts0: np.ndarray,
    pts1: np.ndarray,
    affine_2x3: np.ndarray,
) -> np.ndarray:
    pred = cv2.transform(pts0.reshape(-1, 1, 2).astype(np.float32), affine_2x3)
    pred = pred.reshape(-1, 2)
    return np.linalg.norm(pred - pts1.astype(np.float32), axis=1)

def _point_residuals_homography(
    pts0: np.ndarray,
    pts1: np.ndarray,
    H: np.ndarray,
) -> np.ndarray:
    pred = cv2.perspectiveTransform(pts0.reshape(-1, 1, 2).astype(np.float32), H)
    pred = pred.reshape(-1, 2)
    return np.linalg.norm(pred - pts1.astype(np.float32), axis=1)

class BaseMatcher:
    method_name = "base"

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def infer_raw(
        self,
        mov_img01: np.ndarray,
        ref_img01: np.ndarray,
        mov_valid: np.ndarray,
        ref_valid: np.ndarray,
        **kwargs,
    ) -> RawMatchSet:
        raise NotImplementedError

    def filter_matches(
        self,
        raw: RawMatchSet,
        *,
        image_shape: Tuple[int, int],
        conf_keep_pct: int = 60,
        min_conf: float = 0.20,
        min_matches: int = 80,
        spatial_cell_px: int = 128,
        spatial_max_per_cell: int = 8,
        min_export_matches: int = 12,
        filter_model: str = "tps",

        min_tps_matches_total: int = 12,
        min_tps_train_matches_per_fold: int = 8,
        tps_smoothing: float = 1e-3,
        tps_abs_thresh_px: float = 6.0,
        tps_mad_mult: float = 3.0,
        tps_max_iters: int = 5,
        tps_max_remove_frac: float = 0.20,
        cv_block_px: int = 256,
        cv_max_folds: int = 4,

        ransac_reproj_thresh_px: float = 8.0,
        ransac_confidence: float = 0.999,
        ransac_max_iters: int = 5000,
        ransac_refine_iters: int = 10,
        **_: object,
    ) -> MatchResult:
        filter_model = str(filter_model).lower()
        if filter_model not in {"tps", "affine_ransac", "homography_ransac"}:
            raise ValueError(
                f"Unknown filter_model={filter_model}. "
                "Use one of: tps, affine_ransac, homography_ransac."
            )

        mk0 = raw.pts0.astype(np.float32)
        mk1 = raw.pts1.astype(np.float32)
        conf = raw.conf.astype(np.float32)
        n_raw = int(mk0.shape[0])
        if n_raw < min_export_matches:
            return MatchResult(
                method=f"{self.method_name}+{filter_model}",
                ok=False,
                exportable=False,
                reason=f"too_few_matches_raw({n_raw})",
                match_stage=None,
                n_matches_raw=n_raw,
                n_matches_after_conf=0,
                n_matches_after_spatial=0,
                n_matches_inliers=0,
                inlier_ratio=None,
                pts0=None,
                pts1=None,
                conf=None,
                filter_model=filter_model,
            )

        pct_thr = np.percentile(conf, 100 - conf_keep_pct)
        keep = conf >= max(float(min_conf), float(pct_thr))
        mk0 = mk0[keep]
        mk1 = mk1[keep]
        conf = conf[keep]

        n_after_conf = int(mk0.shape[0])
        min_needed = 4 if filter_model == "homography_ransac" else 3
        min_needed = max(min_needed, int(min_export_matches))

        if n_after_conf < min_needed:
            return MatchResult(
                method=f"{self.method_name}+{filter_model}",
                ok=False,
                exportable=False,
                reason=f"too_few_matches_after_conf({n_after_conf})",
                match_stage=None,
                n_matches_raw=n_raw,
                n_matches_after_conf=n_after_conf,
                n_matches_after_spatial=0,
                n_matches_inliers=0,
                inlier_ratio=None,
                pts0=None,
                pts1=None,
                conf=None,
                filter_model=filter_model,
            )

        mk0, mk1, conf = _spatial_thin_matches(
            mk0,
            mk1,
            conf,
            image_shape=image_shape,
            cell_px=spatial_cell_px,
            max_per_cell=spatial_max_per_cell,
        )
        mk0, mk1, conf = _dedup_match_pairs(mk0, mk1, conf)

        n_after_spatial = int(mk0.shape[0])
        if n_after_spatial < min_needed:
            return MatchResult(
                method=f"{self.method_name}+{filter_model}",
                ok=False,
                exportable=False,
                reason=f"too_few_matches_after_spatial({n_after_spatial})",
                match_stage=None,
                n_matches_raw=n_raw,
                n_matches_after_conf=n_after_conf,
                n_matches_after_spatial=n_after_spatial,
                n_matches_inliers=0,
                inlier_ratio=None,
                pts0=None,
                pts1=None,
                conf=None,
                filter_model=filter_model,
            )

        if filter_model == "affine_ransac":
            model, inliers = cv2.estimateAffine2D(
                mk0,
                mk1,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(ransac_reproj_thresh_px),
                maxIters=int(ransac_max_iters),
                confidence=float(ransac_confidence),
                refineIters=int(ransac_refine_iters),
            )

            if model is None or inliers is None:
                return MatchResult(
                    method=f"{self.method_name}+affine_ransac",
                    ok=False,
                    exportable=False,
                    reason="affine_ransac_failed",
                    match_stage=None,
                    n_matches_raw=n_raw,
                    n_matches_after_conf=n_after_conf,
                    n_matches_after_spatial=n_after_spatial,
                    n_matches_inliers=0,
                    inlier_ratio=None,
                    pts0=None,
                    pts1=None,
                    conf=None,
                    filter_model="affine_ransac",
                )

            in_mask = inliers.reshape(-1).astype(bool)
            pts0_in = mk0[in_mask]
            pts1_in = mk1[in_mask]
            conf_in = conf[in_mask]
            residuals = _point_residuals_affine(pts0_in, pts1_in, model)

            pts0_in, pts1_in, conf_in = _dedup_by_moving_point_only(
                pts0_in,
                pts1_in,
                conf_in,
                src_round_px=0.25,
            )

            n_in = int(len(conf_in))
            exportable = n_in >= min_export_matches
            strong = n_in >= min_matches

            return MatchResult(
                method=f"{self.method_name}+affine_ransac",
                ok=strong and exportable,
                exportable=exportable,
                reason=None if exportable else f"too_few_affine_inliers({n_in})",
                match_stage="affine_ransac_inliers" if exportable else None,
                n_matches_raw=n_raw,
                n_matches_after_conf=n_after_conf,
                n_matches_after_spatial=n_after_spatial,
                n_matches_inliers=n_in,
                inlier_ratio=float(n_in / n_after_spatial) if n_after_spatial > 0 else None,
                pts0=pts0_in if exportable else None,
                pts1=pts1_in if exportable else None,
                conf=conf_in if exportable else None,
                filter_model="affine_ransac",
                n_filter_iters=1,
                residual_median_px=float(np.median(residuals)) if residuals.size else None,
                residual_p90_px=float(np.percentile(residuals, 90)) if residuals.size else None,
            )

        if filter_model == "homography_ransac":
            model, inliers = cv2.findHomography(
                mk0,
                mk1,
                method=cv2.RANSAC,
                ransacReprojThreshold=float(ransac_reproj_thresh_px),
                maxIters=int(ransac_max_iters),
                confidence=float(ransac_confidence),
            )

            if model is None or inliers is None:
                return MatchResult(
                    method=f"{self.method_name}+homography_ransac",
                    ok=False,
                    exportable=False,
                    reason="homography_ransac_failed",
                    match_stage=None,
                    n_matches_raw=n_raw,
                    n_matches_after_conf=n_after_conf,
                    n_matches_after_spatial=n_after_spatial,
                    n_matches_inliers=0,
                    inlier_ratio=None,
                    pts0=None,
                    pts1=None,
                    conf=None,
                    filter_model="homography_ransac",
                )

            in_mask = inliers.reshape(-1).astype(bool)
            pts0_in = mk0[in_mask]
            pts1_in = mk1[in_mask]
            conf_in = conf[in_mask]
            residuals = _point_residuals_homography(pts0_in, pts1_in, model)

            pts0_in, pts1_in, conf_in = _dedup_by_moving_point_only(
                pts0_in,
                pts1_in,
                conf_in,
                src_round_px=0.25,
            )

            n_in = int(len(conf_in))
            exportable = n_in >= min_export_matches
            strong = n_in >= min_matches

            return MatchResult(
                method=f"{self.method_name}+homography_ransac",
                ok=strong and exportable,
                exportable=exportable,
                reason=None if exportable else f"too_few_homography_inliers({n_in})",
                match_stage="homography_ransac_inliers" if exportable else None,
                n_matches_raw=n_raw,
                n_matches_after_conf=n_after_conf,
                n_matches_after_spatial=n_after_spatial,
                n_matches_inliers=n_in,
                inlier_ratio=float(n_in / n_after_spatial) if n_after_spatial > 0 else None,
                pts0=pts0_in if exportable else None,
                pts1=pts1_in if exportable else None,
                conf=conf_in if exportable else None,
                filter_model="homography_ransac",
                n_filter_iters=1,
                residual_median_px=float(np.median(residuals)) if residuals.size else None,
                residual_p90_px=float(np.percentile(residuals, 90)) if residuals.size else None,
            )
        active_idx = np.arange(n_after_spatial, dtype=np.int64)
        last_good_idx = active_idx.copy()
        last_good_residuals: Optional[np.ndarray] = None
        n_iters_run = 0
        failure_reason: Optional[str] = None

        try:
            for it in range(1, int(tps_max_iters) + 1):
                if active_idx.size < min_tps_matches_total:
                    failure_reason = f"too_few_matches_before_iteration({active_idx.size})"
                    break

                residuals, err = _crossfit_tps_bidirectional_residuals(
                    mk0[active_idx],
                    mk1[active_idx],
                    image_shape=image_shape,
                    block_px=cv_block_px,
                    max_folds=cv_max_folds,
                    min_train_matches_per_fold=min_tps_train_matches_per_fold,
                    smoothing=tps_smoothing,
                )
                if residuals is None:
                    failure_reason = f"tps_crossfit_failed:{err}"
                    break

                last_good_residuals = residuals
                tau = _robust_residual_threshold(
                    residuals,
                    abs_thresh_px=tps_abs_thresh_px,
                    mad_mult=tps_mad_mult,
                )
                keep_local = residuals <= tau
                n_iters_run = it

                if keep_local.all():
                    last_good_idx = active_idx.copy()
                    break

                bad_local = np.where(~keep_local)[0]
                max_remove = max(
                    1,
                    int(np.floor(len(active_idx) * float(tps_max_remove_frac))),
                )
                if len(bad_local) > max_remove:
                    order_bad = bad_local[np.argsort(residuals[bad_local])[::-1]]
                    drop_local = order_bad[:max_remove]
                    keep_local = np.ones(len(active_idx), dtype=bool)
                    keep_local[drop_local] = False

                next_idx = active_idx[keep_local]
                if next_idx.size < min_tps_matches_total:
                    failure_reason = f"too_few_matches_after_tps_clean({next_idx.size})"
                    break

                last_good_idx = next_idx.copy()

                if next_idx.size == active_idx.size:
                    break
                active_idx = next_idx

        except Exception as e:
            failure_reason = f"tps_filter_failed:{e}"

        if failure_reason is not None:
            return MatchResult(
                method=f"{self.method_name}+tps",
                ok=False,
                exportable=False,
                reason=failure_reason,
                match_stage=None,
                n_matches_raw=n_raw,
                n_matches_after_conf=n_after_conf,
                n_matches_after_spatial=n_after_spatial,
                n_matches_inliers=0,
                inlier_ratio=None,
                pts0=None,
                pts1=None,
                conf=None,
                filter_model="tps",
                n_filter_iters=n_iters_run,
                residual_median_px=(
                    float(np.median(last_good_residuals))
                    if last_good_residuals is not None and last_good_residuals.size
                    else None
                ),
                residual_p90_px=(
                    float(np.percentile(last_good_residuals, 90))
                    if last_good_residuals is not None and last_good_residuals.size
                    else None
                ),
            )

        pts0_in = mk0[last_good_idx]
        pts1_in = mk1[last_good_idx]
        conf_in = conf[last_good_idx]
        pts0_in, pts1_in, conf_in = _dedup_by_moving_point_only(
            pts0_in,
            pts1_in,
            conf_in,
            src_round_px=0.25,
        )
        n_in = int(len(conf_in))
        exportable = n_in >= min_export_matches
        strong = n_in >= min_matches
        return MatchResult(
            method=f"{self.method_name}+tps",
            ok=strong and exportable,
            exportable=exportable,
            reason=None if exportable else f"too_few_tps_inliers({n_in})",
            match_stage="tps_cv_inliers" if exportable else None,
            n_matches_raw=n_raw,
            n_matches_after_conf=n_after_conf,
            n_matches_after_spatial=n_after_spatial,
            n_matches_inliers=n_in,
            inlier_ratio=float(n_in / n_after_spatial) if n_after_spatial > 0 else None,
            pts0=pts0_in if exportable else None,
            pts1=pts1_in if exportable else None,
            conf=conf_in if exportable else None,
            filter_model="tps",
            n_filter_iters=n_iters_run,
            residual_median_px=(
                float(np.median(last_good_residuals))
                if last_good_residuals is not None and last_good_residuals.size
                else None
            ),
            residual_p90_px=(
                float(np.percentile(last_good_residuals, 90))
                if last_good_residuals is not None and last_good_residuals.size
                else None
            ),
        )

    def match(
        self,
        mov_img01: np.ndarray,
        ref_img01: np.ndarray,
        mov_valid: np.ndarray,
        ref_valid: np.ndarray,
        **kwargs,
    ) -> MatchResult:
        filter_model = str(kwargs.get("filter_model", "tps")).lower()
        try:
            raw = self.infer_raw(
                mov_img01,
                ref_img01,
                mov_valid,
                ref_valid,
                **kwargs,
            )
            return self.filter_matches(raw, image_shape=ref_img01.shape[:2], **kwargs)
        except Exception as e:
            return MatchResult(
                method=f"{self.method_name}+{filter_model}",
                ok=False,
                exportable=False,
                reason=f"exception:{e}",
                match_stage=None,
                n_matches_raw=0,
                n_matches_after_conf=0,
                n_matches_after_spatial=0,
                n_matches_inliers=0,
                inlier_ratio=None,
                pts0=None,
                pts1=None,
                conf=None,
                filter_model=filter_model,
            )
