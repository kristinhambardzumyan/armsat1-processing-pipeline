from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

@dataclass
class GridPair:
    mov_f32: np.ndarray
    mov_valid: np.ndarray
    ref_f32: np.ndarray
    ref_valid: np.ndarray
    mov_img01: np.ndarray
    ref_img01: np.ndarray
    mov_u8: np.ndarray
    ref_u8: np.ndarray
    ref_meta: Dict
    out_nodata: float

def _read_rgb(path: Path):
    with rasterio.open(path) as ds:
        arr = ds.read([1, 2, 3])  # (3, H, W)
        arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3)
        meta = ds.meta.copy()
        nodata = ds.nodata
        crs = ds.crs
        transform = ds.transform

    if crs is None:
        raise RuntimeError(f"No CRS in {path}")

    return arr.astype(np.float32), meta, nodata, crs, transform

def _rgb_to_gray(arr: np.ndarray) -> np.ndarray:
    # Float grayscale, no uint8 clipping
    r = arr[..., 0].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 2].astype(np.float32)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray.astype(np.float32)

def _valid_mask_2d(x_f32: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    m = np.isfinite(x_f32)
    if nodata is not None:
        m &= (x_f32 != float(nodata))
    return m

def _valid_mask_rgb(x_f32: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    m = np.isfinite(x_f32).all(axis=2)
    if nodata is not None:
        m &= (x_f32 != float(nodata)).all(axis=2)
    return m

def _cleanup_reprojected_2d(x: np.ndarray, out_nodata: float, src_nodata: Optional[float]) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x[~np.isfinite(x)] = float(out_nodata)

    if src_nodata is not None:
        x[x == float(src_nodata)] = float(out_nodata)

    # Catch broken nodata artifacts after reprojection / dtype conversion
    x[x < -1e6] = float(out_nodata)
    x[x > 1e20] = float(out_nodata)

    return x

def _cleanup_reprojected_rgb(x: np.ndarray, out_nodata: float, src_nodata: Optional[float]) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x[~np.isfinite(x)] = float(out_nodata)

    if src_nodata is not None:
        x[x == float(src_nodata)] = float(out_nodata)

    x[x < -1e6] = float(out_nodata)
    x[x > 1e20] = float(out_nodata)

    return x

def _normalize_img01_2d(x_f32: np.ndarray, m: np.ndarray) -> np.ndarray:
    v = x_f32[m]
    v = v[np.isfinite(v)]
    v = v[v > -1e6]

    if v.size < 1000:
        raise RuntimeError("Too few valid pixels to normalize.")

    p2, p98 = np.percentile(v, (2, 98))
    y = (x_f32 - p2) / (p98 - p2 + 1e-8)
    y = np.clip(y, 0.0, 1.0).astype(np.float32)
    y[~m] = 0.0
    return y

def _normalize_img01_rgb(x_f32: np.ndarray, m: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x_f32, dtype=np.float32)

    for c in range(3):
        v = x_f32[..., c][m]
        v = v[np.isfinite(v)]
        v = v[v > -1e6]

        if v.size < 1000:
            raise RuntimeError(f"Too few valid pixels to normalize channel {c}.")

        p2, p98 = np.percentile(v, (2, 98))
        y = (x_f32[..., c] - p2) / (p98 - p2 + 1e-8)
        y = np.clip(y, 0.0, 1.0).astype(np.float32)
        y[~m] = 0.0
        out[..., c] = y

    return out

def load_armsat_and_buffered_sentinel_for_matching_gray(
    armsat_path: Path,
    s2_path: Path,
    out_nodata: float = -9999.0,
) -> GridPair:
    armsat_rgb, armsat_meta, armsat_nodata, armsat_crs, armsat_transform = _read_rgb(armsat_path)
    s2_rgb, s2_meta, s2_nodata, s2_crs, s2_transform = _read_rgb(s2_path)

    mov_src = _rgb_to_gray(armsat_rgb)
    ref_f = _rgb_to_gray(s2_rgb)

    ref_valid = _valid_mask_2d(ref_f, s2_nodata)
    mov_valid_native = _valid_mask_2d(mov_src, armsat_nodata)

    mov_f = np.full(ref_f.shape, float(out_nodata), dtype=np.float32)
    mov_valid_u8 = np.zeros(ref_f.shape, dtype=np.uint8)

    reproject(
        source=mov_src,
        destination=mov_f,
        src_transform=armsat_transform,
        src_crs=armsat_crs,
        dst_transform=s2_transform,
        dst_crs=s2_crs,
        resampling=Resampling.bilinear,
        src_nodata=float(armsat_nodata) if armsat_nodata is not None else None,
        dst_nodata=float(out_nodata),
    )

    reproject(
        source=mov_valid_native.astype(np.uint8),
        destination=mov_valid_u8,
        src_transform=armsat_transform,
        src_crs=armsat_crs,
        dst_transform=s2_transform,
        dst_crs=s2_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )

    mov_f = _cleanup_reprojected_2d(mov_f, out_nodata=out_nodata, src_nodata=armsat_nodata)
    mov_valid = mov_valid_u8.astype(bool)
    mov_valid &= np.isfinite(mov_f)
    mov_valid &= (mov_f != float(out_nodata))
    mov_img01 = _normalize_img01_2d(mov_f, mov_valid)
    ref_img01 = _normalize_img01_2d(ref_f, ref_valid)
    mov_u8 = np.clip(np.round(mov_img01 * 255.0), 0, 255).astype(np.uint8)
    ref_u8 = np.clip(np.round(ref_img01 * 255.0), 0, 255).astype(np.uint8)

    return GridPair(
        mov_f32=mov_f,
        mov_valid=mov_valid,
        ref_f32=ref_f,
        ref_valid=ref_valid,
        mov_img01=mov_img01,
        ref_img01=ref_img01,
        mov_u8=mov_u8,
        ref_u8=ref_u8,
        ref_meta=s2_meta.copy(),
        out_nodata=float(out_nodata),
    )

def load_armsat_and_buffered_sentinel_for_matching_rgb(
    armsat_path: Path,
    s2_path: Path,
    out_nodata: float = -9999.0,
) -> GridPair:
    armsat_rgb, armsat_meta, armsat_nodata, armsat_crs, armsat_transform = _read_rgb(armsat_path)
    s2_rgb, s2_meta, s2_nodata, s2_crs, s2_transform = _read_rgb(s2_path)

    ref_f = s2_rgb.astype(np.float32)
    ref_valid = _valid_mask_rgb(ref_f, s2_nodata)

    mov_src = armsat_rgb.astype(np.float32)
    mov_valid_native = _valid_mask_rgb(mov_src, armsat_nodata)

    h, w, _ = ref_f.shape
    mov_f = np.full((h, w, 3), float(out_nodata), dtype=np.float32)
    mov_valid_u8 = np.zeros((h, w), dtype=np.uint8)

    for c in range(3):
        reproject(
            source=mov_src[..., c],
            destination=mov_f[..., c],
            src_transform=armsat_transform,
            src_crs=armsat_crs,
            dst_transform=s2_transform,
            dst_crs=s2_crs,
            resampling=Resampling.bilinear,
            src_nodata=float(armsat_nodata) if armsat_nodata is not None else None,
            dst_nodata=float(out_nodata),
        )

    reproject(
        source=mov_valid_native.astype(np.uint8),
        destination=mov_valid_u8,
        src_transform=armsat_transform,
        src_crs=armsat_crs,
        dst_transform=s2_transform,
        dst_crs=s2_crs,
        resampling=Resampling.nearest,
        src_nodata=0,
        dst_nodata=0,
    )

    mov_f = _cleanup_reprojected_rgb(
        mov_f,
        out_nodata=out_nodata,
        src_nodata=armsat_nodata,
    )

    mov_valid = mov_valid_u8.astype(bool)
    mov_valid &= np.isfinite(mov_f).all(axis=2)
    mov_valid &= (mov_f != float(out_nodata)).all(axis=2)
    mov_img01 = _normalize_img01_rgb(mov_f, mov_valid)
    ref_img01 = _normalize_img01_rgb(ref_f, ref_valid)
    mov_u8 = np.clip(np.round(mov_img01 * 255.0), 0, 255).astype(np.uint8)
    ref_u8 = np.clip(np.round(ref_img01 * 255.0), 0, 255).astype(np.uint8)

    return GridPair(
        mov_f32=mov_f,
        mov_valid=mov_valid,
        ref_f32=ref_f,
        ref_valid=ref_valid,
        mov_img01=mov_img01,
        ref_img01=ref_img01,
        mov_u8=mov_u8,
        ref_u8=ref_u8,
        ref_meta=s2_meta.copy(),
        out_nodata=float(out_nodata),
    )
