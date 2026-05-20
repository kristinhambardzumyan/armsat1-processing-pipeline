from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling

def raster_shape(bounds, res_m):
    minx, miny, maxx, maxy = bounds
    width = int(round((maxx - minx) / res_m))
    height = int(round((maxy - miny) / res_m))
    transform = from_origin(minx, maxy, res_m, res_m)
    return height, width, transform

def make_valid_mask(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        nodata = src.nodata
        if nodata is None:
            valid = np.isfinite(arr)
        else:
            valid = (arr != nodata) & np.isfinite(arr)

        return valid.astype(np.uint8), src.transform, src.crs

def build_date_count(block_scenes, bounds, target_crs, res_m):
    height, width, dst_transform = raster_shape(bounds, res_m)
    date_count = np.zeros((height, width), dtype=np.uint16)

    for acq_day, day_df in block_scenes.groupby("acq_day"):
        day_valid = np.zeros((height, width), dtype=np.uint8)

        for _, row in day_df.iterrows():
            valid_src, src_transform, src_crs = make_valid_mask(Path(row["path_R"]))
            valid_dst = np.zeros((height, width), dtype=np.uint8)

            reproject(
                source=valid_src,
                destination=valid_dst,
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=dst_transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest,
                src_nodata=0,
                dst_nodata=0,
            )

            # Same-day scenes merge into one daily mask.
            day_valid = np.maximum(day_valid, valid_dst)

        date_count += day_valid.astype(np.uint16)
    return date_count, dst_transform

def save_count_and_mask(date_count, dst_transform, out_dir, block_id, target_crs, min_distinct_dates):
    out_dir.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": date_count.shape[0],
        "width": date_count.shape[1],
        "count": 1,
        "dtype": "uint16",
        "crs": target_crs,
        "transform": dst_transform,
        "nodata": 0,
        "compress": "deflate",
    }

    count_path = out_dir / f"{block_id}_date_count.tif"
    mask_path = out_dir / f"{block_id}_repeated_mask.tif"

    with rasterio.open(count_path, "w", **profile) as dst:
        dst.write(date_count, 1)

    repeated = (date_count >= min_distinct_dates).astype(np.uint8)

    mask_profile = profile.copy()
    mask_profile["dtype"] = "uint8"

    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(repeated, 1)

    return count_path, mask_path
