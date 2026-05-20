from pathlib import Path
import re
import geopandas as gpd
import rasterio
from rasterio.warp import transform_bounds
from shapely.geometry import box

BANDS = ["R", "G", "B", "N"]

def parse_scene_times(scene_id):
    ts = re.findall(r"\d{8}T\d{6}", scene_id)
    acq_start = ts[0] if len(ts) >= 1 else "unknown"
    acq_end = ts[1] if len(ts) >= 2 else "unknown"
    processing_time = ts[-1] if len(ts) >= 3 else "unknown"
    acq_day = acq_start[:8] if acq_start != "unknown" else "unknown"
    return acq_start, acq_end, acq_day, processing_time

def scene_id_from_r_path(path: Path):
    return path.stem[:-2] if path.stem.endswith("_R") else path.stem

def band_paths_from_r_path(r_path: Path):
    scene_id = scene_id_from_r_path(r_path)
    folder = r_path.parent
    out = {}

    for band in BANDS:
        p = folder / f"{scene_id}_{band}.tif"
        if not p.exists():
            return None
        out[band] = p

    return out

def build_scene_index(roots, target_crs):
    rows = []

    for root in [Path(r) for r in roots]:
        for r_path in sorted(root.rglob("*_R.tif")):
            scene_id = scene_id_from_r_path(r_path)
            band_paths = band_paths_from_r_path(r_path)

            if band_paths is None:
                continue

            acq_start, acq_end, acq_day, processing_time = parse_scene_times(scene_id)

            with rasterio.open(r_path) as src:
                if src.crs is None:
                    continue

                b = transform_bounds(
                    src.crs,
                    target_crs,
                    src.bounds.left,
                    src.bounds.bottom,
                    src.bounds.right,
                    src.bounds.top,
                    densify_pts=21,
                )

                rows.append({
                    "scene_id": scene_id,
                    "acq_start": acq_start,
                    "acq_end": acq_end,
                    "acq_day": acq_day,
                    "processing_time": processing_time,
                    "path_R": str(band_paths["R"]),
                    "path_G": str(band_paths["G"]),
                    "path_B": str(band_paths["B"]),
                    "path_N": str(band_paths["N"]),
                    "minx": float(b[0]),
                    "miny": float(b[1]),
                    "maxx": float(b[2]),
                    "maxy": float(b[3]),
                    "res_x": abs(src.transform.a),
                    "res_y": abs(src.transform.e),
                    "nodata": src.nodata,
                    "geometry": box(*b),
                })

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)
    gdf = gdf.drop_duplicates(subset=["scene_id"]).reset_index(drop=True)
    return gdf
