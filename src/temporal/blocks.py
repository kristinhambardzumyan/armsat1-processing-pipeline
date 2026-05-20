import math
import geopandas as gpd
from shapely.geometry import box

def align_down(x, step):
    return math.floor(x / step) * step

def align_up(x, step):
    return math.ceil(x / step) * step

def build_blocks(scene_gdf, block_size_m, block_margin_m, target_crs):
    minx, miny, maxx, maxy = scene_gdf.total_bounds
    minx = align_down(minx, block_size_m)
    miny = align_down(miny, block_size_m)
    maxx = align_up(maxx, block_size_m)
    maxy = align_up(maxy, block_size_m)
    rows = []
    idx = 0

    y = miny
    while y < maxy:
        x = minx
        while x < maxx:
            idx += 1

            core = box(x, y, x + block_size_m, y + block_size_m)
            expanded = box(
                x - block_margin_m,
                y - block_margin_m,
                x + block_size_m + block_margin_m,
                y + block_size_m + block_margin_m,
            )

            rows.append({
                "block_id": f"BLK_{idx:06d}",
                "core_minx": x,
                "core_miny": y,
                "core_maxx": x + block_size_m,
                "core_maxy": y + block_size_m,
                "expanded_minx": x - block_margin_m,
                "expanded_miny": y - block_margin_m,
                "expanded_maxx": x + block_size_m + block_margin_m,
                "expanded_maxy": y + block_size_m + block_margin_m,
                "geometry": core,
                "expanded_geometry": expanded,
            })

            x += block_size_m
        y += block_size_m

    return gpd.GeoDataFrame(rows, geometry="geometry", crs=target_crs)