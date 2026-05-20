import re
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
import rasterio
from rasterio.warp import transform_bounds

SCENE_RE = re.compile(r"^(20\d{6}T\d{6}_20\d{6}T\d{6}_ARMSAT1_.*?)(?:_[BGNR])$")

def run(cmd, check=True) -> str:
    cmd = list(map(str, cmd))
    print("\n$ " + " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if check and p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{p.stdout}")
    return p.stdout

def is_1qk(p: Path) -> bool:
    s = str(p)
    return "_1QK_" in s or s.endswith("_1QK.tif") or s.endswith("_1QK.tiff")

def is_band_tif(p: Path) -> bool:
    if is_1qk(p):
        return False
    return p.stem.endswith(("_B", "_G", "_N", "_R"))

def scene_key(p: Path) -> str:
    m = SCENE_RE.match(p.stem)
    if not m:
        raise ValueError(f"Not a supported ARMSAT band filename: {p}")
    return m.group(1)

def pick_driver(paths, prefer="R") -> Path:
    paths = sorted(paths)
    for suffix in [f"_{prefer}.tif", f"_{prefer}.tiff"]:
        for p in paths:
            if str(p).endswith(suffix):
                return p
    return paths[0]

def pick_exact_band(paths, band: str) -> Path:
    matches = [p for p in paths if p.stem.endswith(f"_{band}") and p.suffix.lower() in [".tif", ".tiff"]]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly 1 {band} band, found {len(matches)}: {matches}")
    return matches[0]

def validate_same_grid(paths):
    ref = None
    ref_path = None
    for p in paths:
        with rasterio.open(p) as ds:
            cur = (ds.width, ds.height, ds.crs, ds.transform)
        if ref is None:
            ref = cur
            ref_path = p
        elif cur != ref:
            raise RuntimeError(f"Grid mismatch.\nReference: {ref_path} -> {ref}\nCurrent: {p} -> {cur}")

def get_scene_rgb_paths(scene_paths):
    r = pick_exact_band(scene_paths, "R")
    g = pick_exact_band(scene_paths, "G")
    b = pick_exact_band(scene_paths, "B")
    validate_same_grid([r, g, b])
    return r, g, b

def build_armsat_rgb(scene_paths, out_rgb_path: Path) -> tuple[Path, dict]:
    r_path, g_path, b_path = get_scene_rgb_paths(scene_paths)

    with rasterio.open(r_path) as r_ds, rasterio.open(g_path) as g_ds, rasterio.open(b_path) as b_ds:
        profile = r_ds.profile.copy()
        profile.update(count=3)

        out_rgb_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_rgb_path, "w", **profile) as dst:
            dst.write(r_ds.read(1), 1)
            dst.write(g_ds.read(1), 2)
            dst.write(b_ds.read(1), 3)
            dst.colorinterp = (
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
            )

    return out_rgb_path, {
        "armsat_rgb_native": str(out_rgb_path),
        "armsat_rgb_bands": {"R": str(r_path), "G": str(g_path), "B": str(b_path)}
    }

def parse_armsat_date(scene: str):
    m = re.match(r"^(20\d{2})(\d{2})(\d{2})T", scene)
    if not m:
        return None
    return datetime(*map(int, m.groups()))

def bbox_wgs84(tif_path: Path):
    with rasterio.open(tif_path) as ds:
        b = ds.bounds
        if ds.crs is None:
            raise RuntimeError(f"No CRS in {tif_path}")
        if ds.crs.to_epsg() == 4326:
            return (b.left, b.bottom, b.right, b.top)
        return transform_bounds(ds.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21)

def expand_bbox(b, buf_deg):
    minLon, minLat, maxLon, maxLat = b
    return (minLon - buf_deg, minLat - buf_deg, maxLon + buf_deg, maxLat + buf_deg)

def time_window(scene_dt: datetime, window_days: int):
    return (
        (scene_dt - timedelta(days=window_days)).strftime("%Y-%m-%d"),
        (scene_dt + timedelta(days=window_days)).strftime("%Y-%m-%d"),
    )

def gdal_reproject(in_path: Path, out_path: Path, epsg: int):
    run(["gdalwarp", "-t_srs", f"EPSG:{epsg}", "-overwrite", "-of", "GTiff",
         "-co", "COMPRESS=DEFLATE", in_path, out_path])

def gdal_warp_to_template_grid(in_path: Path, out_path: Path, template_tif: Path, epsg: int, dst_nodata=0):
    with rasterio.open(template_tif) as ds:
        b = ds.bounds
        t = ds.transform
        xres = abs(t.a)
        yres = abs(t.e)

    run([
        "gdalwarp",
        "-t_srs", f"EPSG:{epsg}",
        "-te", b.left, b.bottom, b.right, b.top,
        "-te_srs", f"EPSG:{epsg}",
        "-tr", xres, yres,
        "-tap",
        "-r", "bilinear",
        "-dstnodata", str(dst_nodata),
        "-overwrite",
        "-of", "GTiff",
        "-co", "COMPRESS=DEFLATE",
        in_path, out_path
    ])