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

REFERENCE_DIRECTION_OFFSETS = {
    "NW": (-1, 1), "N": (0, 1), "NE": (1, 1),
    "W": (-1, 0),                    "E": (1, 0),
    "SW": (-1, -1), "S": (0, -1), "SE": (1, -1),
}

def reference_candidate_directions(config: dict | None) -> list[str]:
    if not config:
        return ["C"]
    unknown = sorted(set(config) - {"overlap", "directions"})
    if unknown:
        raise ValueError(f"Unsupported multi_reference fields: {unknown}")
    overlap = float(config.get("overlap", -1.0))
    if not 0.0 <= overlap < 1.0:
        raise ValueError("multi_reference overlap must satisfy 0 <= overlap < 1")
    requested = config.get("directions")
    if requested is not None and not isinstance(requested, list):
        raise ValueError("multi_reference directions must be a JSON list")
    directions = list(REFERENCE_DIRECTION_OFFSETS) if requested is None else [str(x).upper() for x in requested]
    invalid = sorted(set(directions) - set(REFERENCE_DIRECTION_OFFSETS))
    if invalid:
        raise ValueError(f"Invalid multi_reference directions: {invalid}")
    return ["C"] + list(dict.fromkeys(directions))

def all_scene_reference_config(
    enabled: bool,
    overlap: float | None,
    directions: list[str] | None,
) -> dict | None:
    """Build and validate one reference configuration for every running scene."""
    if not enabled:
        if overlap is not None or directions is not None:
            raise ValueError(
                "--multi_reference_overlap and --multi_reference_directions "
                "require --multi_reference_all"
            )
        return None
    if overlap is None:
        raise ValueError("--multi_reference_all requires --multi_reference_overlap")
    config = {"overlap": float(overlap)}
    if directions is not None:
        config["directions"] = directions
    reference_candidate_directions(config)
    return config

def reference_download_bbox(center_bbox, config: dict | None, buffer_deg: float):
    """Union the requested WGS84 neighbor extents, then add the usual buffer."""
    if not config:
        return expand_bbox(center_bbox, buffer_deg)
    west, south, east, north = map(float, center_bbox)
    reference_candidate_directions(config)
    overlap = float(config["overlap"])
    step_lon = (east - west) * (1.0 - overlap)
    step_lat = (north - south) * (1.0 - overlap)
    boxes = [(west, south, east, north)]
    for direction in reference_candidate_directions(config)[1:]:
        dx, dy = REFERENCE_DIRECTION_OFFSETS[direction]
        boxes.append((
            west + dx * step_lon,
            south + dy * step_lat,
            east + dx * step_lon,
            north + dy * step_lat,
        ))
    union = (
        min(box[0] for box in boxes), min(box[1] for box in boxes),
        max(box[2] for box in boxes), max(box[3] for box in boxes),
    )
    return expand_bbox(union, buffer_deg)

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

def gdal_warp_reference_candidates(
    in_path: Path,
    out_dir: Path,
    template_tif: Path,
    epsg: int,
    config: dict,
    dst_nodata=0,
) -> list[dict]:
    """Create same-sized, shifted reference grids around the center template."""
    with rasterio.open(template_tif) as ds:
        center = ds.bounds
        width = ds.width
        height = ds.height

    overlap = float(config["overlap"])
    step_x = (center.right - center.left) * (1.0 - overlap)
    step_y = (center.top - center.bottom) * (1.0 - overlap)
    candidates = [{
        "direction": "C",
        "path": "sentinel2_rgb_utm.tif",
        "crop_bounds": [center.left, center.bottom, center.right, center.top],
    }]

    for direction in reference_candidate_directions(config)[1:]:
        dx, dy = REFERENCE_DIRECTION_OFFSETS[direction]
        bounds = (
            center.left + dx * step_x,
            center.bottom + dy * step_y,
            center.right + dx * step_x,
            center.top + dy * step_y,
        )
        out_path = out_dir / f"sentinel2_rgb_utm_ref_{direction}.tif"
        run([
            "gdalwarp", "-t_srs", f"EPSG:{epsg}",
            "-te", *bounds, "-te_srs", f"EPSG:{epsg}",
            "-ts", width, height,
            "-r", "bilinear", "-dstnodata", str(dst_nodata),
            "-overwrite", "-of", "GTiff", "-co", "COMPRESS=DEFLATE",
            in_path, out_path,
        ])
        candidates.append({
            "direction": direction,
            "path": out_path.name,
            "crop_bounds": list(bounds),
        })
    return candidates
