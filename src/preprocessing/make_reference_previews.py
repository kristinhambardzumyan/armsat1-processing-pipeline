from pathlib import Path
import argparse
import numpy as np
import rasterio
from PIL import Image


def stretch_rgb(arr, nodata=None):
    arr = arr.astype(np.float32)

    valid = np.isfinite(arr).all(axis=2)

    if nodata is not None:
        valid &= (arr != nodata).all(axis=2)

    valid &= (arr > -1e6).all(axis=2)

    out = np.zeros_like(arr, dtype=np.uint8)

    for c in range(3):
        band = arr[..., c]
        vals = band[valid]

        if vals.size < 100:
            continue

        p2, p98 = np.percentile(vals, [2, 98])

        x = (band - p2) / (p98 - p2 + 1e-8)
        x = np.clip(x, 0, 1)

        out[..., c] = (x * 255).astype(np.uint8)

    out[~valid] = 0

    return out


def read_rgb_preview(path, max_side=1600):
    with rasterio.open(path) as ds:
        arr = ds.read([1, 2, 3])
        arr = np.transpose(arr, (1, 2, 0))
        nodata = ds.nodata

    img = stretch_rgb(arr, nodata)

    pil = Image.fromarray(img)

    w, h = pil.size

    scale = min(1.0, max_side / max(w, h))

    if scale < 1.0:
        pil = pil.resize(
            (int(w * scale), int(h * scale)),
            Image.Resampling.LANCZOS,
        )

    return pil


def make_side_by_side(left, right):
    h = max(left.height, right.height)

    canvas = Image.new(
        "RGB",
        (left.width + right.width, h),
        color=(0, 0, 0),
    )

    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))

    return canvas


def find_reference_candidates(scene_dir):
    center = scene_dir / "sentinel2_rgb_utm.tif"
    candidates = []
    if center.exists():
        candidates.append(("C", center))

    for path in sorted(scene_dir.glob("sentinel2_rgb_utm_ref_*.tif")):
        direction = path.stem.removeprefix("sentinel2_rgb_utm_ref_")
        candidates.append((direction, path))
    return candidates


def load_scene_names(scenes_root, scene_list):
    if not scene_list:
        return sorted(path.name for path in scenes_root.iterdir() if path.is_dir())
    names = []
    for line in Path(scene_list).read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            names.append(Path(value.rstrip("/")).name)
    return list(dict.fromkeys(names))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--scenes_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_side", type=int, default=1600)
    ap.add_argument("--scene_list", default=None)

    args = ap.parse_args()

    scenes_root = Path(args.scenes_root)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    scene_dirs = [scenes_root / name for name in load_scene_names(scenes_root, args.scene_list)]

    for sd in scene_dirs:
        scene = sd.name

        armsat = sd / "armsat_rgb_utm.tif"
        if not armsat.exists():
            armsat = sd / "armsat_rgb_native.tif"
        references = find_reference_candidates(sd)

        if not armsat.exists() or not references:
            print(f"[SKIP] {scene}: missing tif")
            continue

        try:
            armsat_img = read_rgb_preview(armsat, args.max_side)
            for direction, sentinel in references:
                sentinel_img = read_rgb_preview(sentinel, args.max_side)
                side = make_side_by_side(armsat_img, sentinel_img)
                suffix = "" if direction == "C" else f"__ref_{direction}"
                side.save(out_dir / f"{scene}{suffix}.jpg", quality=95)
                print(f"[OK] {scene} reference={direction}")

        except Exception as e:
            print(f"[FAIL] {scene}: {e}")


if __name__ == "__main__":
    main()
