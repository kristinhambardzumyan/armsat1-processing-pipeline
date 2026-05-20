from pathlib import Path
import argparse
import shutil
import subprocess
import json
import numpy as np
import rasterio

def run(cmd):
    cmd = list(map(str, cmd))
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def filter_tiles(tile_dir, valid_dir, mode, tile_size, min_valid_ratio, nodata_f):
    tile_dir = Path(tile_dir)
    valid_dir = Path(valid_dir)
    valid_dir.mkdir(parents=True, exist_ok=True)

    raw_tiles = 0
    valid_tiles = 0
    rejected_shape = 0
    rejected_valid_ratio = 0

    for p in sorted(tile_dir.glob("*.tif")):
        raw_tiles += 1

        with rasterio.open(p) as src:
            arr = src.read(masked=True).astype(np.float32)

        if arr.shape != (4, tile_size, tile_size):
            rejected_shape += 1
            continue

        mask = np.ma.getmaskarray(arr)
        data = arr.filled(np.nan)

        valid = ~mask
        valid &= np.isfinite(data)
        valid &= data > -1e6

        if mode == "float32":
            valid &= data != nodata_f

        valid_ratio = float(valid.all(axis=0).mean())

        if valid_ratio < min_valid_ratio:
            rejected_valid_ratio += 1
            continue

        shutil.copy2(p, valid_dir / p.name)
        valid_tiles += 1

    return raw_tiles, valid_tiles, rejected_shape, rejected_valid_ratio

def make_float32_stack(scene, r, g, b, n, out_path, out_float_dir, nodata_f):
    band_paths = {
        "R": Path(r),
        "G": Path(g),
        "B": Path(b),
        "N": Path(n),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    band_order = ["R", "G", "B", "N"]

    arrays = []
    profile = None

    radiometry = {
        "scene": scene,
        "dataset": "Armsat-1-2024-float32",
        "band_order": band_order,
        "classifier_dtype": "float32",
        "classifier_nodata": nodata_f,
        "conversion": "raw_uint16 * scale + offset; masked/NoData -> -9999",
        "bands": {},
    }

    for band in band_order:
        p = band_paths[band]

        with rasterio.open(p) as src:
            arr = src.read(1, masked=True)

            dtype = src.dtypes[0]
            nodata = src.nodatavals[0]
            scale = src.scales[0] if src.scales[0] is not None else 1.0
            offset = src.offsets[0] if src.offsets[0] is not None else 0.0

            raw = arr.filled(0).astype(np.float32)
            data = raw * np.float32(scale) + np.float32(offset)

            mask = np.ma.getmaskarray(arr)
            data[mask] = nodata_f

            arrays.append(data.astype(np.float32))

            radiometry["bands"][band] = {
                "source_path": str(p),
                "source_dtype": dtype,
                "source_nodata": None if nodata is None else float(nodata),
                "source_scale": float(scale),
                "source_offset": float(offset),
            }

            if profile is None:
                profile = src.profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=4,
        nodata=nodata_f,
        tiled=True,
        bigtiff="IF_SAFER",
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        for idx, data in enumerate(arrays, start=1):
            dst.write(data, idx)
            dst.set_band_description(idx, band_order[idx - 1])

    json_path = Path(out_float_dir) / "scene_radiometry.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(radiometry, f, indent=2)

def process_scene(
    scene,
    dataset,
    input_root,
    out_root,
    tmp_root,
    report_csv,
    tile_size,
    overlap,
    min_valid_ratio,
    nodata_f,
):
    scene_dir = input_root / scene

    print(f"Processing {dataset} {scene}", flush=True)

    r = scene_dir / f"{scene}_R.tif"
    g = scene_dir / f"{scene}_G.tif"
    b = scene_dir / f"{scene}_B.tif"
    n = scene_dir / f"{scene}_N.tif"

    if not r.exists() or not g.exists() or not b.exists() or not n.exists():
        print(f"[SKIP] Missing bands: {scene}", flush=True)
        return

    scene_tmp = tmp_root / scene
    if scene_tmp.exists():
        shutil.rmtree(scene_tmp)
    scene_tmp.mkdir(parents=True, exist_ok=True)

    if dataset == "Armsat-1-2023":
        out_dataset_dir = out_root / "Armsat-1-2023" / scene
        raw_tile_dir = scene_tmp / "raw_tiles"

        raw_tile_dir.mkdir(parents=True, exist_ok=True)
        out_dataset_dir.mkdir(parents=True, exist_ok=True)

        vrt = scene_tmp / f"{scene}_RGBN.vrt"
        stack = scene_tmp / f"{scene}_RGBN.tif"

        run(["gdalbuildvrt", "-q", "-separate", vrt, r, g, b, n])

        run([
            "gdal_translate", "-q", vrt, stack,
            "-co", "TILED=YES",
            "-co", "BIGTIFF=IF_SAFER",
        ])

        run([
            "gdal_retile.py",
            "-ps", tile_size, tile_size,
            "-overlap", overlap,
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", "BIGTIFF=IF_SAFER",
            "-targetDir", raw_tile_dir,
            stack,
        ])

        raw, valid, rejected_shape, rejected_ratio = filter_tiles(
            raw_tile_dir,
            out_dataset_dir,
            "native",
            tile_size,
            min_valid_ratio,
            nodata_f,
        )

        with report_csv.open("a", encoding="utf-8") as f:
            f.write(
                f"{dataset},{scene},{raw},{valid},"
                f"{rejected_shape},{rejected_ratio},{min_valid_ratio}\n"
            )

    else:
        out_native_dir = out_root / "Armsat-1-2024" / scene
        out_float_dir = out_root / "Armsat-1-2024-float32" / scene

        native_raw = scene_tmp / "native_raw"
        float_raw = scene_tmp / "float32_raw"

        native_raw.mkdir(parents=True, exist_ok=True)
        float_raw.mkdir(parents=True, exist_ok=True)
        out_native_dir.mkdir(parents=True, exist_ok=True)
        out_float_dir.mkdir(parents=True, exist_ok=True)

        native_vrt = scene_tmp / f"{scene}_RGBN_native.vrt"
        native_stack = scene_tmp / f"{scene}_RGBN_native.tif"

        run(["gdalbuildvrt", "-q", "-separate", native_vrt, r, g, b, n])

        run([
            "gdal_translate", "-q", native_vrt, native_stack,
            "-co", "TILED=YES",
            "-co", "BIGTIFF=IF_SAFER",
        ])

        run([
            "gdal_retile.py",
            "-ps", tile_size, tile_size,
            "-overlap", overlap,
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", "BIGTIFF=IF_SAFER",
            "-targetDir", native_raw,
            native_stack,
        ])

        float_stack = scene_tmp / f"{scene}_RGBN_float32.tif"

        make_float32_stack(
            scene=scene,
            r=r,
            g=g,
            b=b,
            n=n,
            out_path=float_stack,
            out_float_dir=out_float_dir,
            nodata_f=nodata_f,
        )

        run([
            "gdal_retile.py",
            "-ps", tile_size, tile_size,
            "-overlap", overlap,
            "-of", "GTiff",
            "-co", "TILED=YES",
            "-co", "BIGTIFF=IF_SAFER",
            "-targetDir", float_raw,
            float_stack,
        ])

        native_raw_n, native_valid, native_rejected_shape, native_rejected_ratio = filter_tiles(
            native_raw,
            out_native_dir,
            "native",
            tile_size,
            min_valid_ratio,
            nodata_f,
        )

        float_raw_n, float_valid, float_rejected_shape, float_rejected_ratio = filter_tiles(
            float_raw,
            out_float_dir,
            "float32",
            tile_size,
            min_valid_ratio,
            nodata_f,
        )

        with report_csv.open("a", encoding="utf-8") as f:
            f.write(
                f"Armsat-1-2024,{scene},{native_raw_n},{native_valid},"
                f"{native_rejected_shape},{native_rejected_ratio},{min_valid_ratio}\n"
            )
            f.write(
                f"Armsat-1-2024-float32,{scene},{float_raw_n},{float_valid},"
                f"{float_rejected_shape},{float_rejected_ratio},{min_valid_ratio}\n"
            )

    shutil.rmtree(scene_tmp)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chunk_file")
    ap.add_argument("--out_root", default="outputs/patch_classifier/datasets")
    ap.add_argument("--tile_size", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--min_valid_ratio", type=float, default=0.99)
    ap.add_argument("--nodata_f", type=float, default=-9999.0)
    args = ap.parse_args()

    chunk_file = Path(args.chunk_file)

    if not chunk_file.exists():
        raise RuntimeError(f"Chunk file not found: {chunk_file}")

    chunk_name = chunk_file.stem

    if chunk_name.startswith("Armsat-1-2023"):
        dataset = "Armsat-1-2023"
        input_root = Path("outputs/georeferencing/2023_all")
    elif chunk_name.startswith("Armsat-1-2024"):
        dataset = "Armsat-1-2024"
        input_root = Path("outputs/georeferencing/2024_all")
    else:
        raise RuntimeError(f"Unknown chunk name: {chunk_name}")

    out_root = Path(args.out_root)
    tmp_root = out_root / "tmp" / chunk_name
    report_dir = out_root / "reports"
    report_csv = report_dir / f"{chunk_name}_tile_report.csv"

    tmp_root.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    with report_csv.open("w", encoding="utf-8") as f:
        f.write(
            "dataset,scene,raw_tiles,valid_tiles,"
            "rejected_shape,rejected_valid_ratio,min_valid_ratio\n"
        )

    with chunk_file.open("r", encoding="utf-8") as f:
        scenes = [line.strip() for line in f if line.strip()]

    for scene in scenes:
        process_scene(
            scene=scene,
            dataset=dataset,
            input_root=input_root,
            out_root=out_root,
            tmp_root=tmp_root,
            report_csv=report_csv,
            tile_size=args.tile_size,
            overlap=args.overlap,
            min_valid_ratio=args.min_valid_ratio,
            nodata_f=args.nodata_f,
        )

    print(f"DONE chunk: {chunk_name}")
    print(f"Report: {report_csv}")

if __name__ == "__main__":
    main()