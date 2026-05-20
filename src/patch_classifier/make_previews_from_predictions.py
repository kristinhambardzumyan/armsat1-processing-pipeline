from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

def save_rgb_preview(tile_path: str, out_path: Path):
    with rasterio.open(tile_path) as src:
        arr = src.read([1, 2, 3], masked=True).astype(np.float32)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    rgb = np.transpose(arr, (1, 2, 0))
    valid = np.isfinite(rgb).all(axis=2)
    out = np.zeros(rgb.shape, dtype=np.uint8)
    for c in range(3):
        vals = rgb[..., c][valid]

        if vals.size < 10:
            continue

        p2, p98 = np.percentile(vals, [2, 98])
        x = (rgb[..., c] - p2) / (p98 - p2 + 1e-8)
        x = np.clip(x, 0, 1)
        x[~np.isfinite(x)] = 0
        out[..., c] = (x * 255).astype(np.uint8)
    
    out[~valid] = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(out_path, quality=95)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--max_per_class",
        type=int,
        default=None,
        help="Maximum previews per predicted class. If omitted, saves all.",
    )
    args = ap.parse_args()
    predictions_csv = Path(args.predictions_csv)
    out_dir = Path(args.out_dir)
    df = pd.read_csv(predictions_csv)
    required_cols = {"path", "scene", "tile_name", "pred_label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in predictions CSV: {missing}")

    if args.max_per_class is not None:
        df = (
            df.groupby("pred_label", group_keys=False)
            .head(args.max_per_class)
            .reset_index(drop=True)
        )

    print("Predictions CSV:", predictions_csv)
    print("Previews to save:", len(df))
    print("Output dir:", out_dir)
    for _, row in df.iterrows():
        tile_path = Path(row["path"])

        year = str(row["year"]) if "year" in df.columns else "unknown_year"

        out_path = (
            out_dir
            / "previews_by_class"
            / row["pred_label"]
            / year
            / row["scene"]
            / f"{tile_path.stem}.jpg"
        )

        try:
            save_rgb_preview(str(tile_path), out_path)
        except Exception as e:
            print(f"[WARN] Preview failed: {tile_path} | {e}")

    print("Finished previews.")

if __name__ == "__main__":
    main()