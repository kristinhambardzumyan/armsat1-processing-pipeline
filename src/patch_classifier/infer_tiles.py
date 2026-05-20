from pathlib import Path
import argparse
import shutil
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from patch_classifier.model import build_resnet50_4ch

CLASSES = ["clear", "cloud", "shadow", "snow"]
ID2LABEL = {i: c for i, c in enumerate(CLASSES)}

MEAN = np.array(
    [61.406791, 109.930943, 132.682767, 58.389750],
    dtype=np.float32,
).reshape(4, 1, 1)

STD = np.array(
    [116.960311, 104.664750, 129.848029, 66.312970],
    dtype=np.float32,
).reshape(4, 1, 1)

def collect_tiles(root: Path, datasets: set[str]) -> pd.DataFrame:
    files = []

    for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(root.rglob(ext))

    rows = []

    for p in sorted(files):
        dataset = p.parent.parent.name
        scene = p.parent.name

        if dataset not in datasets:
            continue

        rows.append(
            {
                "path": str(p),
                "dataset": dataset,
                "scene": scene,
                "tile_name": p.name,
            }
        )
    return pd.DataFrame(rows)

def load_tile(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read(masked=True).astype(np.float32)

    if arr.shape[0] != 4:
        raise RuntimeError(f"Expected 4 bands, got {arr.shape[0]}: {path}")

    if np.ma.isMaskedArray(arr):
        filled = np.empty(arr.shape, dtype=np.float32)

        for b in range(4):
            filled[b] = arr[b].filled(float(MEAN[b, 0, 0]))

        arr = filled

    for b in range(4):
        bad = ~np.isfinite(arr[b])
        arr[b][bad] = MEAN[b, 0, 0]

    arr = (arr - MEAN) / (STD + 1e-6)
    return arr.astype(np.float32)

class TileDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        path = self.df.iloc[idx]["path"]
        x = load_tile(path)

        return torch.from_numpy(x).float(), idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles_root", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--copy_tiles_by_class", action="store_true")
    args = ap.parse_args()
    tiles_root = Path(args.tiles_root)
    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = set(args.datasets)
    df = collect_tiles(tiles_root, datasets)

    print("Tiles root:", tiles_root)
    print("Datasets:", sorted(datasets))
    print("Tiles found:", len(df))

    if len(df) == 0:
        raise RuntimeError(
            f"No tiles found in {tiles_root} for datasets: {sorted(datasets)}"
        )

    print("\nTiles per dataset:")
    print(df["dataset"].value_counts())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice:", device)

    dataset = TileDataset(df)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = build_resnet50_4ch(num_classes=len(CLASSES)).to(device)
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("\nLoaded model:", model_path)
    print("Epoch:", ckpt.get("epoch"))
    print("Best macro F1:", ckpt.get("best_macro_f1"))
    rows = []
    with torch.no_grad():
        for x, indices in loader:
            x = x.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            probs_np = probs.cpu().numpy()
            preds_np = preds.cpu().numpy()
            indices_np = indices.cpu().numpy()

            for j, idx in enumerate(indices_np):
                base = df.iloc[int(idx)].to_dict()

                pred_id = int(preds_np[j])
                pred_label = ID2LABEL[pred_id]

                row = {
                    **base,
                    "pred_id": pred_id,
                    "pred_label": pred_label,
                }

                for cid, cname in enumerate(CLASSES):
                    row[f"prob_{cname}"] = float(probs_np[j, cid])

                rows.append(row)

                if args.copy_tiles_by_class:
                    src_path = Path(base["path"])

                    dst = (
                        out_dir
                        / "tiles_by_class"
                        / pred_label
                        / base["dataset"]
                        / base["scene"]
                        / src_path.name
                    )

                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst)

    pred_df = pd.DataFrame(rows)

    pred_csv = out_dir / "predictions.csv"
    pred_df.to_csv(pred_csv, index=False)

    summary = (
        pred_df["pred_label"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .reset_index()
    )

    summary.columns = ["class", "count"]
    summary["fraction"] = summary["count"] / max(1, len(pred_df))
    summary_csv = out_dir / "prediction_summary.csv"
    summary.to_csv(summary_csv, index=False)

    dataset_summary = (
        pred_df.groupby(["dataset", "pred_label"])
        .size()
        .reset_index(name="count")
    )

    dataset_summary_csv = out_dir / "prediction_summary_by_dataset.csv"
    dataset_summary.to_csv(dataset_summary_csv, index=False)

    scene_summary = (
        pred_df.groupby(["dataset", "scene", "pred_label"])
        .size()
        .reset_index(name="count")
    )

    scene_summary_csv = out_dir / "prediction_summary_by_scene.csv"
    scene_summary.to_csv(scene_summary_csv, index=False)

    print("\nSaved:")
    print(pred_csv)
    print(summary_csv)
    print(dataset_summary_csv)
    print(scene_summary_csv)
    print("\nOverall summary:")
    print(summary)

if __name__ == "__main__":
    main()