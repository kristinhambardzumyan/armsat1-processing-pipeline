from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, Tuple
import gc
import cv2
import numpy as np
import torch
from matching.models.base import BaseMatcher, MatchResult, RawMatchSet

class RoMaMatcher(BaseMatcher):
    method_name = "roma"

    def __init__(
        self,
        variant: str = "outdoor",
        sample_num: int = 10000,
        sample_thresh: Optional[float] = None,
        device: Optional[str] = None,
        w_resized: Optional[int] = None,
        h_resized: Optional[int] = None,
        upsample_res: Optional[Tuple[int, int]] = None,
    ) -> None:
        super().__init__()

        if device is not None:
            self.device = torch.device(device)

        from romatch import roma_indoor, roma_outdoor

        factories = {
            "outdoor": roma_outdoor,
            "indoor": roma_indoor,
        }
        if variant not in factories:
            raise ValueError(f"Unknown RoMa variant: {variant}")

        self.roma_model = factories[variant](device=self.device)

        if w_resized is not None:
            self.roma_model.w_resized = int(w_resized)

        if h_resized is not None:
            self.roma_model.h_resized = int(h_resized)

        if upsample_res is not None:
            self.roma_model.upsample_res = tuple(map(int, upsample_res))

        print(
            "FINAL MODEL SIZES:",
            self.roma_model.w_resized,
            self.roma_model.h_resized,
            self.roma_model.upsample_res,
            flush=True,
        )

        self.sample_num = int(sample_num)
        self.sample_thresh = sample_thresh

        if sample_thresh is not None and hasattr(self.roma_model, "sample_thresh"):
            self.roma_model.sample_thresh = float(sample_thresh)

    @staticmethod
    def _cleanup_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            torch.cuda.synchronize()

    @staticmethod
    def _to_rgb_u8(img01: np.ndarray, valid: np.ndarray) -> np.ndarray:
        x = np.clip(np.round(img01 * 255.0), 0, 255).astype(np.uint8)
        if x.ndim == 2:
            x = np.repeat(x[..., None], 3, axis=2)
        valid_mask = valid if valid.ndim == 2 else valid[..., 0]
        x[~valid_mask] = 0
        return x

    def infer_raw(
        self,
        mov_img01: np.ndarray,
        ref_img01: np.ndarray,
        mov_valid: np.ndarray,
        ref_valid: np.ndarray,
        sample_num: Optional[int] = None,
        **_: object,
    ) -> RawMatchSet:
        h0, w0 = mov_img01.shape[:2]
        h1, w1 = ref_img01.shape[:2]

        mov_rgb = self._to_rgb_u8(mov_img01, mov_valid)
        ref_rgb = self._to_rgb_u8(ref_img01, ref_valid)

        pts0: np.ndarray
        pts1: np.ndarray
        conf: np.ndarray

        with TemporaryDirectory(prefix="roma_match_") as td:
            td_path = Path(td)
            mov_path = td_path / "mov.png"
            ref_path = td_path / "ref.png"

            if not cv2.imwrite(str(mov_path), cv2.cvtColor(mov_rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to write temporary RoMa input: {mov_path}")

            if not cv2.imwrite(str(ref_path), cv2.cvtColor(ref_rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to write temporary RoMa input: {ref_path}")

            warp = certainty = matches = certainty_s = kpts0 = kpts1 = None

            try:
                with torch.inference_mode():
                    warp, certainty = self.roma_model.match(
                        str(mov_path),
                        str(ref_path),
                        device=self.device,
                    )

                    matches, certainty_s = self.roma_model.sample(
                        warp,
                        certainty,
                        num=sample_num if sample_num is not None else self.sample_num,
                    )

                    kpts0, kpts1 = self.roma_model.to_pixel_coordinates(
                        matches,
                        h0,
                        w0,
                        h1,
                        w1,
                    )

                    # Move outputs to CPU before leaving inference mode.
                    pts0 = kpts0.detach().cpu().numpy().astype(np.float32, copy=False)
                    pts1 = kpts1.detach().cpu().numpy().astype(np.float32, copy=False)
                    conf = certainty_s.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)

            finally:
                del warp, certainty, matches, certainty_s, kpts0, kpts1
                self._cleanup_cuda()

        # Release large CPU image buffers after tiled inference.
        del mov_rgb, ref_rgb

        if pts0.shape[0] != pts1.shape[0] or pts0.shape[0] != conf.shape[0]:
            raise RuntimeError(
                f"RoMa output shape mismatch: pts0={pts0.shape}, pts1={pts1.shape}, conf={conf.shape}"
            )

        valid = (
            np.isfinite(pts0).all(axis=1)
            & np.isfinite(pts1).all(axis=1)
            & np.isfinite(conf)
            & (pts0[:, 0] >= 0.0)
            & (pts0[:, 0] < float(w0))
            & (pts0[:, 1] >= 0.0)
            & (pts0[:, 1] < float(h0))
            & (pts1[:, 0] >= 0.0)
            & (pts1[:, 0] < float(w1))
            & (pts1[:, 1] >= 0.0)
            & (pts1[:, 1] < float(h1))
        )

        pts0 = pts0[valid]
        pts1 = pts1[valid]
        conf = conf[valid]

        if len(conf) == 0:
            return RawMatchSet(
                pts0=np.empty((0, 2), dtype=np.float32),
                pts1=np.empty((0, 2), dtype=np.float32),
                conf=np.empty((0,), dtype=np.float32),
            )

        return RawMatchSet(pts0=pts0, pts1=pts1, conf=conf)

def create_roma_matcher(
    *,
    roma_variant: str = "outdoor",
    roma_sample_num: int = 10000,
    roma_sample_thresh: Optional[float] = None,
    roma_device: Optional[str] = None,
    roma_w_resized: Optional[int] = None,
    roma_h_resized: Optional[int] = None,
    roma_upsample_res: Optional[Tuple[int, int]] = None,
) -> BaseMatcher:
    return RoMaMatcher(
        variant=roma_variant,
        sample_num=roma_sample_num,
        sample_thresh=roma_sample_thresh,
        device=roma_device,
        w_resized=roma_w_resized,
        h_resized=roma_h_resized,
        upsample_res=roma_upsample_res,
    )

__all__ = [
    "RoMaMatcher",
    "create_roma_matcher",
    "RawMatchSet",
    "MatchResult",
]
