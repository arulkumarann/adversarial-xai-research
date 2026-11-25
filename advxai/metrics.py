"""Metrics.

Input-space metrics compare a clean image x and its adversarial version x'.
Explanation metrics compare two heat-maps (224x224, values in [0,1]).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as _ssim

from .config import TOPK_FRAC


# ------------------------------------------------------------- input space
def input_metrics(x: torch.Tensor, xa: torch.Tensor) -> dict:
    """x, xa: (1,3,224,224) in [0,1]."""
    d = (xa - x).flatten().cpu()
    a, b = x[0].permute(1, 2, 0).cpu().numpy(), xa[0].permute(1, 2, 0).cpu().numpy()
    mse = float((d ** 2).mean())
    changed_px = ((xa - x).abs().amax(1) > 1e-8).float().mean().item()
    return {
        "linf": float(d.abs().max()),
        "l2": float(d.norm()),
        "l0": int((d.abs() > 1e-8).sum()),
        "px_changed_frac": changed_px,
        "mse": mse,
        "psnr": float(10 * np.log10(1.0 / mse)) if mse > 0 else float("inf"),
        "ssim_img": float(_ssim(a, b, channel_axis=2, data_range=1.0)),
    }


# ---------------------------------------------------------- explanation space
def _topk_mask(m: np.ndarray, frac: float) -> np.ndarray:
    thr = np.quantile(m, 1.0 - frac)
    return (m >= thr) & (m > 0)


def explanation_similarity(a: np.ndarray, b: np.ndarray) -> dict:
    """Similarity of two heat-maps. 1 = identical, lower = more distortion."""
    ssim = float(_ssim(a, b, data_range=1.0))
    if a.std() < 1e-9 or b.std() < 1e-9:
        rho = 0.0
    else:
        rho = float(spearmanr(a.ravel(), b.ravel())[0])
    ma, mb = _topk_mask(a, TOPK_FRAC), _topk_mask(b, TOPK_FRAC)
    union = (ma | mb).sum()
    iou = float((ma & mb).sum() / union) if union else 0.0
    return {"ssim": ssim, "spearman": rho, "iou": iou}
