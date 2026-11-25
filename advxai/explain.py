"""The four explanation methods.  Every function returns a (224,224) float32
heat-map in [0,1] explaining ONE class `c` for ONE image `x` (1,3,224,224).

    cam       Class Activation Mapping  (Zhou 2016)       - needs GAP + linear head
    gradcam   Gradient-weighted CAM     (Selvaraju 2017)  - needs gradients
    lime      Local surrogate on superpixels (Ribeiro 2016)
    shap      Shapley values with a partition explainer (Lundberg & Lee 2017)

Theory note (checked numerically in tests/test_explain.py):
for DenseNet-121 the network ends  A -> ReLU -> global-avg-pool -> linear.
Grad-CAM taken at the post-ReLU map F = relu(norm5) has weights
alpha_k = (1/Z) dy_c/dF_k = w_ck / (HW) and therefore equals CAM up to a scale.
Selvaraju et al. prove the same.  To get a genuinely different method we take
Grad-CAM at the output of the last dense block (before norm5).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .config import GRADCAM_LAYER, LIME_SAMPLES, LIME_SEGMENTS, SHAP_BLUR, SHAP_EVALS
from .model import DEVICE

# ---------------------------------------------------------------- helpers
def normalize01(m: np.ndarray) -> np.ndarray:
    m = m.astype(np.float32)
    lo, hi = float(m.min()), float(m.max())
    return np.zeros_like(m) if hi - lo < 1e-12 else (m - lo) / (hi - lo)


def _upsample(cam: torch.Tensor) -> np.ndarray:
    cam = F.interpolate(cam[None, None], size=(224, 224), mode="bilinear", align_corners=False)
    return cam[0, 0].detach().cpu().numpy()


# -------------------------------------------------------------------- CAM
@torch.no_grad()
def cam(model, x, c: int) -> np.ndarray:
    """CAM_c = ReLU( sum_k w_ck * F_k ),  F = ReLU(norm5 output)."""
    Fm = F.relu(model.feature_maps(x.to(DEVICE)))[0]  # (1024,7,7)
    w = model.classifier_weight[c]  # (1024,)
    return normalize01(_upsample(F.relu(torch.einsum("k,khw->hw", w, Fm))))


# --------------------------------------------------------------- Grad-CAM
def gradcam(model, x, c: int, layer: str = GRADCAM_LAYER) -> np.ndarray:
    """Grad-CAM_c = ReLU( sum_k alpha_k A_k ),  alpha_k = mean_ij d y_c / d A_k,ij."""
    # Model weights are frozen, so make the INPUT require grad to build a graph.
    xin = x.to(DEVICE).clone().requires_grad_(True)
    h = model.normalize(xin)
    target = None
    for name, module in model.net.features.named_children():
        h = module(h)
        if name == layer:
            h.retain_grad()
            target = h
    if layer == "relu5":  # post-ReLU final map F (used to verify Grad-CAM == CAM)
        h = F.relu(h)
        h.retain_grad()
        target = h
        logits = model.net.classifier(F.adaptive_avg_pool2d(h, 1).flatten(1))  # no 2nd ReLU
    else:
        logits = model.head(h)
    logits[0, c].backward()
    A = target[0].detach()
    alpha = target.grad[0].mean(dim=(1, 2))
    return normalize01(_upsample(F.relu((alpha[:, None, None] * A).sum(0))))


# ------------------------------------------------------------------- LIME
def _predict_fn(model, logits: bool = False, batch: int = 64):
    def fn(imgs: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch):
                t = torch.from_numpy(imgs[i : i + batch]).float().permute(0, 3, 1, 2).to(DEVICE)
                z = model(t)
                out.append((z if logits else F.softmax(z, 1)).cpu().numpy())
        return np.concatenate(out)

    return fn


def clean_segments(x, n_segments: int = LIME_SEGMENTS) -> np.ndarray:
    """SLIC superpixels of an image (computed on the CLEAN image for the fixed-segment LIME)."""
    from skimage.segmentation import slic

    img = x[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    return slic(img, n_segments=n_segments, compactness=10, start_label=0)


def lime_map(model, x, c: int, num_samples: int = LIME_SAMPLES, n_segments: int = LIME_SEGMENTS,
             seed: int = 0, segments: np.ndarray | None = None) -> np.ndarray:
    """LIME heat-map.  If `segments` is given, those superpixels are used instead of
    re-segmenting x; with the clean image's segments this isolates the effect of the
    attack on the MODEL from its effect on the segmentation."""
    from lime import lime_image
    from skimage.segmentation import slic

    img = x[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)
    explainer = lime_image.LimeImageExplainer(random_state=seed)
    if segments is not None:
        seg_fn = lambda im: segments
    else:
        seg_fn = lambda im: slic(im, n_segments=n_segments, compactness=10, start_label=0)
    exp = explainer.explain_instance(
        img, _predict_fn(model), labels=(c,), top_labels=None, num_samples=num_samples,
        batch_size=64, segmentation_fn=seg_fn, random_seed=seed,
    )
    seg = exp.segments
    heat = np.zeros(seg.shape, dtype=np.float32)
    for s, w in exp.local_exp[c]:
        heat[seg == s] = max(w, 0.0)  # positive evidence for class c
    return normalize01(heat)


# ------------------------------------------------------------------- SHAP
def shap_map(model, x, c: int, max_evals: int = SHAP_EVALS, blur: int = SHAP_BLUR) -> np.ndarray:
    import shap

    img = x[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    masker = shap.maskers.Image(f"blur({blur},{blur})", img.shape)
    explainer = shap.Explainer(_predict_fn(model, logits=True), masker, output_names=None)
    sv = explainer(img[None], max_evals=max_evals, batch_size=64, outputs=[c], silent=True)
    vals = np.asarray(sv.values)[0]  # (H,W,3,1)
    heat = vals.reshape(224, 224, 3, -1)[..., 0].sum(-1)
    return normalize01(np.maximum(heat, 0.0))


METHODS = ("gradcam", "cam", "lime", "shap")


def explain(model, x, c: int, method: str, **kw) -> np.ndarray:
    return {"gradcam": gradcam, "cam": cam, "lime": lime_map, "shap": shap_map}[method](model, x, c, **kw)
