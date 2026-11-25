"""Sanity tests.  Run:  python -m pytest -q tests   (needs data/ from advxai.data)."""
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")

from advxai import attacks as A
from advxai import config as C
from advxai import data
from advxai import explain as E
from advxai import metrics as Mx
from advxai import model as M

EPS = C.EPS


@pytest.fixture(scope="module")
def setup():
    model = M.load_model()
    p = sorted(data.DATA_DIR.glob("*.JPEG"))[0]
    for p in sorted(data.DATA_DIR.glob("*.JPEG")):
        x = data.load_image(p)[None]
        if int(M.predict(model, x)[0]) == data.label_of(p):
            break
    x = x.to(M.DEVICE)
    y = torch.tensor([data.label_of(p)]).to(M.DEVICE)
    return model, x, y


def test_linf_attacks_respect_budget(setup):
    model, x, y = setup
    for xa in (A.fgsm(model, x, y, EPS), A.bim(model, x, y, EPS, C.BIM_ALPHA, C.BIM_STEPS),
               A.latent_pgd(model, x, y, EPS, C.LATENT_ALPHA, C.LATENT_STEPS), A.noise(x, EPS)):
        assert (xa - x).abs().max() <= EPS + 1e-6
        assert xa.min() >= 0 and xa.max() <= 1


def test_bim_fools_the_model(setup):
    model, x, y = setup
    xa = A.bim(model, x, y, EPS, C.BIM_ALPHA, C.BIM_STEPS)
    assert int(M.predict(model, xa)[0]) != int(y)


def test_noise_control_does_not_fool(setup):
    model, x, y = setup
    assert int(M.predict(model, A.noise(x, EPS))[0]) == int(y)


def test_jsma_sparse_and_increasing(setup):
    model, x, y = setup
    xa = A.jsma(model, x, y, C.JSMA_GAMMA, C.JSMA_K, C.JSMA_THETA)
    d = xa - x
    assert (d >= -1e-7).all()  # only increases values
    assert (d.abs() > 1e-8).sum() <= C.JSMA_GAMMA * x.numel() + C.JSMA_K


def test_gradcam_on_post_relu_map_equals_cam(setup):
    model, x, y = setup
    g = E.gradcam(model, x, int(y), layer="relu5")
    c = E.cam(model, x, int(y))
    assert np.allclose(g, c, atol=1e-4)


def test_maps_are_normalised(setup):
    model, x, y = setup
    for m in (E.gradcam(model, x, int(y)), E.cam(model, x, int(y))):
        assert m.shape == (224, 224) and 0 <= m.min() and m.max() <= 1 + 1e-6


def test_similarity_identity():
    m = np.random.RandomState(0).rand(224, 224).astype(np.float32)
    s = Mx.explanation_similarity(m, m)
    assert abs(s["ssim"] - 1) < 1e-6 and abs(s["spearman"] - 1) < 1e-6 and abs(s["iou"] - 1) < 1e-6


def test_input_metrics_known_values():
    x = torch.zeros(1, 3, 224, 224)
    xa = x.clone()
    xa[0, 0, 0, 0] = 0.5
    r = Mx.input_metrics(x, xa)
    assert r["l0"] == 1 and abs(r["linf"] - 0.5) < 1e-6 and abs(r["l2"] - 0.5) < 1e-6


def test_lime_fixed_segments_reuse_clean_superpixels(setup):
    model, x, y = setup
    seg = E.clean_segments(x)
    a = E.lime_map(model, x, int(y), num_samples=64, seed=0, segments=seg)
    b = E.lime_map(model, x, int(y), num_samples=64, seed=0, segments=seg)
    assert np.allclose(a, b)  # same segments + same seed -> identical map
