"""Run the benchmark: 5 conditions x 4 explanation methods on N images.

For each correctly-classified image x with true class c:
  1. explain x with Grad-CAM, CAM, LIME, SHAP (class c)             -> clean maps
  2. for each condition in {noise control, FGSM, BIM, JSMA, latent PGD}:
        build x', record input metrics, explain x' for the SAME class c,
        compare each map with its clean counterpart.
  3. noise floor: LIME on the clean image again with a new random seed.
  4. epsilon sweep: BIM at several budgets, Grad-CAM / CAM only (cheap).

Results are written per image to results/raw/ so the run can be resumed and
split over several processes:   python run_experiment.py --part 0 --parts 2
"""
from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from advxai import attacks as A
from advxai import config as C
from advxai import data
from advxai import explain as E
from advxai import metrics as Mx
from advxai import model as M

RAW = Path("results/raw")


def select_images(model, n: int) -> list[Path]:
    paths = sorted(data.DATA_DIR.glob("*.JPEG"))
    random.Random(C.SEED).shuffle(paths)
    keep = []
    for p in paths:
        x = data.load_image(p)[None]
        pred, _ = M.predict(model, x)
        if int(pred) == data.label_of(p):
            keep.append(p)
        if len(keep) == n:
            break
    return keep


def make_attack(name: str, model, x, y, seed: int):
    if name == "noise":
        return A.noise(x, C.EPS, seed)
    if name == "fgsm":
        return A.fgsm(model, x, y, C.EPS)
    if name == "bim":
        return A.bim(model, x, y, C.EPS, C.BIM_ALPHA, C.BIM_STEPS)
    if name == "jsma":
        return A.jsma(model, x, y, C.JSMA_GAMMA, C.JSMA_K, C.JSMA_THETA)
    if name == "latent":
        return A.latent_pgd(model, x, y, C.EPS, C.LATENT_ALPHA, C.LATENT_STEPS, seed)
    raise ValueError(name)


def explain_all(model, x, c: int, segments=None) -> dict[str, np.ndarray]:
    out = {
        "gradcam": E.gradcam(model, x, c),
        "cam": E.cam(model, x, c),
        "gradcam_b3": E.gradcam(model, x, c, layer=C.GRADCAM_LAYER_MID),
        "lime": E.lime_map(model, x, c, seed=0),
        "shap": E.shap_map(model, x, c),
    }
    if segments is not None:  # attacked image, but LIME forced to use the CLEAN superpixels
        out["lime_fixed"] = E.lime_map(model, x, c, seed=0, segments=segments)
    return out


def process(model, idx: int, path: Path) -> None:
    out_json, out_npz = RAW / f"{idx:03d}.json", RAW / f"{idx:03d}.npz"
    if out_json.exists():
        return
    t0 = time.time()
    x = data.load_image(path)[None].to(M.DEVICE)
    c = data.label_of(path)
    y = torch.tensor([c]).to(M.DEVICE)
    _, clean_conf = M.predict(model, x)

    arrays: dict[str, np.ndarray] = {"x_clean": x[0].cpu().numpy().astype(np.float16)}
    rows_in, rows_ex, rows_sweep = [], [], []

    clean = explain_all(model, x, c)
    clean_seg = E.clean_segments(x)
    clean["lime_fixed"] = clean["lime"]  # same segments, same seed: identical to clean LIME
    for k, v in clean.items():
        arrays[f"heat_clean_{k}"] = v.astype(np.float16)
    # noise floor: LIME re-run on the CLEAN image with another seed
    floor = E.lime_map(model, x, c, seed=1)
    arrays["heat_clean_lime_rerun"] = floor.astype(np.float16)
    rows_ex.append(dict(cond="clean_rerun", method="lime", target="orig",
                        **Mx.explanation_similarity(clean["lime"], floor)))

    for cond in C.ATTACKS:
        xa = make_attack(cond, model, x, y, seed=idx)
        pred, conf = M.predict(model, xa)
        row = dict(cond=cond, adv_pred=int(pred), adv_conf=float(conf), clean_conf=float(clean_conf),
                   success=int(int(pred) != c), **Mx.input_metrics(x, xa))
        rows_in.append(row)
        arrays[f"x_{cond}"] = xa[0].cpu().numpy().astype(np.float16)

        adv = explain_all(model, xa, c, segments=clean_seg)  # same class c as the clean explanation
        for k, v in adv.items():
            arrays[f"heat_{cond}_{k}"] = v.astype(np.float16)
            rows_ex.append(dict(cond=cond, method=k, target="orig",
                                **Mx.explanation_similarity(clean[k], v)))
        # what a user would actually see: explain the (new) predicted class
        if int(pred) != c:
            for k, fn in (("gradcam", E.gradcam), ("cam", E.cam)):
                v = fn(model, xa, int(pred))
                arrays[f"heat_{cond}_{k}_advcls"] = v.astype(np.float16)
                rows_ex.append(dict(cond=cond, method=k, target="adv",
                                    **Mx.explanation_similarity(clean[k], v)))

    for eps in C.SWEEP_EPS:  # strength sweep  I_0 -> I_n
        xa = A.bim(model, x, y, eps, eps / 4, C.BIM_STEPS)
        pred, _ = M.predict(model, xa)
        r = dict(eps=eps * 255, success=int(int(pred) != c), ssim_img=Mx.input_metrics(x, xa)["ssim_img"])
        for k, fn in (("gradcam", E.gradcam), ("cam", E.cam)):
            r[f"{k}_ssim"] = Mx.explanation_similarity(clean[k], fn(model, xa, c))["ssim"]
        rows_sweep.append(r)

    np.savez_compressed(out_npz, **arrays)
    for r in rows_in + rows_ex + rows_sweep:
        r["image"] = path.name
        r["true_class"] = c
    out_json.write_text(json.dumps({"image": path.name, "input": rows_in, "expl": rows_ex,
                                    "sweep": rows_sweep, "seconds": time.time() - t0}))
    print(f"[{idx:03d}] {path.name} done in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=C.N_IMAGES)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    RAW.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(C.SEED)
    model = M.load_model()
    paths = select_images(model, args.n)
    (RAW.parent / "selected_images.json").write_text(json.dumps([p.name for p in paths]))
    for idx, p in enumerate(paths):
        if idx % args.parts == args.part:
            process(model, idx, p)


if __name__ == "__main__":
    main()
