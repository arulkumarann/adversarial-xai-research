"""Aggregate results/raw -> tables (csv), statistics, summary.json and figures.

    python analyze.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

RAW, OUT, FIG = Path("results/raw"), Path("results"), Path("results/figures")
FIG.mkdir(parents=True, exist_ok=True)

ATTACKS = ["noise", "fgsm", "bim", "jsma", "latent"]
ATTACK_LABEL = {"noise": "Random noise (control)", "fgsm": "FGSM", "bim": "BIM", "jsma": "JSMA",
                "latent": "Latent-feature PGD"}
METHODS = ["gradcam", "cam", "lime", "lime_fixed", "shap"]
METHOD_LABEL = {"gradcam": "Grad-CAM", "cam": "CAM", "lime": "LIME", "lime_fixed": "LIME (fixed superpixels)",
                "shap": "SHAP", "gradcam_b3": "Grad-CAM (dense block 3)"}
# shades of black / grey used throughout (white background)
SHADES = {"gradcam": "#111111", "cam": "#4d4d4d", "lime": "#8c8c8c", "lime_fixed": "#b3b3b3", "shap": "#dcdcdc"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 13, "axes.edgecolor": "#222222", "axes.labelcolor": "#111111",
    "xtick.color": "#111111", "ytick.color": "#111111", "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 220, "savefig.bbox": "tight",
})


# ------------------------------------------------------------------ loading
def load():
    inp, ex, sw = [], [], []
    for f in sorted(RAW.glob("*.json")):
        d = json.loads(f.read_text())
        inp += d["input"]
        ex += d["expl"]
        sw += d["sweep"]
    return pd.DataFrame(inp), pd.DataFrame(ex), pd.DataFrame(sw)


def boot_ci(v, n=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    m = rng.choice(v, (n, len(v)), replace=True).mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    out = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        out[i] = min(1.0, run)
    return out


# ------------------------------------------------------------------- tables
def tables(inp, ex, sw):
    summ = {"n_images": int(inp["image"].nunique())}

    # attack table
    rows = []
    for a in ATTACKS:
        d = inp[inp.cond == a]
        rows.append({
            "attack": ATTACK_LABEL[a], "success_rate": d.success.mean(),
            "linf": d.linf.mean(), "l2": d.l2.mean(), "l0_values": d.l0.mean(),
            "pixels_changed_%": 100 * d.px_changed_frac.mean(), "ssim_img": d.ssim_img.mean(),
            "psnr_db": d.psnr.replace(np.inf, np.nan).mean(), "mean_adv_conf": d.adv_conf.mean(),
        })
    t_att = pd.DataFrame(rows)
    t_att.to_csv(OUT / "table_attacks.csv", index=False)

    # explanation table (original class, all attacks)
    exo = ex[(ex.target == "orig") & ex.cond.isin(ATTACKS)]
    rows = []
    for a in ATTACKS:
        for m in METHODS + ["gradcam_b3"]:
            d = exo[(exo.cond == a) & (exo.method == m)]
            if d.empty:
                continue
            lo, hi = boot_ci(d.ssim)
            rows.append({"attack": a, "method": m, "n": len(d), "ssim": d.ssim.mean(),
                         "ssim_ci_lo": lo, "ssim_ci_hi": hi, "spearman": d.spearman.mean(),
                         "iou_top20": d.iou.mean()})
    t_ex = pd.DataFrame(rows)
    t_ex.to_csv(OUT / "table_explanations.csv", index=False)

    # successful attacks only
    succ = inp[["image", "cond", "success"]]
    exs = exo.merge(succ, left_on=["image", "cond"], right_on=["image", "cond"])
    exs = exs[exs.success == 1]
    t_succ = (exs.groupby(["cond", "method"]).agg(n=("ssim", "size"), ssim=("ssim", "mean"),
                                                  spearman=("spearman", "mean"), iou_top20=("iou", "mean"))
              .reset_index())
    t_succ.to_csv(OUT / "table_explanations_success_only.csv", index=False)

    # target = adversarial class (what a user would see)
    exa = ex[ex.target == "adv"]
    t_adv = exa.groupby(["cond", "method"]).agg(n=("ssim", "size"), ssim=("ssim", "mean"),
                                                spearman=("spearman", "mean"),
                                                iou_top20=("iou", "mean")).reset_index()
    t_adv.to_csv(OUT / "table_explanations_advclass.csv", index=False)

    # LIME noise floor
    fl = ex[ex.cond == "clean_rerun"]
    summ["lime_noise_floor"] = {"ssim": float(fl.ssim.mean()), "spearman": float(fl.spearman.mean()),
                                "iou": float(fl.iou.mean())}

    # statistics: paired Wilcoxon, Grad-CAM vs others, per attack, Holm-corrected
    rows = []
    for a in ATTACKS[1:]:
        pw = exo[exo.cond == a].pivot(index="image", columns="method", values="ssim")
        for m in ["cam", "lime", "lime_fixed", "shap"]:
            d = (pw["gradcam"] - pw[m]).dropna()
            p = wilcoxon(d).pvalue if (len(d) > 5 and (d != 0).any()) else float("nan")
            rows.append({"attack": a, "other": m, "comparison": f"Grad-CAM vs {METHOD_LABEL[m]}",
                         "mean_diff": d.mean(), "p": p})
    t_stat = pd.DataFrame(rows)
    t_stat["p_holm"] = holm(t_stat.p.fillna(1.0).values)
    t_stat.to_csv(OUT / "stats_methods.csv", index=False)

    # statistics: each attack vs the random-noise control (per method)
    rows = []
    pw_all = exo.pivot_table(index=["image", "method"], columns="cond", values="ssim").reset_index()
    for m in METHODS:
        pm = pw_all[pw_all.method == m]
        for a in ATTACKS[1:]:
            d = (pm["noise"] - pm[a]).dropna()
            p = wilcoxon(d).pvalue if (len(d) > 5 and (d != 0).any()) else float("nan")
            rows.append({"method": m, "attack": a, "mean_drop_vs_noise": d.mean(), "p": p})
    t_noise = pd.DataFrame(rows)
    t_noise["p_holm"] = holm(t_noise.p.fillna(1.0).values)
    t_noise.to_csv(OUT / "stats_attack_vs_noise.csv", index=False)

    # sweep
    t_sw = sw.groupby("eps").agg(success=("success", "mean"), ssim_img=("ssim_img", "mean"),
                                 gradcam=("gradcam_ssim", "mean"), cam=("cam_ssim", "mean")).reset_index()
    t_sw.to_csv(OUT / "table_sweep.csv", index=False)

    # Grad-CAM vs CAM agreement on clean images
    summ["clean_images"] = int(inp["image"].nunique())
    summ["table_attacks"] = t_att.round(4).to_dict("records")
    summ["table_explanations"] = t_ex.round(4).to_dict("records")
    summ["stats_methods"] = t_stat.round(5).to_dict("records")
    summ["stats_attack_vs_noise"] = t_noise.round(5).to_dict("records")
    summ["sweep"] = t_sw.round(4).to_dict("records")
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    return t_att, t_ex, t_sw, summ


# ------------------------------------------------------------------ figures
def fig_bars(t_ex, summ):
    fig, ax = plt.subplots(figsize=(11, 4.8))
    w = 0.16
    xs = np.arange(len(ATTACKS))
    for i, m in enumerate(METHODS):
        d = t_ex[t_ex.method == m].set_index("attack").reindex(ATTACKS)
        err = np.vstack([d.ssim - d.ssim_ci_lo, d.ssim_ci_hi - d.ssim])
        ax.bar(xs + (i - 2) * w, d.ssim, w, yerr=err, color=SHADES[m], edgecolor="#111111",
               linewidth=0.6, capsize=2, label=METHOD_LABEL[m], error_kw={"elinewidth": 0.8})
    fl = summ["lime_noise_floor"]["ssim"]
    ax.axhline(fl, color="#6e6e6e", ls="--", lw=1, label=f"LIME re-run, no attack ({fl:.2f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([ATTACK_LABEL[a].replace(" (control)", "\n(control)").replace("Latent-feature PGD", "Latent-feature\nPGD") for a in ATTACKS])
    ax.set_ylabel("Explanation SSIM, clean vs. attacked\n(1 = unchanged)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.25), fontsize=11)
    fig.savefig(FIG / "fig_explanation_ssim.png")
    plt.close(fig)


def fig_sweep(t_sw):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(t_sw.eps, t_sw.gradcam, "-o", color="#111111", label="Grad-CAM SSIM")
    ax.plot(t_sw.eps, t_sw.cam, "-s", color="#777777", label="CAM SSIM")
    ax.plot(t_sw.eps, t_sw.success, "--^", color="#aaaaaa", label="Attack success rate")
    ax.set_xscale("log", base=2)
    ax.set_xticks(t_sw.eps)
    ax.set_xticklabels([f"{int(e)}/255" for e in t_sw.eps])
    ax.set_xlabel(r"BIM budget $\epsilon$ (L$_\infty$)")
    ax.set_ylabel("value (0 – 1)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(FIG / "fig_epsilon_sweep.png")
    plt.close(fig)


def _overlay(ax, img, heat, title):
    ax.imshow(img)
    if heat is not None:
        ax.imshow(heat, cmap="inferno", alpha=0.55, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def fig_qualitative(idx_list):
    """One figure per attack for each chosen image index."""
    meta = {r["image"]: r for r in (json.loads(f.read_text()) for f in sorted(RAW.glob("*.json")))}
    for f in sorted(RAW.glob("*.npz")):
        i = int(f.stem)
        if i not in idx_list:
            continue
        z = np.load(f)
        j = json.loads((RAW / f"{i:03d}.json").read_text())
        name = j["image"].rsplit(".", 1)[0].split("_", 1)[1].replace("_", " ")
        xc = z["x_clean"].astype(np.float32).transpose(1, 2, 0)
        for a in ATTACKS:
            xa = z[f"x_{a}"].astype(np.float32).transpose(1, 2, 0)
            pert = np.abs(xa - xc)
            pert = pert / max(pert.max(), 1e-8)
            row = next(r for r in j["input"] if r["cond"] == a)
            fig, axes = plt.subplots(5, 3, figsize=(7.4, 12.2))
            axes[0, 0].imshow(xc); axes[0, 0].set_title(f"Clean: {name}", fontsize=11)
            axes[0, 1].imshow(xa)
            axes[0, 1].set_title(f"{ATTACK_LABEL[a]}\nSSIM {row['ssim_img']:.2f}, L0 {row['l0']:,}", fontsize=11)
            axes[0, 2].imshow(pert); axes[0, 2].set_title("Perturbation (rescaled)", fontsize=11)
            for k, m in enumerate(["gradcam", "cam", "lime", "shap"], start=1):
                hc = z[f"heat_clean_{m}"].astype(np.float32)
                ha = z[f"heat_{a}_{m}"].astype(np.float32)
                _overlay(axes[k, 0], xc, hc, f"{METHOD_LABEL[m]} - clean")
                _overlay(axes[k, 1], xa, ha, f"{METHOD_LABEL[m]} - attacked")
                axes[k, 2].imshow(np.abs(hc - ha), cmap="gray_r", vmin=0, vmax=1)
                axes[k, 2].set_title("|difference|", fontsize=11)
            for ax in axes.ravel():
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(FIG / f"qual_{i:03d}_{a}.png")
            plt.close(fig)


def fig_deck(idx_list):
    """Landscape 3x5 figure per attack, sized for slides:
    rows = clean / attacked / |difference|, cols = image, Grad-CAM, CAM, LIME, SHAP."""
    from torchvision.models import DenseNet121_Weights

    from advxai.metrics import explanation_similarity

    cats = DenseNet121_Weights.IMAGENET1K_V1.meta["categories"]
    for f in sorted(RAW.glob("*.npz")):
        i = int(f.stem)
        if i not in idx_list:
            continue
        z = np.load(f)
        j = json.loads((RAW / f"{i:03d}.json").read_text())
        name = j["image"].rsplit(".", 1)[0].split("_", 1)[1].replace("_", " ")
        xc = z["x_clean"].astype(np.float32).transpose(1, 2, 0)
        for a in ATTACKS:
            xa = z[f"x_{a}"].astype(np.float32).transpose(1, 2, 0)
            row = next(r for r in j["input"] if r["cond"] == a)
            pert = np.abs(xa - xc)
            pert = pert / max(pert.max(), 1e-8)
            fig, axes = plt.subplots(3, 5, figsize=(13.2, 8.2), gridspec_kw={"hspace": 0.12, "wspace": 0.04})
            axes[0, 0].imshow(xc)
            axes[0, 0].set_title(f"Clean: {name}", fontsize=13)
            axes[1, 0].imshow(xa)
            axes[1, 0].set_title(f"Predicted: {cats[row['adv_pred']]} ({row['adv_conf']:.0%})", fontsize=13)
            axes[2, 0].imshow(pert)
            axes[2, 0].set_title("Perturbation (rescaled)", fontsize=13)
            for k, m in enumerate(["gradcam", "cam", "lime", "shap"], start=1):
                hc = z[f"heat_clean_{m}"].astype(np.float32)
                ha = z[f"heat_{a}_{m}"].astype(np.float32)
                sim = explanation_similarity(hc, ha)
                _overlay(axes[0, k], xc, hc, METHOD_LABEL[m])
                _overlay(axes[1, k], xa, ha, "")
                axes[2, k].imshow(np.abs(hc - ha), cmap="gray_r", vmin=0, vmax=1)
                axes[2, k].set_title(f"SSIM {sim['ssim']:.2f}", fontsize=13)
            for ax in axes.ravel():
                ax.axis("off")
            fig.savefig(FIG / f"deck_{i:03d}_{a}.png")
            plt.close(fig)


def pick_example(inp):
    """Image where all four real attacks fooled the model (for qualitative figures)."""
    ok = inp[inp.cond.isin(["fgsm", "bim", "jsma", "latent"])].groupby("image").success.min()
    imgs = list(ok[ok == 1].index)
    return imgs[:3] if imgs else list(inp.image.unique()[:2])


def main():
    inp, ex, sw = load()
    if inp.empty:
        raise SystemExit("no results yet")
    t_att, t_ex, t_sw, summ = tables(inp, ex, sw)
    # how alike are Grad-CAM (dense block 4) and CAM on CLEAN images?
    from advxai.metrics import explanation_similarity
    agree = []
    for f in sorted(RAW.glob("*.npz")):
        z = np.load(f)
        a, b = z["heat_clean_gradcam"].astype(np.float32), z["heat_clean_cam"].astype(np.float32)
        agree.append({**explanation_similarity(a, b), "pearson": float(np.corrcoef(a.ravel(), b.ravel())[0, 1])})
    agree = pd.DataFrame(agree)
    summ["gradcam_vs_cam_clean"] = agree.mean().round(4).to_dict()
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    fig_bars(t_ex, summ)
    fig_sweep(t_sw)
    names = pick_example(inp)
    sel = json.loads(Path("results/selected_images.json").read_text())
    idx = [sel.index(n) for n in names if n in sel]
    (OUT / "qualitative_examples.json").write_text(json.dumps({"images": names, "indices": idx}))
    fig_qualitative(set(idx[:2]))
    fig_deck(set(idx[:2]))
    print(t_att.round(3).to_string())
    print(t_ex.round(3).to_string())
    print(summ["lime_noise_floor"])


if __name__ == "__main__":
    main()
