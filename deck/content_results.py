"""Result slides.  Numbers come from results/*.csv; interpretation text from
deck/results_text.json (written after the numbers were inspected)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .content_static import S
from .lib import *  # noqa: F401,F403

R = Path("results")
FIG = R / "figures"
AL = {"noise": "Random noise (control)", "fgsm": "FGSM", "bim": "BIM", "jsma": "JSMA", "latent": "Latent-feature PGD"}
ML = {"gradcam": "Grad-CAM", "cam": "CAM", "lime": "LIME", "lime_fixed": "LIME (fixed)", "shap": "SHAP"}
ATT = ["noise", "fgsm", "bim", "jsma", "latent"]
MET = ["gradcam", "cam", "lime", "lime_fixed", "shap"]


def T():
    p = Path("deck/results_text.json")
    return json.loads(p.read_text(encoding="utf8")) if p.exists() else {}


def pfmt(p):
    if p != p:
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def n_images():
    return json.loads((R / "summary.json").read_text())["n_images"]


# ------------------------------------------------------------ attack table
def res_attacks(prs):
    t = pd.read_csv(R / "table_attacks.csv")
    tx = T()
    s = S(prs, "05  |  Results", tx.get("attack_title", "How strong are the attacks?"))
    rows = [["Attack", "Success", "L∞", "L2", "L0 (values)", "Pixels changed", "Image SSIM", "PSNR (dB)"]]
    for _, r in t.iterrows():
        rows.append([r.attack, f"{100 * r.success_rate:.0f}%", f"{r.linf:.3f}", f"{r.l2:.1f}", f"{r.l0_values:,.0f}",
                     f"{r['pixels_changed_%']:.1f}%", f"{r.ssim_img:.3f}", f"{r.psnr_db:.1f}"])
    table(s, MX, 1.95, CW, rows, col_w=[3.0, 1.1, 1.0, 1.0, 1.5, 1.6, 1.5, 1.3], size=14, row_h=0.56,
          align=["l"] + ["c"] * 7, bold_first_col=True)
    tb(s, MX, 5.45, CW, 1.4, tx.get("attack_takeaway", ""), size=16, color=INK, line=1.12, after=0)
    tb(s, MX, 6.62, CW, 0.3, f"Means over {n_images()} images. Success = top-1 class differs from the true class.",
       size=12, italic=True, color=G3, after=0)
    return s


# ------------------------------------------------------------- qualitative
def res_qual(prs, attack):
    ex = json.loads((R / "qualitative_examples.json").read_text())
    i = ex["indices"][0]
    f = FIG / f"deck_{i:03d}_{attack}.png"
    tx = T()
    s = S(prs, "05  |  Results", tx.get("qual_title", {}).get(attack, f"Example: {AL[attack]}"), size=26)
    pic(s, f, MX, 1.6, w=8.55, h=5.3, align="l")
    tb(s, 9.55, 1.95, W - MX - 9.55, 4.8, tx.get("qual", {}).get(attack, ""), size=15, color=G1, line=1.1, after=11)
    return s


# ------------------------------------------------------------- explanation
def res_expl_bars(prs):
    tx = T()
    s = S(prs, "05  |  Results", tx.get("ssim_title", "How much do the explanations change?"))
    pic(s, FIG / "fig_explanation_ssim.png", MX, 1.85, w=8.0, h=4.9, align="l")
    tb(s, 9.1, 2.0, W - MX - 9.1, 4.7, tx.get("ssim_takeaway", ""), size=15, color=G1, line=1.1, after=10)
    tb(s, MX, 6.72, CW, 0.28, f"Mean explanation SSIM with 95% bootstrap intervals over {n_images()} images; same class c explained on both images.",
       size=11, italic=True, color=G3, after=0)
    return s


def _cell(r, key="ssim"):
    return f"{r[key]:.2f}"


def res_expl_table(prs):
    t = pd.read_csv(R / "table_explanations.csv")
    tx = T()
    s = S(prs, "05  |  Results", tx.get("table_title", "Explanation similarity, three ways"))
    rows = [["Attack"] + [ML[m] for m in MET]]
    for a in ATT:
        row = [AL[a]]
        for m in MET:
            r = t[(t.attack == a) & (t.method == m)]
            if r.empty:
                row.append("-")
                continue
            r = r.iloc[0]
            row.append(f"{r.ssim:.2f}  [{r.ssim_ci_lo:.2f}, {r.ssim_ci_hi:.2f}]")
        rows.append(row)
    tb(s, MX, 1.8, CW, 0.3, "Explanation SSIM, mean and 95% interval", size=14, bold=True, color=G2, after=0)
    table(s, MX, 2.12, CW, rows, col_w=[2.7, 1.85, 1.85, 1.85, 1.85, 1.85], size=12, row_h=0.34,
          align=["l", "c", "c", "c", "c", "c"], bold_first_col=True)
    rows2 = [["Attack"] + [ML[m] for m in MET]]
    for a in ATT:
        row = [AL[a]]
        for m in MET:
            r = t[(t.attack == a) & (t.method == m)]
            row.append("-" if r.empty else f"{r.iloc[0].spearman:.2f}  /  {r.iloc[0].iou_top20:.2f}")
        rows2.append(row)
    tb(s, MX, 4.3, CW, 0.3, "Spearman ρ  /  top-20% IoU", size=14, bold=True, color=G2, after=0)
    table(s, MX, 4.62, CW, rows2, col_w=[2.7, 1.85, 1.85, 1.85, 1.85, 1.85], size=12, row_h=0.34,
          align=["l", "c", "c", "c", "c", "c"], bold_first_col=True)
    return s


def stars(p):
    if p != p:
        return "n/a"
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


def res_stats_noise(prs):
    sn = pd.read_csv(R / "stats_attack_vs_noise.csv")
    tx = T()
    s = S(prs, "05  |  Results", tx.get("stats_noise_title", "Does an attack move explanations more than random noise?"))
    rows = [["Attack"] + [ML[m] for m in MET]]
    for a in ATT[1:]:
        row = [AL[a]]
        for m in MET:
            r = sn[(sn.method == m) & (sn.attack == a)]
            row.append("-" if r.empty else f"{r.iloc[0].mean_drop_vs_noise:+.2f}  {stars(r.iloc[0].p_holm)}")
        rows.append(row)
    tb(s, MX, 1.85, CW, 0.3, "Mean explanation-SSIM of the noise control minus that of the attack (positive = attack moves the explanation more)",
       size=13, bold=True, color=G2, after=0)
    table(s, MX, 2.3, CW, rows, col_w=[3.0, 1.8, 1.8, 1.8, 1.8, 1.8], size=14, row_h=0.5,
          align=["l", "c", "c", "c", "c", "c"], bold_first_col=True)
    tb(s, MX, 5.0, CW, 1.6, tx.get("stats_noise_takeaway", ""), size=15, color=INK, line=1.1, after=0)
    tb(s, MX, 6.62, CW, 0.3, [{"text": "Paired Wilcoxon signed-rank tests across images, Holm-corrected.  *** p<0.001, ** p<0.01, * p<0.05, ns not significant.", "raw": True}],
       size=11, italic=True, color=G3, after=0)
    return s


def res_stats_methods(prs):
    sm = pd.read_csv(R / "stats_methods.csv")
    tx = T()
    s = S(prs, "05  |  Results", tx.get("stats_methods_title", "Is Grad-CAM really more robust than the others?"))
    others = ["cam", "lime", "lime_fixed", "shap"]
    rows = [["Attack"] + [f"vs {ML[m]}" for m in others]]
    for a in ATT[1:]:
        row = [AL[a]]
        for m in others:
            r = sm[(sm.attack == a) & (sm.other == m)]
            row.append("-" if r.empty else f"{r.iloc[0].mean_diff:+.2f}  {stars(r.iloc[0].p_holm)}")
        rows.append(row)
    tb(s, MX, 1.85, CW, 0.3, "Grad-CAM explanation-SSIM minus the other method's (positive = Grad-CAM keeps its explanation better)",
       size=13, bold=True, color=G2, after=0)
    table(s, MX, 2.3, CW, rows, col_w=[3.0, 2.2, 2.2, 2.2, 2.2], size=14, row_h=0.5,
          align=["l", "c", "c", "c", "c"], bold_first_col=True)
    tb(s, MX, 5.0, CW, 1.6, tx.get("stats_methods_takeaway", ""), size=15, color=INK, line=1.1, after=0)
    tb(s, MX, 6.62, CW, 0.3, [{"text": "Paired Wilcoxon signed-rank tests across images, Holm-corrected.  *** p<0.001, ** p<0.01, * p<0.05, ns not significant.", "raw": True}],
       size=11, italic=True, color=G3, after=0)
    return s


def res_sweep(prs):
    tx = T()
    sw = pd.read_csv(R / "table_sweep.csv")
    s = S(prs, "05  |  Results", tx.get("sweep_title", "Stronger attacks: how does explanation damage grow?"))
    pic(s, FIG / "fig_epsilon_sweep.png", MX, 1.9, w=7.0, h=4.7, align="l")
    rows = [["ε (/255)", "Success", "Grad-CAM", "CAM"]]
    for _, r in sw.iterrows():
        rows.append([f"{int(r.eps)}", f"{100 * r.success:.0f}%", f"{r.gradcam:.2f}", f"{r.cam:.2f}"])
    table(s, 8.1, 1.95, W - MX - 8.1, rows, col_w=[1, 1, 1.2, 1], size=13, row_h=0.42, align=["c"] * 4)
    tb(s, 8.1, 4.75, W - MX - 8.1, 2.0, tx.get("sweep_takeaway", ""), size=14, color=G1, line=1.08, after=0)
    return s


def res_supp(prs):
    tx = T()
    summ = json.loads((R / "summary.json").read_text())
    so = pd.read_csv(R / "table_explanations_success_only.csv")
    ad = pd.read_csv(R / "table_explanations_advclass.csv")

    def line(df, order=("fgsm", "jsma", "bim", "latent")):
        lab = {"fgsm": "FGSM", "jsma": "JSMA", "bim": "BIM", "latent": "latent"}
        d = df[df.method == "gradcam"].set_index("cond").ssim
        return "Grad-CAM SSIM: " + ", ".join(f"{lab[c]} {d[c]:.2f}" for c in order if c in d.index)

    s = S(prs, "05  |  Results", tx.get("supp_title", "Checks behind the main result"))
    g = summ.get("gradcam_vs_cam_clean", {})
    fl = summ["lime_noise_floor"]
    cards = [("Grad-CAM vs CAM on clean images", f"SSIM {g.get('ssim', float('nan')):.2f}, Pearson {g.get('pearson', float('nan')):.3f}",
              "Nearly the same map on DenseNet-121, as the theory predicts, so the two count as one family."),
             ("LIME against itself (no attack)", f"SSIM {fl['ssim']:.2f}, Spearman {fl['spearman']:.2f}",
              "Two LIME runs on the same clean image already differ; attack effects must be read against this floor."),
             ("Successful attacks only", line(so),
              "Dropping the images an attack failed on barely changes the picture."),
             ("Explaining the new (wrong) class", line(ad),
              "For BIM the original-class map collapses (0.25) but the new-class map still resembles the clean one: "
              "part of the damage is that evidence for the old class disappears.")]
    cw, ch = 5.8, 2.15
    for i, (t_, v, d) in enumerate(cards):
        r, c = divmod(i, 2)
        x, y = MX + c * (cw + 0.33), 1.95 + r * (ch + 0.25)
        card(s, x, y, cw, ch)
        tb(s, x + 0.25, y + 0.15, cw - 0.5, 0.35, t_, size=15, bold=True, after=0)
        tb(s, x + 0.25, y + 0.55, cw - 0.5, 0.6, v, size=14, bold=True, color=INK, line=1.05, after=0)
        tb(s, x + 0.25, y + 1.3, cw - 0.5, 0.8, d, size=13, color=G1, line=1.05, after=0)
    tb(s, MX, 6.62, CW, 0.3, tx.get("supp_note", ""), size=12, italic=True, color=G3, after=0)
    return s


# ------------------------------------------------------------- discussion
def discussion(prs):
    tx = T()
    s = S(prs, "06  |  Discussion", "What the paper will say")
    msgs = tx.get("messages", [])
    cw, ch = 5.8, 2.15
    for i, (t, d) in enumerate(msgs[:4]):
        r, c = divmod(i, 2)
        x, y = MX + c * (cw + 0.33), 1.95 + r * (ch + 0.25)
        card(s, x, y, cw, ch)
        tb(s, x + 0.25, y + 0.15, cw - 0.5, 0.35, f"{i + 1}.  {t}", size=17, bold=True, after=0)
        tb(s, x + 0.25, y + 0.62, cw - 0.5, 1.5, d, size=14, color=G1, line=1.07, after=0)
    tb(s, MX, 6.62, CW, 0.3, tx.get("discussion_note", ""), size=12, italic=True, color=G3, after=0)
    return s


def limitations(prs):
    s = S(prs, "06  |  Discussion", "Limitations and threats to validity")
    items = [
        ("Small sample", f"{n_images()} images, one per class, so intervals are wide and class effects are untested."),
        ("Attack fidelity", "The latent attack is LAFEAT-inspired, not LAFEAT; JSMA uses single features; no adaptive or explanation-aware attacks."),
        ("One model", "DenseNet-121 only. Grad-CAM and CAM coincide on any network ending in pooling plus a linear layer (also ResNet-50, EfficientNet); a ViT would separate them."),
        ("Sampling budgets", "LIME 300 samples and SHAP 400 evaluations (CPU limit); maps are coarse and LIME is random."),
        ("Metric choice", "SSIM between heat-maps is one view; fidelity tests such as deletion curves are not included."),
        ("Not a defence", "We measure damage; we do not propose or test a defence.")]
    cw, ch = 3.75, 2.05
    for i, (t, d) in enumerate(items):
        r, c = divmod(i, 3)
        x, y = MX + c * (cw + 0.34), 1.95 + r * (ch + 0.25)
        card(s, x, y, cw, ch)
        tb(s, x + 0.2, y + 0.15, cw - 0.4, 0.35, t, size=16, bold=True, after=0)
        tb(s, x + 0.2, y + 0.6, cw - 0.4, 1.4, d, size=13, color=G1, line=1.05, after=0)
    return s


def next_steps(prs):
    s = S(prs, "06  |  Discussion", "Next steps toward an IEEE TAI paper")
    rows = [["Step", "Why"],
            ["Scale to 500+ images and several models (ResNet-50, a ViT)", "Narrow the intervals; test across architectures (a ViT also separates Grad-CAM from CAM)"],
            ["Add Integrated Gradients and plain saliency; add PGD", "These are the most attacked methods in the literature"],
            ["Add deletion/insertion fidelity metrics (e.g. via Quantus)", "Show whether heat-map change means loss of faithfulness"],
            ["Add an explanation-targeted attack as an upper bound", "Separate side effects from deliberate manipulation"],
            ["Prepare for TAI rules: 10 pages, impact statement of 100 to 150 words, 3 to 6 keywords, anonymised text and code links",
             "Double-anonymous review; abstract at most 250 words"]]
    table(s, MX, 1.95, CW, rows, col_w=[6.6, 5.3], size=14, row_h=0.76, bold_first_col=False)
    return s
