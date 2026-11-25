"""Build deck/Adversarial_XAI_deck.pptx.   python -m deck.build_deck"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import content_refs as refs
from . import content_results as res
from . import content_static as st
from .lib import new_deck

OUT = Path("misc/Adversarial_XAI_deck.pptx")  # generated document, git-ignored
R = Path("results")


def make_hero():
    """Small picture for the title slide: clean vs BIM-attacked image with Grad-CAM."""
    ex = json.loads((R / "qualitative_examples.json").read_text())
    i = ex["indices"][0]
    z = np.load(R / "raw" / f"{i:03d}.npz")
    xc = z["x_clean"].astype(np.float32).transpose(1, 2, 0)
    xa = z["x_bim"].astype(np.float32).transpose(1, 2, 0)
    fig, ax = plt.subplots(2, 2, figsize=(4.05, 4.4), gridspec_kw={"hspace": 0.12, "wspace": 0.04})
    for r, (img, key, lab) in enumerate([(xc, "clean", "Clean"), (xa, "bim", "BIM attack")]):
        ax[r, 0].imshow(img)
        ax[r, 0].set_title(lab, fontsize=13, family="serif")
        ax[r, 1].imshow(img)
        ax[r, 1].imshow(z[f"heat_{key}_gradcam"].astype(np.float32), cmap="inferno", alpha=0.55, vmin=0, vmax=1)
        ax[r, 1].set_title("Grad-CAM", fontsize=13, family="serif")
    for a in ax.ravel():
        a.axis("off")
    p = Path("deck/_eq/hero.png")
    fig.savefig(p, dpi=220, bbox_inches="tight", transparent=False, facecolor="white")
    plt.close(fig)
    return p


def build():
    prs = new_deck()
    has_results = (R / "summary.json").exists()
    hero = make_hero() if has_results else None
    st.title_slide(prs, hero)
    st.contents(prs)
    st.problem(prs)
    st.glossary(prs)
    st.objective(prs)
    st.hypotheses(prs)
    st.threat_model(prs)
    st.lit_timeline(prs)
    st.lit_attacks(prs)
    st.lit_xai(prs)
    st.lit_xai_attacks(prs)
    st.lit_findings(prs)
    st.lit_defences(prs)
    st.lit_eval(prs)
    st.gap(prs)
    st.pipeline(prs)
    st.notation(prs)
    st.setup(prs)
    st.attacks(prs)
    st.xai(prs)
    st.metrics(prs)
    st.explanation_metrics(prs)
    if has_results:
        res.res_attacks(prs)
        for a in res.ATT:
            res.res_qual(prs, a)
        res.res_expl_bars(prs)
        res.res_expl_table(prs)
        res.res_stats_noise(prs)
        res.res_stats_methods(prs)
        res.res_sweep(prs)
        res.res_supp(prs)
        res.discussion(prs)
    res.limitations(prs)
    res.next_steps(prs)
    refs.refs(prs)
    prs.save(OUT)
    print("saved", OUT, len(prs.slides), "slides")


if __name__ == "__main__":
    build()
