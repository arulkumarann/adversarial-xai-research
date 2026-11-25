"""Static (result-independent) slides: problem, objective, literature, method."""
from __future__ import annotations

from .lib import *  # noqa: F401,F403


def S(prs, kicker, title, size=30):
    s = blank(prs)
    header(s, kicker, title, size=size)
    footer(s, len(prs.slides))
    return s


# =============================================================== 1. title
def title_slide(prs, hero=None):
    s = blank(prs)
    tb(s, MX, 1.35, 7.6, 0.3, "RESEARCH PROGRESS  |  NOVEMBER 2026", size=12, color=G3, after=0)
    tb(s, MX, 1.85, 7.6, 2.2, "Do explanations survive adversarial attacks?", size=46, bold=True, line=0.98, after=0)
    tb(s, MX, 4.25, 7.0, 1.3,
       "A benchmark of four attacks (FGSM, BIM, JSMA, latent-feature PGD) and four explanation "
       "methods (Grad-CAM, CAM, LIME, SHAP) on DenseNet-121 and ImageNet",
       size=18, color=G2, line=1.12, after=0)
    tb(s, MX, 6.35, 7.6, 0.6, ["Adversarial attacks on XAI: from an image-classification perspective",
                               "Code and results: branch rebuild-from-scratch"],
       size=13, color=G4, after=2)
    if hero:
        pic(s, hero, 8.55, 0.9, w=4.05, h=5.7)
    notes(s, "Overview of the project: what we want to find out, what is already known, how we test it, "
             "and what the experiments show.")
    return s


# =========================================================== 2. contents
def contents(prs):
    s = S(prs, "Agenda", "What this deck covers")
    items = [("01", "The problem", "Why explanations can be attacked"),
             ("02", "Research objective", "Questions, hypotheses, scope"),
             ("03", "Literature survey", "What exists and what is missing"),
             ("04", "Method", "Attacks, explanation methods, metrics"),
             ("05", "Experiments", "Setup, results, statistics"),
             ("06", "Discussion", "What the paper will say, limits, next steps"),
             ("07", "References", "Every source cited")]
    for i, (n, t, d) in enumerate(items):
        col, row = divmod(i, 4)
        x, y = MX + col * 6.1, 2.15 + row * 1.15
        tb(s, x, y, 0.9, 0.7, n, size=34, color=G5, bold=False, after=0)
        tb(s, x + 1.0, y + 0.02, 4.9, 0.4, t, size=22, bold=True, after=0)
        tb(s, x + 1.0, y + 0.45, 4.9, 0.4, d, size=15, color=G3, after=0)
    return s


# ============================================================= 3. problem
def problem(prs):
    s = S(prs, "01  |  The problem", "A model can be right for the wrong reasons, and an attack can change the reasons")
    # two rows of the pipeline
    def row(y, label, attacked):
        tb(s, MX, y - 0.42, 3, 0.3, label, size=14, bold=True, color=G2, after=0)
        box_text(s, MX, y, 1.55, 0.8, "Input image" if not attacked else "Image + tiny noise", size=14, fill=G7)
        arrow(s, MX + 1.55, y + 0.4, MX + 2.05, y + 0.4)
        box_text(s, MX + 2.05, y, 1.65, 0.8, "Classifier f", size=14, fill=G6)
        arrow(s, MX + 3.7, y + 0.4, MX + 4.2, y + 0.4)
        box_text(s, MX + 4.2, y, 2.05, 0.8, "Prediction" if not attacked else "Different prediction",
                 size=14, fill=G7, line=G5)
        box_text(s, MX + 2.05, y + 1.1, 1.65, 0.7, "Explainer E", size=14, fill=G6)
        arrow(s, MX + 2.88, y + 0.8, MX + 2.88, y + 1.1)
        arrow(s, MX + 3.7, y + 1.45, MX + 4.2, y + 1.45)
        box_text(s, MX + 4.2, y + 1.1, 2.05, 0.7, "Heat-map" if not attacked else "Distorted heat-map",
                 size=14, fill=G7, line=G5)
    row(2.45, "Clean", False)
    row(4.85, "Attacked", True)
    # right column
    card(s, 8.0, 2.0, 4.58, 4.75, fill=G7)
    tb(s, 8.3, 2.25, 4.0, 0.35, "Why this matters", size=18, bold=True, after=0)
    bullets(s, 8.3, 2.8, 4.0, 3.8, [
        "Explanations are used to **trust, debug and audit** models, in medicine, finance and regulation.",
        "Adversarial attacks already fool the **prediction** with changes humans cannot see.",
        "If the **explanation** also shifts, a reviewer may be shown a convincing but wrong story.",
        "So we ask: how stable are explanations when the input is attacked?"], size=15, after=9)
    notes(s, "An image classifier returns a class. An explanation method returns a heat-map that shows which "
             "pixels supported that class. An adversarial attack adds a very small, structured change to the image. "
             "The prediction can flip. The open question is what happens to the heat-map.")
    return s


# =========================================================== 4. glossary
def glossary(prs):
    s = S(prs, "01  |  The problem", "Eight terms you need")
    terms = [
        ("Classifier f", "A neural network mapping an image to 1000 class scores (logits). Here DenseNet-121."),
        ("Adversarial example x'", "An input changed on purpose so the model errs, while it looks unchanged to a person."),
        ("Perturbation δ and budget ε", "δ = x' − x. The L∞ budget ε bounds the largest change to any pixel value."),
        ("White-box evasion", "The attacker knows the model's weights and modifies the input at test time only."),
        ("Explanation (XAI)", "A method that says why the model chose a class for a given image."),
        ("Heat-map E(x, c)", "A 224×224 map in [0,1]; bright pixels supported class c more."),
        ("Explanation robustness", "How little the heat-map changes when the input is perturbed. Measured with SSIM."),
        ("Attack success", "The attacked image is classified as something other than its true class."),
    ]
    cw, ch = 2.8, 2.15
    for i, (t, d) in enumerate(terms):
        r, c = divmod(i, 4)
        x, y = MX + c * (cw + 0.243), 2.0 + r * (ch + 0.25)
        card(s, x, y, cw, ch)
        tb(s, x + 0.18, y + 0.17, cw - 0.36, 0.5, t, size=16, bold=True, after=0)
        tb(s, x + 0.18, y + 0.78, cw - 0.36, ch - 0.9, d, size=14, color=G1, after=0, line=1.05)
    return s


# =========================================================== 5. objective
def objective(prs):
    s = S(prs, "02  |  Research objective", "Research objective")
    card(s, MX, 1.95, CW, 1.45, fill=G7)
    tb(s, MX + 0.35, 2.12, CW - 0.7, 1.2,
       "Measure how strongly standard adversarial attacks distort the explanations of an image "
       "classifier, and rank four explanation methods by how well their explanations survive.",
       size=22, bold=True, line=1.1, after=0)
    rqs = [("RQ1", "How much?", "How far do explanations move when an image is attacked, compared with random noise of the same size?"),
           ("RQ2", "Which method?", "Do Grad-CAM, CAM, LIME and SHAP differ in robustness, and is the difference statistically reliable?"),
           ("RQ3", "Is looking enough?", "Does an attacked image that looks almost identical (high SSIM) also keep a stable explanation?")]
    w = (CW - 0.5) / 3
    for i, (a, b, c) in enumerate(rqs):
        x = MX + i * (w + 0.25)
        tb(s, x, 3.75, w, 0.4, a, size=14, color=G3, after=0)
        tb(s, x, 4.1, w, 0.5, b, size=22, bold=True, after=0)
        tb(s, x, 4.75, w, 1.8, c, size=15, color=G1, line=1.1, after=0)
    notes(s, "The paper is a benchmark study. It does not invent a new attack. It applies four well-known attacks and "
             "measures their side effect on four well-known explanation methods under one common protocol.")
    return s


# ============================================================ 6. hypotheses
def hypotheses(prs):
    s = S(prs, "02  |  Research objective", "What we expect the paper to show, and how we test it")
    rows = [["", "Hypothesis", "How it is tested"],
            ["H1", "Attacks that fool the model also move its explanations by more than random noise of the same size does.",
             "Paired comparison of every attack against the random-noise control (Wilcoxon, Holm-corrected)."],
            ["H2", "Explanation methods differ in robustness; gradient-based CAM-family maps are more stable than the sampling-based LIME and SHAP.",
             "Explanation SSIM, Spearman and top-20% IoU per method; paired tests of Grad-CAM against the others."],
            ["H3", "Input similarity (SSIM, PSNR) does not predict explanation stability: a near-invisible attack can still disturb the explanation.",
             "Compare the sparse JSMA attack and the dense L∞ attacks on both input metrics and explanation metrics."],
            ["H4", "Stronger attacks (larger ε) move explanations more.",
             "Epsilon sweep of BIM from 1/255 to 16/255."]]
    table(s, MX, 1.95, CW, rows, col_w=[0.6, 5.7, 5.2], size=14, row_h=0.88, head_h=0.45,
          align=["c", "l", "l"], bold_first_col=True)
    tb(s, MX, 6.6, CW, 0.35, "These are hypotheses. The results section reports which ones the data support.",
       size=14, italic=True, color=G3, after=0)
    return s


# ============================================================== 7. threat
def threat_model(prs):
    s = S(prs, "02  |  Research objective", "Scope and threat model")
    w = (CW - 0.3) / 2
    card(s, MX, 1.95, w, 2.2)
    tb(s, MX + 0.3, 2.12, w - 0.6, 0.4, "Attacker", size=18, bold=True, after=0)
    bullets(s, MX + 0.3, 2.62, w - 0.6, 1.8, [
        "**White-box**: knows DenseNet-121 and its gradients.",
        "**Evasion**: changes the test image only, never the training data.",
        "Goal: change the predicted class within a small budget."], size=15, after=6)
    card(s, MX + w + 0.3, 1.95, w, 2.2)
    tb(s, MX + w + 0.6, 2.12, w - 0.6, 0.4, "User of the explanation", size=18, bold=True, after=0)
    bullets(s, MX + w + 0.6, 2.62, w - 0.6, 1.8, [
        "Receives the attacked image and runs an explainer on it.",
        "We explain the **original class c** for both images, so the heat-maps are directly comparable.",
        "The attack does not target the explainer."], size=15, after=6)
    rows = [["Dimension (NIST taxonomy)", "Our setting"],
            ["Phase and technique", "Testing phase, Evasion attack (gradient-based)"],
            ["Attacker knowledge", "White box"],
            ["Consequence", "Integrity violation (misclassification)"],
            ["Out of scope", "Poisoning, backdoors, explanation-targeted attacks, defences"]]
    table(s, MX, 4.4, CW, rows, col_w=[3.6, 8.3], size=14, row_h=0.44)
    notes(s, "NIST IR 8269 (draft, withdrawn and superseded by NIST AI 100-2e2023) organises adversarial machine learning "
             "by attack technique, attacker knowledge and consequence. Our setting is a testing-phase, gradient-based, "
             "white-box evasion attack.")
    return s


# ================================================= 8. literature timeline
def lit_timeline(prs):
    s = S(prs, "03  |  Literature survey", "Two research lines that recently met")
    ev = [("2013", "Adversarial examples\nSzegedy et al."), ("2014", "FGSM\nGoodfellow et al."),
          ("2016", "CAM, LIME, JSMA,\nBIM"), ("2017", "Grad-CAM, SHAP,\nPGD, C&W"),
          ("2018", "Sanity checks for\nsaliency maps"), ("2019", "Explanations fooled:\nGhorbani, Dombrowski,\nHeo, Subramanya"),
          ("2020", "LIME and SHAP\nfooled: Slack et al."), ("2021", "LAFEAT\nYu et al."),
          ("2023-24", "Surveys and\nbackdoors: Noppel,\nBaniecki & Biecek")]
    n = len(ev)
    x0, x1, y = MX + 0.3, W - MX - 0.3, 3.2
    hline(s, x0, y, x1 - x0, color=G5, width=1.5)
    step = (x1 - x0) / (n - 1)
    for i, (yr, t) in enumerate(ev):
        cx = x0 + i * step
        dot = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - 0.08), Inches(y - 0.08), Inches(0.16), Inches(0.16))
        dot.fill.solid(); dot.fill.fore_color.rgb = rgb(INK); dot.line.fill.background()
        tb(s, cx - 0.75, y - 0.5, 1.5, 0.3, yr, size=16, bold=True, align="c", after=0)
        ty = y + 0.3 if i % 2 == 0 else y + 1.45
        if i % 2 == 1:
            c = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(cx), Inches(y + 0.1), Inches(cx), Inches(ty - 0.05))
            c.line.color.rgb = rgb(G5)
            c.line.width = Pt(0.75)
        tb(s, cx - 0.82, ty, 1.64, 1.1, t, size=12, color=G2, align="c", after=0, line=1.0)
    tb(s, MX, 5.75, CW, 0.8,
       "Line 1 (2013 onward): attacks on the prediction.  Line 2 (2016 onward): methods that explain the prediction.  "
       "From 2019 the lines meet: attacks on the explanation.",
       size=15, color=G1, line=1.1, after=0)
    return s


# ================================================= 9. lit: attacks
def lit_attacks(prs):
    s = S(prs, "03  |  Literature survey", "Attacks on the prediction (evasion)")
    rows = [["Attack", "Year", "Idea", "Norm", "Reference"],
            ["L-BFGS", "2013", "First adversarial examples, found by constrained optimisation", "L2", "Szegedy et al."],
            ["FGSM", "2015", "One gradient-sign step; explained by locally linear behaviour", "L∞", "Goodfellow et al."],
            ["JSMA", "2016", "Changes the few input features with the highest saliency", "L0", "Papernot et al."],
            ["DeepFool", "2016", "Smallest step to the nearest decision boundary", "L2", "Moosavi-Dezfooli et al."],
            ["BIM / I-FGSM", "2017", "FGSM repeated in small steps, clipped to the ε-ball", "L∞", "Kurakin et al."],
            ["C&W", "2017", "Optimisation attack that breaks defensive distillation", "L2", "Carlini and Wagner"],
            ["PGD", "2018", "BIM with random start; the standard strong first-order attack", "L∞", "Madry et al."],
            ["Latent-feature", "2016, 2021", "Perturb inputs through intermediate-layer features (Sabour; LAFEAT)", "L∞", "Sabour et al.; Yu et al."]]
    table(s, MX, 1.95, CW, rows, col_w=[1.5, 1.3, 5.4, 0.7, 2.7], size=13, row_h=0.5, align=["l", "c", "l", "c", "l"],
          bold_first_col=True)
    tb(s, MX, 6.65, CW, 0.3, "We implement the first, third, fifth and last attack types of this list.", size=13,
       color=G3, italic=True, after=0)
    return s


# ================================================= 10. lit: XAI
def lit_xai(prs):
    s = S(prs, "03  |  Literature survey", "Explanation methods for images")
    rows = [["Method", "Year", "Family", "What it needs", "Output", "Reference"],
            ["CAM", "2016", "Activation based", "Global average pool + linear head", "Class heat-map", "Zhou et al."],
            ["Grad-CAM", "2017", "Gradient + activation", "Gradients at a conv layer", "Class heat-map", "Selvaraju et al."],
            ["LIME", "2016", "Perturbation, surrogate", "Only model outputs", "Superpixel weights", "Ribeiro et al."],
            ["SHAP", "2017", "Perturbation, game theory", "Only model outputs", "Per-region Shapley values", "Lundberg and Lee"],
            ["Saliency / IG", "2014, 2017", "Input gradient", "Gradients", "Pixel attribution", "Simonyan; Sundararajan"]]
    table(s, MX, 1.95, CW, rows, col_w=[1.4, 1.0, 2.3, 2.9, 2.2, 2.1], size=13, row_h=0.6,
          bold_first_col=True)
    card(s, MX, 5.75, CW, 1.0)
    tb(s, MX + 0.3, 5.9, CW - 0.6, 0.8,
       "We test the first four. Saliency and Integrated Gradients are the methods most attacked in earlier work, "
       "so they are good candidates to add next.", size=15, color=G1, line=1.1, after=0)
    return s


# ================================================= 11. attacks on explanations
def lit_xai_attacks(prs):
    s = S(prs, "03  |  Literature survey", "Attacks on explanations: a taxonomy")
    cols = [("Adversarial example", "Perturb the input: prediction stays, heat-map moves.",
             "Ghorbani 2019, Dombrowski 2019, Subramanya 2019, Zhang 2020"),
            ("Model manipulation", "Fine-tune the model: accuracy stays, explanations change.",
             "Heo 2019, Dimanov 2020, Anders 2020"),
            ("Backdoor", "A trigger in the input switches the explanation.", "Viering 2019, Noppel 2023"),
            ("Adversarial surrogate", "A fair-looking surrogate hides a biased model from LIME and SHAP.",
             "Slack 2020, Le Merrer and Trédan 2020"),
            ("Data poisoning", "Poison data used to build the explanation.", "Zhang 2021, Baniecki 2022, Laberge 2023"),
            ("Trust manipulation", "Faithful but misleading explanations shift user trust.", "Lakkaraju and Bastani 2020")]
    cw, ch = 3.75, 2.2
    for i, (t, d, r) in enumerate(cols):
        rr, c = divmod(i, 3)
        x, y = MX + c * (cw + 0.34), 1.95 + rr * (ch + 0.25)
        card(s, x, y, cw, ch)
        tb(s, x + 0.2, y + 0.16, cw - 0.4, 0.4, t, size=17, bold=True, after=0)
        tb(s, x + 0.2, y + 0.62, cw - 0.4, 0.8, d, size=14, color=G1, line=1.05, after=0)
        tb(s, x + 0.2, y + 1.5, cw - 0.4, 0.6, r, size=12, color=G3, line=1.05, after=0)
    tb(s, MX, 6.72, CW, 0.3, "Taxonomy follows Baniecki and Biecek (2024). Our study sits in the first group, with unmodified standard attacks.",
       size=13, italic=True, color=G3, after=0)
    return s


# ================================================= 12. findings of earlier work
def lit_findings(prs):
    s = S(prs, "03  |  Literature survey", "What earlier studies found")
    items = [("Adebayo et al., 2018", "Some saliency maps barely change when the model's weights are randomised, so a plausible picture is not proof of a faithful explanation."),
             ("Ghorbani et al., 2019", "Visually identical images with the same predicted label can have very different saliency, IG and DeepLIFT maps."),
             ("Dombrowski et al., 2019", "Explanations can be pushed to an arbitrary target map. The curvature of the decision boundary is to blame; smoothing helps."),
             ("Subramanya et al., 2019", "Small adversarial patches fool Grad-CAM and occlusion explanations while the model stays confident."),
             ("Heo et al., 2019", "Fine-tuning a network changes its LRP, Grad-CAM and gradient explanations with almost no loss of accuracy."),
             ("Slack et al., 2020", "A biased model can pass as fair under LIME and SHAP because their perturbed samples are easy to detect.")]
    cw, ch = 3.75, 2.2
    for i, (t, d) in enumerate(items):
        rr, c = divmod(i, 3)
        x, y = MX + c * (cw + 0.34), 1.95 + rr * (ch + 0.25)
        card(s, x, y, cw, ch)
        tb(s, x + 0.2, y + 0.16, cw - 0.4, 0.35, t, size=15, bold=True, after=0)
        tb(s, x + 0.2, y + 0.6, cw - 0.4, 1.55, d, size=14, color=G1, line=1.05, after=0)
    return s


# ================================================= 13. defences
def lit_defences(prs):
    s = S(prs, "03  |  Literature survey", "Defences proposed so far")
    rows = [["Defence family", "Idea", "Examples"],
            ["Robust training / regularisation", "Penalise explanation change during training", "Chen 2019; Boopathy 2020; Dombrowski 2022; Joo 2023"],
            ["Smoothing", "Smoother decision surface gives smoother attributions", "Wang 2020 (smoothed geometry); softplus activations"],
            ["Aggregation", "Average several explanation methods; one attack rarely fools all", "Rieger and Hansen 2020"],
            ["Certified robustness", "Provable bounds on explanation change", "Liu 2022; Wicker 2023"],
            ["Better sampling for LIME / SHAP", "Keep perturbed samples on the data manifold", "Ghalebikesabi 2021; Vreš 2022; Blesch 2023"],
            ["Detection", "Flag adversarial or out-of-distribution inputs", "Carmichael and Scheirer 2023"]]
    table(s, MX, 1.95, CW, rows, col_w=[3.2, 4.6, 4.1], size=13, row_h=0.62, bold_first_col=True)
    tb(s, MX, 6.5, CW, 0.4, "These defences are outside our scope, but the benchmark gives them a baseline to beat.",
       size=14, italic=True, color=G3, after=0)
    return s


# ================================================= 14. evaluation
def lit_eval(prs):
    s = S(prs, "03  |  Literature survey", "How explanations are evaluated today")
    rows = [["Resource", "What it provides", "Reference"],
            ["ROAR", "Removes the pixels an explanation marks and retrains; many methods are no better than random", "Hooker et al., 2019"],
            ["CLEVR-XAI", "Ground-truth heat-maps for explanation quality", "Arras et al., 2022"],
            ["OpenXAI", "Benchmark with metrics and leaderboards (tabular in first release)", "Agarwal et al., 2022"],
            ["Quantus", "Python toolkit of explanation-quality metrics", "Hedström et al., 2023"],
            ["SAFARI", "Black-box evaluation of interpretation robustness", "Huang et al., 2023"]]
    table(s, MX, 1.95, CW, rows, col_w=[1.8, 7.6, 2.5], size=13, row_h=0.6, bold_first_col=True)
    card(s, MX, 5.8, CW, 0.95)
    tb(s, MX + 0.3, 5.95, CW - 0.6, 0.8,
       "The 2024 survey notes that these toolkits do not implement adversarial attacks on explanations and that new "
       "benchmarks for attacks and defences would be valuable.", size=15, line=1.1, after=0)
    return s


# ================================================= 15. gap + contribution
def gap(prs):
    s = S(prs, "03  |  Literature survey", "The gap and our contribution")
    w = (CW - 0.3) / 2
    card(s, MX, 1.95, w, 4.6)
    tb(s, MX + 0.3, 2.12, w - 0.6, 0.4, "What is missing (our reading)", size=18, bold=True, after=0)
    bullets(s, MX + 0.3, 2.7, w - 0.6, 3.8, [
        "Most attacks on explanations target **one** method, usually gradient saliency, with custom attack code.",
        "Few studies put **Grad-CAM, CAM, LIME and SHAP** side by side on the same model and images.",
        "Controls are rare: **random noise** of equal size, and the **self-similarity** of randomised explainers.",
        "Paired **significance tests** are seldom reported."], size=15, after=9)
    card(s, MX + w + 0.3, 1.95, w, 4.6, fill=G6)
    tb(s, MX + w + 0.6, 2.12, w - 0.6, 0.4, "What this project adds", size=18, bold=True, after=0)
    bullets(s, MX + w + 0.6, 2.7, w - 0.6, 3.8, [
        "One **common protocol**: 4 standard attacks × 4 explanation methods on DenseNet-121.",
        "**Controls**: random-noise attack and LIME re-run floor.",
        "**Statistics**: bootstrap confidence intervals and paired Wilcoxon tests.",
        "**Open code** written from scratch, tests included, so others can add attacks or explainers."], size=15, after=9)
    tb(s, MX, 6.65, CW, 0.3, "This gap statement is based on the surveys we read; it is not an exhaustive literature search.",
       size=12, italic=True, color=G3, after=0)
    return s


# ============================================================ 16. pipeline
def pipeline(prs):
    s = S(prs, "04  |  Method", "Experimental pipeline")
    steps = [("1", "Image bank", f"{n_images()} ImageNet images, one per class, all classified correctly by DenseNet-121"),
             ("2", "Attack", "Noise control, FGSM, BIM, JSMA, latent-feature PGD produce x' for each x"),
             ("3", "Explain", "Grad-CAM, CAM, LIME, SHAP on x and on x', same class c"),
             ("4", "Compare", "SSIM, Spearman rank correlation and top-20% IoU between the two heat-maps"),
             ("5", "Test", "Means with bootstrap 95% intervals; paired Wilcoxon tests, Holm-corrected")]
    w, gapx = 2.05, 0.42
    for i, (n, t, d) in enumerate(steps):
        x = MX + i * (w + gapx)
        card(s, x, 2.3, w, 3.2)
        tb(s, x + 0.2, 2.45, 0.6, 0.6, n, size=30, color=G5, after=0)
        tb(s, x + 0.2, 3.1, w - 0.4, 0.4, t, size=19, bold=True, after=0)
        tb(s, x + 0.2, 3.62, w - 0.4, 1.8, d, size=14, color=G1, line=1.05, after=0)
        if i < len(steps) - 1:
            arrow(s, x + w + 0.04, 3.9, x + w + gapx - 0.04, 3.9)
    tb(s, MX, 5.9, CW, 0.8,
       "Extra experiment: BIM is repeated at ε = 1, 2, 4, 8, 16 (out of 255) to see how explanation damage grows with attack strength.",
       size=15, color=G1, line=1.1, after=0)
    return s


# ============================================================ 17. notation
def notation(prs):
    s = S(prs, "04  |  Method", "Notation")
    rows = [["Symbol", "Meaning"],
            ["x", "clean image, values in [0, 1], size 3×224×224; n = 150,528 input values"],
            ["y, c", "true class of x (1 of 1000); the class we explain"],
            ["z = f(x)", "the 1000 logits (class scores); softmax(z) gives probabilities"],
            ["x' = x + δ", "adversarial image; δ is the perturbation"],
            ["ε", "L∞ budget: every value changes by at most ε (8/255 ≈ 0.031)"],
            ["ℒ(f(x), y)", "cross-entropy loss of the model on x with label y"],
            ["A, w", "final feature maps (1024 maps of 7×7) and the classifier weights"],
            ["E(x, c)", "heat-map for class c, 224×224, scaled to [0, 1]"]]
    table(s, MX, 1.95, CW, rows, col_w=[2.6, 9.3], size=14, row_h=0.45, bold_first_col=True)
    eq(s, r"$S_{a,m}=\frac{1}{N}\sum_{\mathbf{x}}\ \mathrm{SSIM}\!\left(E_m(f,\mathbf{x},c),\,E_m(f,a(\mathbf{x}),c)\right)$",
       MX, 6.3, size=18)
    tb(s, MX + 7.1, 6.4, 5.0, 0.4, "Robustness score of method m under attack a (1 = unchanged).", size=12, color=G3, after=0)
    return s


# ============================================================ 18. setup
def setup(prs):
    s = S(prs, "04  |  Method", "Experimental setup")
    rows = [["Item", "Value"],
            ["Model", "DenseNet-121, ImageNet-1k pretrained (torchvision), about 8 M parameters; mean/std normalisation inside the model"],
            ["Data", f"{n_images()} images, one per class, drawn with seed 42 from a public 1000-image ImageNet sample; only images the model classifies correctly"],
            ["Input", "Resize 256, centre crop 224; values in [0,1]; attacks and explainers share this space"],
            ["Attack budgets", "ε = 8/255 for FGSM, BIM, latent PGD and noise control; JSMA limited to 2% of the 150,528 values"],
            ["Explained class", "Always the true class c, for the clean and the attacked image"],
            ["Randomness", "Seed 42; LIME run twice on clean images to measure its own variability"],
            ["Hardware", f"CPU only (12 threads); {n_images()} images (extendable to 50 or more), so conclusions are about this sample"]]
    table(s, MX, 1.95, CW, rows, col_w=[2.3, 9.6], size=14, row_h=0.6, bold_first_col=True)
    return s


# ======================================================= attack slide helper
def attack_slide(prs, kicker, title, idea, eqs, symbols, params, pros, notes_txt):
    s = S(prs, kicker, title)
    lw = 7.3
    tb(s, MX, 1.95, lw, 1.0, idea, size=16, color=INK, line=1.1, after=0)
    y = 3.0
    for e, sz in eqs:
        _, eh = eq(s, e, MX, y, size=sz, max_w=lw)
        y += eh + 0.16
    tb(s, MX, y + 0.05, lw, 0.3, "Symbols", size=13, bold=True, color=G3, after=2)
    tb(s, MX, y + 0.38, lw, 6.9 - y - 0.4, symbols, size=14, color=G1, line=1.05, after=3)
    # right column
    rx, rw = 8.4, W - MX - 8.4
    card(s, rx, 1.95, rw, 2.55)
    tb(s, rx + 0.2, 2.07, rw - 0.4, 0.3, "Parameters in our experiments", size=14, bold=True, after=0)
    table(s, rx + 0.12, 2.42, rw - 0.24, [["Setting", "Value"]] + [[a, b] for a, b in params], col_w=[1.7, 2.0], size=13,
          row_h=0.35, hdr_fill="FFFFFF")
    card(s, rx, 4.7, rw, 2.15, fill=G6)
    tb(s, rx + 0.2, 4.82, rw - 0.4, 0.3, "Strengths and limits", size=14, bold=True, after=0)
    bullets(s, rx + 0.2, 5.18, rw - 0.4, 1.65, pros, size=13, after=3, line=1.03)
    notes(s, notes_txt)
    return s


def attacks(prs):
    attack_slide(
        prs, "04  |  Attacks", "FGSM: fast gradient sign (Goodfellow et al., 2015)",
        "Take one step of size ε in the direction that increases the loss the most, using only the sign of the gradient. "
        "Cheap, and every pixel moves by exactly ±ε.",
        [(r"$\mathbf{x}'=\mathrm{clip}_{[0,1]}\!\left(\mathbf{x}+\varepsilon\cdot\mathrm{sign}\!\left(\nabla_{\mathbf{x}}\,\mathcal{L}(f(\mathbf{x}),y)\right)\right)$", 22),
         (r"$\mathcal{L}(f(\mathbf{x}),y)=-\log\,\mathrm{softmax}(\mathbf{z})_y,\qquad\|\mathbf{x}'-\mathbf{x}\|_\infty\leq\varepsilon$", 22)],
        ["**∇ℒ** gradient of the loss with respect to the image pixels",
         "**sign(·)** +1, −1 or 0 per value, so all values change by the same amount",
         "**clip** keeps pixel values inside [0, 1]",
         "**z** logits;  **y** true class"],
        [("Budget ε", "8/255 ≈ 0.031"), ("Norm", "L∞"), ("Steps", "1"), ("Label used", "true class y")],
        ["Strength: one gradient call.", "Limit: single step, so weaker than iterative attacks at the same ε.",
         "Touches every one of the 150,528 values."],
        "FGSM linearises the loss around x. The gradient sign tells every pixel which direction increases the loss; "
        "the step size ε bounds the change. The result is dense noise-like change over the whole image.")
    attack_slide(
        prs, "04  |  Attacks", "BIM: basic iterative method (Kurakin et al., 2017)",
        "Repeat small FGSM steps of size α and, after each step, project back into the ε-ball around the clean image. "
        "More steps find a stronger perturbation inside the same budget.",
        [(r"$\mathbf{x}'_{0}=\mathbf{x},\qquad \mathbf{x}'_{t+1}=\mathrm{clip}_{[0,1]}\,\Pi_{\varepsilon}\!\left(\mathbf{x}'_t+\alpha\cdot\mathrm{sign}\!\left(\nabla_{\mathbf{x}}\mathcal{L}(f(\mathbf{x}'_t),y)\right)\right)$", 20),
         (r"$\Pi_{\varepsilon}(\mathbf{u})=\min\!\left(\max(\mathbf{u},\,\mathbf{x}-\varepsilon),\ \mathbf{x}+\varepsilon\right)$", 22)],
        ["**t** iteration counter, t = 0 … T − 1",
         "**α** step size of each iteration (smaller than ε)",
         "**Π_ε** projection: keeps every value within ε of the clean image",
         "**T** number of iterations"],
        [("Budget ε", "8/255"), ("Step α", "2/255"), ("Iterations T", "10"), ("Norm", "L∞")],
        ["Strength: much higher success than FGSM at equal ε.", "Limit: T gradient calls; no random start (PGD adds one).",
         "Usually very small L2 change for the same L∞."],
        "BIM is FGSM iterated. Because each step is small and the result is clipped to the ε-ball, the final perturbation "
        "never exceeds ε in any pixel but is better aimed than a single FGSM step.")
    attack_slide(
        prs, "04  |  Attacks", "JSMA: saliency-map attack (Papernot et al., 2016)",
        "A sparse, L0 attack. It ranks input values by how much raising them increases a target class and lowers all others, "
        "then pushes the best few to the maximum, and repeats.",
        [(r"$a_i=\frac{\partial z_t}{\partial x_i},\qquad b_i=\sum_{j\neq t}\frac{\partial z_j}{\partial x_i}$", 22),
         (r"$s_i=a_i\,|b_i|\ \ \mathrm{if}\ a_i>0\ \mathrm{and}\ b_i<0,\ \ \mathrm{else}\ 0\qquad x_i\leftarrow\min(x_i+\theta,\,1)$", 19)],
        ["**t** target class: here the runner-up class of the clean image",
         "**a, b** gradient of the target logit, and of the sum of all other logits, at input value i",
         "**s** saliency score; the top k values are raised by θ each iteration",
         "Stops when the model outputs t or **γ·n** values have changed"],
        [("Target t", "runner-up class"), ("θ (increase)", "+1, clipped"), ("k per iteration", "20 values"), ("Budget γ", "2% of values")],
        ["Strength: changes only about 1,000 to 3,000 values.", "Limit: changed values can be saturated, so the L∞ size is large.",
         "We use single values; Papernot's pairs search is too costly at this size."],
        "The original JSMA searches pairs of features, which scales as n squared. With 150,528 inputs that is not "
        "feasible, so we use the common single-feature variant. The sparse changes are bright dots in the image.")
    attack_slide(
        prs, "04  |  Attacks", "Latent-feature PGD (LAFEAT-inspired)",
        "Instead of the output loss, maximise how far the intermediate features of the network move away from those of the clean image, "
        "using gradients taken at hidden layers.",
        [(r"$\max_{\|\mathbf{x}'-\mathbf{x}\|_\infty\leq\varepsilon}\ \sum_{l\in\mathcal{S}}\frac{\|\phi_l(\mathbf{x}')-\phi_l(\mathbf{x})\|_2^2}{\|\phi_l(\mathbf{x})\|_2^2}$", 24),
         (r"$\mathbf{x}'_{t+1}=\Pi_{\varepsilon}\!\left(\mathbf{x}'_t+\alpha\cdot\mathrm{sign}(\nabla_{\mathbf{x}}\,\mathrm{loss})\right),\quad \mathbf{x}'_0=\mathbf{x}+\mathrm{Uniform}(-\varepsilon,\varepsilon)$", 19)],
        ["**φ(x)** activations of the layers in 𝒮 = {dense blocks 2, 3, 4 and norm5}",
         "**Normalised distance** makes each layer count equally",
         "**Random start** is needed: at x' = x the distance and its gradient are zero"],
        [("Budget ε", "8/255"), ("Step α", "1/255"), ("Iterations", "20"), ("Layers 𝒮", "4 (listed left)")],
        ["Strength: attacks the representation itself.", "Honest label: this is **not** the original LAFEAT, which trains "
         "latent classifiers; we keep its core idea only."],
        "The original LAFEAT (Yu et al., CVPR 2021) breaks defended models by using gradients from latent-feature classifiers. "
        "We do not re-implement that. Our attack is a feature-disruption PGD, so the slides and paper call it latent-feature PGD.")
    attack_slide(
        prs, "04  |  Attacks", "Control: random sign noise (not an attack)",
        "Add ±ε with a random sign to every value. It has the same L∞ size and almost the same L2 size as FGSM but no "
        "adversarial direction, so it shows how much of the explanation change is caused by noise alone.",
        [(r"$\mathbf{x}'=\mathrm{clip}_{[0,1]}\!\left(\mathbf{x}+\varepsilon\cdot\mathbf{r}\right),\qquad r_i\in\{-1,+1\}\ \mathrm{uniformly}$", 24)],
        ["**r** random sign vector, same size as the image",
         "Any explanation change here is the baseline; an attack must beat it to count as harmful"],
        [("Budget ε", "8/255"), ("Norm", "L∞"), ("Seed", "image index"), ("Expected", "model not fooled")],
        ["Needed to interpret every result.", "Without it, a drop in explanation similarity could be ordinary noise sensitivity."],
        "Earlier studies often compare only with the clean heat-map. A matched random-noise control tells us whether adversarial "
        "direction matters or whether any noise of this size would change the explanation.")


# ======================================================= xai slides
def xai(prs):
    attack_slide(
        prs, "04  |  Explanation methods", "CAM: class activation mapping (Zhou et al., 2016)",
        "For networks that end in global average pooling and one linear layer, the class score is a weighted sum of the last "
        "feature maps. Showing that weighted sum as an image gives the class heat-map.",
        [(r"$M_c(i,j)=\mathrm{ReLU}\!\left(\sum_{k=1}^{1024}w^{c}_{k}\,F_k(i,j)\right),\qquad F=\mathrm{ReLU}(A)$", 22),
         (r"$y^c=\sum_k w^c_k\cdot\frac{1}{HW}\sum_{i,j}F_k(i,j)$", 22)],
        ["**F** the 1024 final feature maps (7×7) after the last ReLU",
         "**w** classifier weight linking channel k to class c",
         "The 7×7 map is upsampled to 224×224 and scaled to [0, 1]"],
        [("Layer", "norm5 + ReLU"), ("Map size", "7×7 to 224×224"), ("Class", "true class c"), ("Needs", "GAP + linear head")],
        ["Strength: needs no gradient and one forward pass.", "Limit: only works for GAP-plus-linear architectures; DenseNet qualifies."],
        "DenseNet-121 ends with ReLU, global average pooling and one linear layer, so CAM applies directly. The classifier's own "
        "weights decide how much each feature map contributes to the heat-map.")
    attack_slide(
        prs, "04  |  Explanation methods", "Grad-CAM: gradient-weighted CAM (Selvaraju et al., 2017)",
        "Generalises CAM to any convolutional network. The weight of each feature map is the average gradient of the class "
        "score flowing into it, so no special architecture is needed.",
        [(r"$\alpha^{c}_{k}=\frac{1}{Z}\sum_{i,j}\frac{\partial y^{c}}{\partial A^{k}_{ij}}\qquad L^{c}=\mathrm{ReLU}\!\left(\sum_k\alpha^{c}_{k}A^{k}\right)$", 24)],
        ["**y** class score (logit) of class c;  **A** feature maps of the chosen layer",
         "**Z** number of positions (here 7×7 = 49) so α is a spatial average",
         "**ReLU** keeps only evidence that supports class c"],
        [("Layer", "last dense block"), ("Map size", "7×7 to 224×224"), ("Extra variant", "dense block 3 (14×14)"), ("Class", "true class c")],
        ["Strength: works on any CNN layer.", "Limit: gradients can be noisy; coarse 7×7 resolution."],
        "Grad-CAM is computed at the output of the last dense block, before the final batch-norm. The extra variant at dense block 3 "
        "has higher resolution and is reported as a supplementary result.")
    s = S(prs, "04  |  Explanation methods", "On DenseNet-121, CAM and Grad-CAM nearly coincide")
    tb(s, MX, 1.95, 6.9, 1.0,
       "If Grad-CAM is taken on the final post-ReLU feature map F, the gradient of the class score is the classifier weight, "
       "so the Grad-CAM weights equal the CAM weights up to the constant 1/Z.", size=16, line=1.1, after=0)
    eq(s, r"$\frac{\partial y^c}{\partial F_k(i,j)}=\frac{w^c_k}{HW}\ \Rightarrow\ \alpha^c_k=\frac{w^c_k}{Z}\ \Rightarrow\ L^c=\frac{1}{Z}\,M_c$",
       MX, 3.2, size=24, max_w=7.2)
    tb(s, MX, 4.5, 6.9, 1.9, [
        "We confirm this numerically in the test suite: Grad-CAM on the post-ReLU map equals CAM to 1e-4.",
        "Using the last dense block instead gives a slightly different map, but on our 20 clean images the two agree almost exactly (SSIM 0.98, Pearson 0.998).",
        "Consequence: Grad-CAM and CAM are **not independent evidence**. We report both, but treat them as one family."],
       size=15, color=G1, line=1.1, after=7)
    card(s, 8.4, 1.95, W - MX - 8.4, 4.8, fill=G6)
    tb(s, 8.65, 2.15, 3.7, 0.4, "Why this matters", size=17, bold=True, after=0)
    tb(s, 8.65, 2.7, 3.7, 3.9,
       "An earlier draft of this project showed different robustness numbers for Grad-CAM and CAM on DenseNet. "
       "Mathematically they cannot differ much on this network, so such a difference would be a warning sign. "
       "ResNet-50 and EfficientNet end the same way, so Grad-CAM and CAM coincide there too. A paper that wants two independent "
       "gradient-based methods needs a different head, such as a ViT or VGG-16.",
       size=14, color=G1, line=1.08, after=0)
    notes(s, "Selvaraju et al. (2017) show Grad-CAM reduces to CAM for architectures built from convolution, global average "
             "pooling and a linear layer. DenseNet-121 is such an architecture at its final layer.")
    attack_slide(
        prs, "04  |  Explanation methods", "LIME: local surrogate model (Ribeiro et al., 2016)",
        "Split the image into superpixels, switch random subsets off, ask the model for its output on each variant, and fit a "
        "simple weighted linear model. Its weights are the explanation.",
        [(r"$\xi(\mathbf{x})=\arg\min_{g\in G}\ \mathcal{L}(f,g,\pi_{\mathbf{x}})+\Omega(g),\quad g(\mathbf{z})=\mathbf{w}\cdot\mathbf{z}$", 21),
         (r"$\mathcal{L}=\sum_{\mathbf{z}}\pi_{\mathbf{x}}(\mathbf{z})\,\left(f(h(\mathbf{z}))-g(\mathbf{z})\right)^2,\quad \pi_{\mathbf{x}}(\mathbf{z})=e^{-D(\mathbf{x},\mathbf{z})^2/\sigma^2}$", 21)],
        ["**z** binary vector: which superpixels are kept;  **h(z)** the masked image",
         "**π** weight of a sample by closeness to the original;  **Ω** model complexity",
         "**w** one weight per superpixel; we keep the positive ones as the heat-map"],
        [("Superpixels", "50 (SLIC)"), ("Samples", "300"), ("Variants", "own / fixed superpixels"), ("Seed", "0; re-run with 1")],
        ["Strength: needs only model outputs.", "Limit: random sampling, so two runs differ; we measure that floor. We report LIME with the attacked image's own "
         "superpixels and with the clean image's (fixed)."],
        "LIME is random. We therefore re-run LIME on the clean image with a second seed and report its self-similarity: no attack "
        "can be blamed for changes smaller than this.")
    attack_slide(
        prs, "04  |  Explanation methods", "SHAP: Shapley values (Lundberg and Lee, 2017)",
        "Assign each image region its Shapley value from cooperative game theory: its average marginal contribution to the "
        "class score over all orders of adding regions.",
        [(r"$\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!\,(|N|-|S|-1)!}{|N|!}\left[v(S\cup\{i\})-v(S)\right]$", 24),
         (r"$\sum_{i}\phi_i=v(N)-v(\emptyset)$", 22)],
        ["**N** set of regions (features);  **S** a subset not containing i",
         "**v(S)** class logit when only regions in S are visible, the rest blurred",
         "**φ** contribution of region i; efficiency: contributions sum to the prediction gap"],
        [("Algorithm", "Partition explainer"), ("Masker", "blur 32×32"), ("Evaluations", "400 per map"), ("Value", "class-c logit")],
        ["Strength: fair attribution with a clear axiom set.", "Limit: exact values are exponential, so a hierarchical approximation is used; "
         "few evaluations give coarse maps."],
        "Exact Shapley values need 2 to the power N model calls. The Partition explainer groups pixels hierarchically; with 400 "
        "evaluations the maps are coarse. This budget limits what we can say about SHAP.")


# ======================================================= metrics
def metrics(prs):
    s = S(prs, "04  |  Metrics", "Input-space metrics: how visible is the attack?")
    rows = [["Metric", "Definition", "Reads as"],
            ["L∞", "largest |δ| over all values", "largest change to any value; the attack budget"],
            ["L2", "square root of Σ δ²", "total energy of the perturbation"],
            ["L0", "count of values with δ ≠ 0", "how many values were changed (of 150,528)"],
            ["PSNR (dB)", "10 · log10(1 / MSE)", "higher = closer to the original"],
            ["SSIM (image)", "structure, contrast and luminance similarity", "1 = identical, lower = more visible change"],
            ["Success rate", "share of images with argmax f(x') ≠ y", "how often the attack fools the model"]]
    table(s, MX, 1.95, 7.6, rows, col_w=[1.5, 2.6, 3.5], size=13, row_h=0.64, bold_first_col=True)
    card(s, 8.7, 1.95, W - MX - 8.7, 4.55)
    tb(s, 8.95, 2.12, 3.4, 0.35, "Image SSIM", size=16, bold=True, after=0)
    eq(s, r"$\mathrm{SSIM}(a,b)=\frac{(2\mu_a\mu_b+c_1)(2\sigma_{ab}+c_2)}{(\mu_a^2+\mu_b^2+c_1)(\sigma_a^2+\sigma_b^2+c_2)}$",
       8.95, 2.65, size=15, max_w=3.45)
    tb(s, 8.95, 4.0, 3.4, 2.4,
       "μ are local means, σ² local variances and σ(ab) the covariance in an 11×11 window; c₁, c₂ avoid division by zero. "
       "Computed on the RGB image with data range 1.", size=13, color=G1, line=1.05, after=0)
    return s


def explanation_metrics(prs):
    s = S(prs, "04  |  Metrics", "Explanation-space metrics: how much did the heat-map move?")
    rows = [["Metric", "Definition", "Sensitive to"],
            ["Explanation SSIM", "SSIM between the two 224×224 heat-maps", "local structure and contrast"],
            ["Spearman ρ", "rank correlation of all 50,176 pixel values", "overall ordering, ignores scale"],
            ["Top-20% IoU", "overlap of the 20% most important pixels", "whether the 'where' agrees"]]
    table(s, MX, 1.95, CW, rows, col_w=[2.4, 5.6, 3.9], size=14, row_h=0.55, bold_first_col=True)
    eq(s, r"$\rho=1-\frac{6\sum_i d_i^2}{n\,(n^2-1)}\qquad\qquad \mathrm{IoU}=\frac{|T_{20}(a)\cap T_{20}(b)|}{|T_{20}(a)\cup T_{20}(b)|}$",
       MX, 4.35, size=22, max_w=CW)
    tb(s, MX, 5.15, CW, 0.5, "d = rank difference of a pixel between the two maps;  T₂₀(·) = set of the top 20% pixels of a map.",
       size=13, color=G3, after=0)
    card(s, MX, 5.7, CW, 1.1)
    tb(s, MX + 0.3, 5.82, CW - 0.6, 0.9,
       "**Controls and statistics.**  Each attack is compared with the random-noise control; LIME is re-run on the clean image to "
       "get its self-similarity; 95% intervals come from 2,000 bootstrap resamples of images; method and attack comparisons use "
       "paired Wilcoxon signed-rank tests with Holm correction.", size=14, color=INK, line=1.08, after=0)
    return s
