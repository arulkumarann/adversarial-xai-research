"""Adversarial attacks, written from scratch in PyTorch (white-box, evasion).

All attacks take an image batch x in [0,1] (shape B,3,224,224), the true label
y, and return an adversarial image in [0,1].

    fgsm        one gradient-sign step                       (Goodfellow 2015)
    bim         iterated FGSM with projection                (Kurakin 2017)
    jsma        sparse, saliency-guided pixel flipping       (Papernot 2016)
    latent_pgd  PGD that disrupts intermediate features      (LAFEAT-inspired,
                                                              see note below)
    noise       random-sign noise, used as a CONTROL (not an attack)

Note on `latent_pgd`: the original LAFEAT (Yu et al., CVPR 2021) trains small
per-layer "latent classifiers" and uses their gradients to break defended
models.  We do NOT re-implement that.  Our attack keeps the core idea - perturb
the input using gradients taken at latent layers - by maximising the relative
distance between clean and adversarial intermediate DenseNet features under an
L_inf budget.  We call it "latent-feature PGD" everywhere.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _clip01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


# ----------------------------------------------------------------------
# FGSM:  x' = clip( x + eps * sign( grad_x CE(f(x), y) ) )
# ----------------------------------------------------------------------
def fgsm(model, x, y, eps: float):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    return _clip01(x.detach() + eps * grad.sign()).detach()


# ----------------------------------------------------------------------
# BIM:  x_{t+1} = Proj_{B_inf(x,eps)} ( x_t + alpha * sign( grad CE ) )
# ----------------------------------------------------------------------
def bim(model, x, y, eps: float, alpha: float, steps: int):
    x0 = x.clone().detach()
    xa = x0.clone()
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), y)
        (grad,) = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * grad.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps)  # project to eps-ball
        xa = _clip01(xa)
    return xa.detach()


# ----------------------------------------------------------------------
# JSMA (single-feature variant)
#   target t = runner-up class.  a_i = d logit_t / d x_i,
#   b_i = d (sum_{j != t} logit_j) / d x_i.
#   A feature is useful if it raises the target (a_i > 0) and lowers the rest
#   (b_i < 0); score s_i = a_i * |b_i|.  Each iteration sets the k best
#   features to +theta (clipped to 1) until the model outputs t or the
#   budget gamma * n_features is spent.
#   (Papernot's original searches feature PAIRS, O(n^2); infeasible for the
#   150,528 inputs of a 224x224 RGB image, so we use single features.)
# ----------------------------------------------------------------------
def jsma(model, x, y, gamma: float = 0.02, k: int = 20, theta: float = 1.0):
    assert x.shape[0] == 1, "jsma works on one image at a time"
    xa = x.clone().detach()
    n = xa.numel()
    budget = int(gamma * n)
    changed = torch.zeros(n, dtype=torch.bool, device=x.device)

    with torch.no_grad():
        logits = model(xa)
        logits[0, y[0]] = -float("inf")
        target = int(logits.argmax(1))

    max_iters = budget // k + 1
    for _ in range(max_iters):
        xa.requires_grad_(True)
        out = model(xa)
        if int(out.argmax(1)) == target or int(changed.sum()) >= budget:
            break
        (a,) = torch.autograd.grad(out[0, target], xa, retain_graph=True)
        (s,) = torch.autograd.grad(out.sum(), xa)
        a, b = a.flatten(), (s - a.view_as(s)).flatten()
        xf = xa.detach().flatten()

        valid = (~changed) & (xf < 1.0 - 1e-6)
        score = torch.where((a > 0) & (b < 0), a * b.abs(), torch.zeros_like(a))
        if float(score[valid].max()) <= 0:  # fallback: relaxed score
            score = a - b
        score = torch.where(valid, score, torch.full_like(score, -float("inf")))
        idx = score.topk(k).indices
        xf = xf.clone()
        xf[idx] = (xf[idx] + theta).clamp(0, 1)
        changed[idx] = True
        xa = xf.view_as(x).detach()
    return xa.detach()


# ----------------------------------------------------------------------
# Latent-feature PGD.  Maximise  sum_l ||phi_l(x') - phi_l(x)||^2 / ||phi_l(x)||^2
# over layers l in {denseblock2, denseblock3, denseblock4, norm5}.
# Needs a random start: at x' = x the distance (and its gradient) is zero.
# ----------------------------------------------------------------------
LATENT_LAYERS = ("denseblock2", "denseblock3", "denseblock4", "norm5")


def latent_features(model, x) -> list[torch.Tensor]:
    h = model.normalize(x)
    feats = []
    for name, module in model.net.features.named_children():
        h = module(h)
        if name in LATENT_LAYERS:
            feats.append(h)
    return feats


def latent_pgd(model, x, y, eps: float, alpha: float, steps: int, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x0 = x.clone().detach()
    with torch.no_grad():
        ref = [f.clone() for f in latent_features(model, x0)]
    start = (torch.rand(x0.shape, generator=g) * 2 - 1).to(x0.device) * eps
    xa = _clip01(x0 + start)
    for _ in range(steps):
        xa.requires_grad_(True)
        feats = latent_features(model, xa)
        loss = sum(
            ((f - r) ** 2).sum() / (r ** 2).sum().clamp_min(1e-12)
            for f, r in zip(feats, ref)
        )
        (grad,) = torch.autograd.grad(loss, xa)
        xa = xa.detach() + alpha * grad.sign()
        xa = torch.min(torch.max(xa, x0 - eps), x0 + eps)
        xa = _clip01(xa)
    return xa.detach()


# ----------------------------------------------------------------------
# Control: random sign noise with the same L_inf size as FGSM (not adversarial)
# ----------------------------------------------------------------------
def noise(x, eps: float, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sign = (torch.randint(0, 2, x.shape, generator=g) * 2 - 1).float().to(x.device)
    return _clip01(x + eps * sign)
