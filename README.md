# Adversarial attacks on explainable AI (image classification)

A from-scratch benchmark that measures how much standard adversarial attacks distort the
explanations of an ImageNet classifier (DenseNet-121), and which explanation methods are
most robust.

| | |
|---|---|
| **Attacks** (own PyTorch code, `advxai/attacks.py`) | FGSM, BIM, JSMA (single-feature variant), latent-feature PGD, plus a random-noise **control** |
| **Explanation methods** (`advxai/explain.py`) | Grad-CAM, CAM (own code), LIME and SHAP (libraries `lime`, `shap`) |
| **Input metrics** | L-inf, L2, L0, PSNR, image SSIM, attack success rate |
| **Explanation metrics** | SSIM, Spearman rank correlation, top-20% IoU between clean and attacked heat-maps |
| **Statistics** | 95% bootstrap intervals, paired Wilcoxon tests with Holm correction, LIME self-similarity floor |
| **Extra** | BIM epsilon sweep (1/255 ... 16/255) |

## Honest notes

* `latent_pgd` is **inspired by** LAFEAT (Yu et al., CVPR 2021) but is not the original method.
* `jsma` uses single features, not Papernot's feature pairs (infeasible at 150,528 inputs).
* On DenseNet-121, Grad-CAM and CAM are nearly the same method (they are identical when
  Grad-CAM is taken on the final post-ReLU map; see `tests/test_core.py`).
* The reported run uses 20 images (one per ImageNet class) on CPU, so results describe this sample.
  The run is resumable and can be extended to 50 (or more, up to the 69 correctly classified of 80
  downloaded) without redoing finished images, see "Extend the run".

## Reproduce

```bash
pip install -r requirements.txt
python -c "from advxai import data; data.download_subset(80, 42)"   # ~9 MB, public sample images
python -m pytest -q tests                                          # sanity tests
python run_experiment.py --part 0 --parts 2 --threads 6 &          # resumable, splittable
python run_experiment.py --part 1 --parts 2 --threads 6
python analyze.py                                                  # tables, statistics, figures
python -m deck.build_deck                                          # PowerPoint deck
```

## Extend the run

Finished images are skipped, so growing the sample only computes the new ones:

```bash
python run_experiment.py --n 50 --part 0 --parts 2 --threads 6 &
python run_experiment.py --n 50 --part 1 --parts 2 --threads 6
python analyze.py && python -m deck.build_deck
```

## Layout

```
advxai/     config.py, data.py, model.py, attacks.py, explain.py, metrics.py
tests/      unit tests (budgets, sparsity, Grad-CAM == CAM identity, metrics)
run_experiment.py   benchmark runner (writes results/raw)
analyze.py          tables, stats, figures (results/*.csv, results/figures/)
deck/               PowerPoint builder (python-pptx); it writes to misc/
```

`data/` (images), `results/raw/` (per-image arrays) and `misc/` (generated documents such as the
PowerPoint deck and meeting minutes) are git-ignored and not committed.
