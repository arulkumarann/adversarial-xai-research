"""Data loading for the benchmark.

Images come from the public repo EliSchwartz/imagenet-sample-images (one
ImageNet image per class, file name = <wnid>_<class_name>.JPEG).  The ImageNet
class index of a file is the position of its WordNet id (wnid) in the sorted
list of the 1000 wnids, which is the ordering torchvision models use.
"""
from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

REPO = "EliSchwartz/imagenet-sample-images"
BRANCH = "master"
TREE_URL = f"https://api.github.com/repos/{REPO}/git/trees/{BRANCH}"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/"

DATA_DIR = Path("data/imagenet_samples")

# 224x224 centre crop, pixels kept in [0, 1]. Mean/std normalisation lives
# INSIDE the model wrapper so that every attack works in [0, 1] pixel space.
PREPROCESS = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "advxai-research"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def list_remote_files() -> list[str]:
    tree = json.loads(_get(TREE_URL))["tree"]
    return sorted(t["path"] for t in tree if t["path"].lower().endswith((".jpeg", ".jpg")))


def download_subset(n: int = 80, seed: int = 42, out_dir: Path = DATA_DIR) -> list[Path]:
    """Download `n` images (random, fixed seed) and the sorted wnid list."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = list_remote_files()
    (out_dir / "all_wnids.json").write_text(
        json.dumps(sorted({f.split("_")[0] for f in files}))
    )
    rng = random.Random(seed)
    chosen = rng.sample(files, n)
    paths = []
    for f in chosen:
        p = out_dir / f
        if not p.exists():
            p.write_bytes(_get(RAW_URL + urllib.request.quote(f)))
        paths.append(p)
    return paths


def wnid_to_index(wnid: str, out_dir: Path = DATA_DIR) -> int:
    wnids = json.loads((out_dir / "all_wnids.json").read_text())
    return wnids.index(wnid)


def load_image(path: Path) -> torch.Tensor:
    """Return a (3,224,224) float tensor in [0,1]."""
    return PREPROCESS(Image.open(path).convert("RGB"))


def label_of(path: Path, out_dir: Path = DATA_DIR) -> int:
    return wnid_to_index(path.name.split("_")[0], out_dir)


def class_name_of(path: Path) -> str:
    return path.stem.split("_", 1)[1].replace("_", " ")
