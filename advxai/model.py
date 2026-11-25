"""DenseNet-121 wrapper.

The ImageNet mean/std normalisation is part of the model, so the input is a
plain [0,1] image.  This keeps epsilon budgets (e.g. 8/255) meaningful and
lets every attack and explanation method share one input space.

DenseNet-121 forward pass (torchvision):
    A = features(x)                       # (B,1024,7,7)   output of norm5
    F = relu(A)
    p = global_avg_pool(F)                # (B,1024)
    logits = classifier(p)                # (B,1000)  = W p + b
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import DenseNet121_Weights, densenet121

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class NormalizedDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.register_buffer("mean", torch.tensor(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(STD).view(1, 3, 1, 1))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalize(x))

    # --- pieces used by CAM / Grad-CAM / latent attack -------------------
    def feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """A = output of norm5 (before the final ReLU), shape (B,1024,7,7)."""
        return self.net.features(self.normalize(x))

    def head(self, A: torch.Tensor) -> torch.Tensor:
        """relu -> global average pool -> linear classifier."""
        Fm = F.relu(A)
        p = F.adaptive_avg_pool2d(Fm, 1).flatten(1)
        return self.net.classifier(p)

    @property
    def classifier_weight(self) -> torch.Tensor:
        return self.net.classifier.weight  # (1000,1024)


def load_model() -> NormalizedDenseNet:
    model = NormalizedDenseNet().to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def predict(model: NormalizedDenseNet, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (predicted class, softmax confidence of that class) for a batch."""
    probs = F.softmax(model(x.to(DEVICE)), dim=1)
    conf, pred = probs.max(1)
    return pred.cpu(), conf.cpu()
