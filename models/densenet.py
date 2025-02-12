import torch
import torchvision.models as models
from config import DEVICE

def load_model():
    model = models.densenet121(pretrained=True).eval().to(DEVICE)
    return model
