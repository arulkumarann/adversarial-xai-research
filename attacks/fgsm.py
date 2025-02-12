import torch
import numpy as np
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

def fgsm_attack(model, img, label, epsilon):
    classifier = PyTorchClassifier(model=model, loss=torch.nn.CrossEntropyLoss(), input_shape=(3, 224, 224), nb_classes=1000)
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    
    adv_img = attack.generate(x=img.cpu().numpy())
    return torch.tensor(adv_img).to(img.device)
