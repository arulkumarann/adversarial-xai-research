from art.attacks.evasion import BasicIterativeMethod
from art.estimators.classification import PyTorchClassifier
import torch

def bim_attack(model, img, label, epsilon, alpha, iterations):
    classifier = PyTorchClassifier(model=model, loss=torch.nn.CrossEntropyLoss(), input_shape=(3, 224, 224), nb_classes=1000)
    attack = BasicIterativeMethod(estimator=classifier, eps=epsilon, eps_step=alpha, max_iter=iterations)
    
    adv_img = attack.generate(x=img.cpu().numpy())
    return torch.tensor(adv_img).to(img.device)
