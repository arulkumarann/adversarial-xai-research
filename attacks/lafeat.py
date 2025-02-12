from art.attacks.evasion import ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
import torch

def lafeat_attack(model, img, label):
    classifier = PyTorchClassifier(model=model, loss=torch.nn.CrossEntropyLoss(), input_shape=(3, 224, 224), nb_classes=1000)
    attack = ProjectedGradientDescent(estimator=classifier, eps=0.3, eps_step=0.1, max_iter=40)
    
    adv_img = attack.generate(x=img.cpu().numpy())
    return torch.tensor(adv_img).to(img.device)
