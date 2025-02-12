from art.attacks.evasion import SaliencyMapMethod
from art.estimators.classification import PyTorchClassifier
import torch
def jsma_attack(model, img, label):
    classifier = PyTorchClassifier(model=model, loss=torch.nn.CrossEntropyLoss(), input_shape=(3, 224, 224), nb_classes=1000)
    attack = SaliencyMapMethod(classifier=classifier, theta=1, gamma=0.1)
    
    adv_img = attack.generate(x=img.cpu().numpy())
    return torch.tensor(adv_img).to(img.device)
