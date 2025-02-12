import torch
import cv2
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.feature_maps = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate_cam(self, image, class_idx):
        self.model.zero_grad()
        output = self.model(image)
        class_score = output[:, class_idx]
        class_score.backward()

        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        cam = torch.nn.functional.relu(cam).squeeze().cpu().detach().numpy()
        cam = cv2.resize(cam, (224, 224))  # Resize to original image size
        #cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize
        denom = np.max(cam) - np.min(cam)
        if denom > 0:  # Only normalize if there's a non-zero denominator
            cam = (cam - np.min(cam)) / denom
        else:
            cam = np.zeros_like(cam)  # If all values are the same, return zeros
        return cam
