import torch
import torchvision.transforms as transforms
import os
import json
import pandas as pd
from PIL import Image
from datetime import datetime
from models.densenet import load_model
from attacks.fgsm import fgsm_attack
from attacks.bim import bim_attack
from attacks.jsma import jsma_attack
from attacks.lafeat import lafeat_attack
from utils.gradcam import GradCAM
from utils.visualize import save_comparison
from utils.metrics import compute_metrics, top_k_accuracy
from config import DEVICE, EPSILON, ALPHA, ITERATIONS, K_TOP
import requests

# Load ImageNet class mappings
IMAGENET_CLASSES_URL = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
try:
    response = requests.get(IMAGENET_CLASSES_URL)
    class_idx = response.json()
    idx_to_class = {int(k): v[1] for k, v in class_idx.items()}  # Using human-readable labels
except:
    print("Warning: Could not fetch ImageNet classes. Using index numbers instead.")
    idx_to_class = lambda x: f"Class {x}"

class ImageFolderDataset:
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0, self.image_files[idx]  # Return filename as well

def save_metrics_to_csv(metrics_dict, image_name, attack_name, csv_path="results/metrics_log.csv"):
    """Save metrics to CSV file, creating new or appending to existing"""
    # Ensure all required metrics exist with default values
    default_metrics = {
        'L2_Norm': 0.0,
        'Linf_Norm': 0.0,
        'SSIM': 0.0,
        'PSNR': 0.0,
        'Original Top-1': 0.0,
        'Adversarial Top-1': 0.0,
        'Original Top-5': 0.0,
        'Adversarial Top-5': 0.0,
        'original_class': '',
        'adversarial_class': ''
    }
    
    # Update default values with actual metrics
    for key in metrics_dict:
        # Convert keys to match the format in default_metrics
        normalized_key = key.replace(' ', '_')
        default_metrics[normalized_key] = metrics_dict[key]
    
    # Prepare data for the row
    row_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_name': image_name,
        'attack_type': attack_name,
        'original_class': default_metrics['original_class'],
        'adversarial_class': default_metrics['adversarial_class'],
        'original_top1': default_metrics['Original_Top-1'],
        'adversarial_top1': default_metrics['Adversarial_Top-1'],
        'original_top5': default_metrics['Original_Top-5'],
        'adversarial_top5': default_metrics['Adversarial_Top-5'],
        'l2_norm': default_metrics['L2_Norm'],
        'linf_norm': default_metrics['Linf_Norm'],
        'ssim': default_metrics['SSIM'],
        'psnr': default_metrics['PSNR']
    }
    
    # Convert to DataFrame
    df_new = pd.DataFrame([row_data])
    
    # If file exists, append; otherwise create new
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    # Save to CSV
    df_combined.to_csv(csv_path, index=False)

# Ensure results directory exists
os.makedirs("results", exist_ok=True)

# Load model
model = load_model()
model.eval()
target_layer = model.features[-1]
grad_cam = GradCAM(model, target_layer)

# Load dataset from directory
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Replace this path with your images directory
dataset = ImageFolderDataset(root_dir=r"/teamspace/studios/this_studio/xai-fgsm/src/data/imagenet", transform=transform)
img, _, img_name = dataset[5]  # Get image and its filename
img = img.unsqueeze(0).to(DEVICE)
# Using 0 as dummy label since we don't have actual labels
label = torch.tensor([0]).to(DEVICE)

# Get original model predictions
with torch.no_grad():
    logits = model(img)
    orig_top1 = top_k_accuracy(logits, label, k=1)
    orig_top5 = top_k_accuracy(logits, label, k=5)
    orig_pred = torch.argmax(logits, dim=1).item()
    orig_class_name = idx_to_class[orig_pred]

# Dictionary to store attacks
attacks = {
    "fgsm": lambda: fgsm_attack(model, img, label, EPSILON),
    "bim": lambda: bim_attack(model, img, label, EPSILON, ALPHA, ITERATIONS),
    "jsma": lambda: jsma_attack(model, img, label),
    "lafeat": lambda: lafeat_attack(model, img, label),
}

# Iterate through attacks
for attack_name, attack_func in attacks.items():
    print(f"Running {attack_name.upper()} attack...")

    # Generate adversarial image
    adv_img = attack_func()

    # Get adversarial predictions
    with torch.no_grad():
        adv_logits = model(adv_img)
        adv_top1 = top_k_accuracy(adv_logits, label, k=1)
        adv_top5 = top_k_accuracy(adv_logits, label, k=5)
        adv_pred = torch.argmax(adv_logits, dim=1).item()
        adv_class_name = idx_to_class[adv_pred]

    # Compute Grad-CAM
    cam_original = grad_cam.generate_cam(img, label)
    cam_adv = grad_cam.generate_cam(adv_img, label)

    # Compute noise
    noise = (adv_img - img).squeeze().permute(1, 2, 0).cpu().numpy()

    # Save results
    save_comparison(
        img.squeeze(), adv_img.squeeze(), cam_original, cam_adv, noise, 
        f"results/{attack_name}_result.png", orig_class_name, adv_class_name
    )

    # Compute metrics
    metrics = compute_metrics(model, img, adv_img, label)
    metrics.update({
        "Original Top-1": orig_top1,
        "Adversarial Top-1": adv_top1,
        "Original Top-5": orig_top5,
        "Adversarial Top-5": adv_top5,
        "original_class": orig_class_name,
        "adversarial_class": adv_class_name
    })
    
    # Save metrics to CSV
    save_metrics_to_csv(metrics, img_name, attack_name)
    
    print(f"{attack_name.upper()} attack complete. Results saved in results/{attack_name}_result.png")
    print(f"Metrics: {metrics}")

print("All attacks completed successfully. Metrics have been saved to results/metrics_log.csv")
