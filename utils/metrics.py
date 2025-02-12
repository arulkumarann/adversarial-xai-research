import torch
import numpy as np
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def top_k_accuracy(logits, labels, k=5):
    """Computes Top-k accuracy."""
    top_k_preds = torch.topk(logits, k, dim=1).indices
    correct_top_k = (top_k_preds == labels.view(-1, 1)).any(dim=1).float()
    return correct_top_k.mean().item()
def compute_metrics(model, original, adversarial, label):
    with torch.no_grad():
        original_logits = model(original)
        adversarial_logits = model(adversarial)

    # Top-1 and Top-5 Accuracy
    top1_orig = top_k_accuracy(original_logits, label, k=1)
    top5_orig = top_k_accuracy(original_logits, label, k=5)
    top1_adv = top_k_accuracy(adversarial_logits, label, k=1)
    top5_adv = top_k_accuracy(adversarial_logits, label, k=5)

    # Fooling Rate
    fooling_rate = 1 - (torch.argmax(original_logits, dim=1) == torch.argmax(adversarial_logits, dim=1)).float().mean()

    # L2 Norm and L∞ Norm
    l2_norm = torch.norm(original - adversarial, p=2).item()
    l_inf_norm = torch.norm(original - adversarial, p=float("inf")).item()

    # Convert images to NumPy arrays for SSIM and PSNR
    original_np = original.squeeze().permute(1, 2, 0).cpu().numpy()
    adversarial_np = adversarial.squeeze().permute(1, 2, 0).cpu().numpy()

    # Compute PSNR and SSIM
    psnr_value = psnr(original_np, adversarial_np, data_range=1.0)
    
    # Add win_size parameter for SSIM
    try:
        ssim_value = ssim(
            original_np, 
            adversarial_np, 
            channel_axis=2,  # Specify channel axis for RGB images
            data_range=1.0,
            win_size=7  # Set minimum window size
        )
    except ValueError:
        # If images are too small, return None for SSIM
        ssim_value = None

    return {
        "Top-1 Accuracy (Orig)": top1_orig,
        "Top-5 Accuracy (Orig)": top5_orig,
        "Top-1 Accuracy (Adv)": top1_adv,
        "Top-5 Accuracy (Adv)": top5_adv,
        "Fooling Rate": fooling_rate.item(),
        "L2 Norm": l2_norm,
        "L∞ Norm": l_inf_norm,
        "PSNR": psnr_value,
        "SSIM": ssim_value
    }