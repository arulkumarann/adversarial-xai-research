def save_comparison(original_img, adversarial_img, original_cam, adversarial_cam, noise, output_path, original_pred=None, adversarial_pred=None):
    """
    Save a comparison of original and adversarial images with their GradCAM visualizations.
    
    Args:
        original_img: Original input image
        adversarial_img: Adversarial image
        original_cam: GradCAM visualization for original image
        adversarial_cam: GradCAM visualization for adversarial image
        noise: Difference between original and adversarial images
        output_path: Path to save the comparison image
        original_pred: Original prediction (optional)
        adversarial_pred: Adversarial prediction (optional)
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 8))
    
    # Plot original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_img.permute(1, 2, 0).cpu().numpy())
    plt.title(f'Original\nPred: {original_pred}' if original_pred is not None else 'Original')
    plt.axis('off')
    
    # Plot adversarial image
    plt.subplot(2, 3, 2)
    plt.imshow(adversarial_img.permute(1, 2, 0).cpu().numpy())
    plt.title(f'Adversarial\nPred: {adversarial_pred}' if adversarial_pred is not None else 'Adversarial')
    plt.axis('off')
    
    # Plot noise
    plt.subplot(2, 3, 3)
    plt.imshow(noise, cmap='viridis')
    plt.title('Perturbation')
    plt.axis('off')
    
    # Plot original GradCAM
    plt.subplot(2, 3, 4)
    plt.imshow(original_cam, cmap='jet')
    plt.title('Original GradCAM')
    plt.axis('off')
    
    # Plot adversarial GradCAM
    plt.subplot(2, 3, 5)
    plt.imshow(adversarial_cam, cmap='jet')
    plt.title('Adversarial GradCAM')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()