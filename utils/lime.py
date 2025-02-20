import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries

def generate_lime_explanation(model, image_tensor):
    """
    Generates LIME explanation for a given image and model.
    """
    def batch_predict(images):
        images = torch.tensor(images).permute(0, 3, 1, 2).float().to(DEVICE)
        with torch.no_grad():
            logits = model(images)
            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
        return probs

    # Convert image tensor to numpy format
    image = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    
    # Define LIME explainer
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(image, batch_predict, top_labels=1, hide_color=0, num_samples=1000)
    
    # Get the explanation mask
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=False)
    lime_result = mark_boundaries(temp, mask)

    return lime_result
