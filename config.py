import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Attack hyperparameters
EPSILON = 8 / 255  # Perturbation magnitude for FGSM & BIM
ALPHA = 2 / 255  # Step size for BIM
ITERATIONS = 10  # Number of iterations for BIM
K_TOP = 5  # Top-k accuracy evaluation

# Dataset paths
IMAGENET_VAL_PATH = "./data/imagenet_val"

# Visualization settings
SAVE_RESULTS_PATH = "results/"
