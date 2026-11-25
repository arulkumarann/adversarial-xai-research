"""All experiment hyper-parameters in one place."""

SEED = 42
N_IMAGES = 20          # correctly-classified images in the reported run (extend with --n 50)
N_DOWNLOAD = 80        # images downloaded (some are misclassified and dropped)

# ---- attack budgets (pixel space [0,1]) ------------------------------
EPS = 8 / 255          # L_inf budget shared by FGSM, BIM, latent PGD and the noise control
BIM_ALPHA = 2 / 255    # BIM step size
BIM_STEPS = 10
LATENT_ALPHA = 1 / 255
LATENT_STEPS = 20
JSMA_GAMMA = 0.02      # max fraction of the 150,528 input values JSMA may change
JSMA_K = 20            # features changed per JSMA iteration
JSMA_THETA = 1.0       # amount added to each chosen feature (then clipped to 1)

# ---- explanation settings --------------------------------------------
GRADCAM_LAYER = "denseblock4"     # last dense block (7x7x1024)
GRADCAM_LAYER_MID = "denseblock3" # supplementary, 14x14x1024
LIME_SAMPLES = 300
LIME_SEGMENTS = 50
SHAP_EVALS = 400
SHAP_BLUR = 32

# ---- explanation-similarity settings ---------------------------------
TOPK_FRAC = 0.20       # IoU of the top-20 % most important pixels

# ---- epsilon sweep (the "I0 -> In" experiment) -----------------------
SWEEP_EPS = [1 / 255, 2 / 255, 4 / 255, 8 / 255, 16 / 255]

ATTACKS = ("noise", "fgsm", "bim", "jsma", "latent")
METHODS = ("gradcam", "cam", "lime", "shap")
