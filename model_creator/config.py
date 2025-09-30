from pathlib import Path


MODEL_SAVE_FREQUENCY = 10


MODEL_NAME = "medium_small_model"

BASE_RESULTS_DIR = Path("results")
RESULTS_DIR = BASE_RESULTS_DIR / MODEL_NAME
RESULTS_PLOTS_DIR = RESULTS_DIR / "plots"
MODELS_DIR = RESULTS_DIR / "models"
COMPARISONS_DIR = RESULTS_DIR / "comparisons"

COMPARISON_SOURCE_IMAGES_DIR = Path("comparison_images")
TRAINING_DATA_DIR = Path("cropped")

LOSS_PLOT_FILENAME = "loss_curve.jpg"
AVERAGE_DIFFERENCE_PLOT_FILENAME = "avg_difference_curve.jpg"
EPOCH_TIME_PLOT_FILENAME = "epoch_time_curve.jpg"

TRAIN_EPOCHS = 120
IMG_DIM = 224