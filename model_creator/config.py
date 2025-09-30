from pathlib import Path


MODEL_SAVE_FREQUENCY = 10

ENCODED_SIZE = 6272

# '6272_0_small_model', "6272_1_medium_small_model", "6272_2_medium_model","6272_3_big_model"
# '9604_0_small_model', "9604_1_medium_small_model", "9604_2_medium_model","9604_3_big_model"
MODEL_NAME ='9604_0_small_model'

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

TRAIN_EPOCHS = 500
IMG_DIM = 224