
"""Configuration values for the image denoiser training pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

DATASET_DIRECTORIES: Sequence[Path] = (Path("cropped_subset"),)

EPOCHS: int = 10

# Size of the mini-batches used during training.
BATCH_SIZE: int = 8

#############################################################################################

# Supported image file extensions.
IMAGE_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg", ".png")

# Ratio of images that should be used for validation instead of training.
VALIDATION_SPLIT: float = 0.1

# Random seed used for shuffling and augmentation. Setting this to a constant
# ensures that runs are reproducible.
RANDOM_SEED: int = 1337

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

# Whether the denoising network should be applied repeatedly during training.
MULTI_STEP: bool = True

# Number of passes performed when ``MULTI_STEP`` is enabled.
DENOISING_SEQUENCE_PASSES: int = 10

# Learning rate for the optimiser.
LEARNING_RATE: float = 1e-3

# Standard deviation of the Gaussian noise injected into the clean images. The
# noise is added in the ``[0, 1]`` floating point colour space, therefore values
# in the range of ``0.05`` – ``0.2`` tend to work well.
NOISE_STDDEV: float = 0.1

# ---------------------------------------------------------------------------
# Output locations
# ---------------------------------------------------------------------------

RESULTS_DIRECTORY: Path = Path("results")
MODEL_FILENAME: str = "denoiser_autoencoder.keras"
PLOTS_DIRECTORY_NAME: str = "plots"
COMPARISON_DIRECTORY_NAME: str = "comparison"
HALLUCINATED_DIRECTORY_NAME: str = "hallucinated_images"

# Directories containing images that should be rendered for visual comparisons
# after every epoch.
BENCHMARK_DIRECTORIES: Sequence[Path] = (Path("benchmark"),)

# Target image size used by the autoencoder. Images are resized to this shape
# during training.
IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3

MODEL_SIZE: str = "model_small"