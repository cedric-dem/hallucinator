
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib
matplotlib.use("Agg")

DATASET_DIRECTORIES: Sequence[Path] = (Path("cropped_subset"),)

####################################

MULTI_STEP: bool = True

if MULTI_STEP:
    # takes more space,
    MODEL_NAME: str = "small_model"
    BATCH_SIZE: int = 7

else:
    # takes less space,
    MODEL_NAME: str = "huge_model"
    BATCH_SIZE: int = 16

EPOCHS: int = 20

HALLUCINATION_SEQUENCE_COUNT: int = 10
HALLUCINATION_SEQUENCE_LENGTH: int = 32

DENOISING_SEQUENCE_PASSES: int = 15

TRAINING_REFINEMENT_STEPS: int = DENOISING_SEQUENCE_PASSES
INFERENCE_REFINEMENT_STEPS: int = max(
    HALLUCINATION_SEQUENCE_LENGTH, DENOISING_SEQUENCE_PASSES
)

####################################

IMAGE_EXTENSIONS: Sequence[str] = (".jpg", ".jpeg")

RESULTS_DIRECTORY: Path = Path("results")
MODEL_FILENAME: str = "denoiser_autoencoder.keras"
PLOTS_DIRECTORY_NAME: str = "plots"
HALLUCINATION_DIRECTORY_NAME: str = "hallucinated_images"
DENOISED_DIRECTORY_NAME: str = "denoised_images"

BENCHMARK_DIRECTORY: Path = Path("benchmark")

IMAGE_HEIGHT: int = 224
IMAGE_WIDTH: int = 224
IMAGE_CHANNELS: int = 3
VALIDATION_SPLIT: float = 0.1
SHUFFLE_BUFFER_SIZE: int = 2048
RANDOM_SEED: int = 1337

NOISE_STDDEV: float = 0.5
NOISE_BLEND_MIN: float = 0.0
NOISE_BLEND_MAX: float = 1.0

ModelConfig = dict[str, Sequence[int] | int]

MODEL_CONFIGURATIONS: dict[str, ModelConfig] = {
    "huge_model": {
        "down_filters": (64, 128, 256),
        "bottleneck_filters": 512,
        "up_filters": (256, 128, 64),
    },
    "large_model": {
        "down_filters": (32, 64, 128),
        "bottleneck_filters": 256,
        "up_filters": (128, 64, 32),
    },
    "medium_model": {
        "down_filters": (8, 16, 32),
        "bottleneck_filters": 64,
        "up_filters": (32, 16, 8),
    },
    "small_model": { #biggest possible with 7 epoch mon my gpu
        "down_filters": (6, 12, 24),
        "bottleneck_filters": 48,
        "up_filters": (24, 12, 6),
    },
    "tiny_model": { #biggest possible with 8 epoch mon my gpu
        "down_filters": (4, 8, 16),
        "bottleneck_filters": 32,
        "up_filters": (16, 8, 4),
    },
}