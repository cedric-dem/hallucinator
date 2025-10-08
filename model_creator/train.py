"""Entry point for training the image denoising model."""

from __future__ import annotations

from model_creator.misc import train


def main() -> None:
    """Run the training routine defined in :mod:`model_creator.misc`."""

    train()


if __name__ == "__main__":
    main()
