import unittest

import numpy as np
import torch

from torch_jpeg_blockiness.blockiness import (
    calculate_image_blockiness,
    rgb_to_grayscale,
)
from torch_jpeg_blockiness.original_blockiness import DCT, process_image


class TestImageBlockiness(unittest.TestCase):
    """Unit tests for verifying the consistency of blockiness computations."""

    def test_blockiness_consistency(self) -> None:
        """Test that the torch-based and numpy-based blockiness computations are consistent.

        Multiple fake images of different sizes are generated. The relative difference between
        the torch and numpy implementations is asserted to be small.
        """
        torch.set_float32_matmul_precision("highest")
        torch.set_printoptions(precision=8)

        # Define a list of image sizes (height, width)
        image_sizes: list[tuple[int, int]] = [
            (64, 64),
            (80, 120),
            (128, 128),
            (150, 200),
            (300, 350),
            (1600, 2000),
        ]
        for image_size in image_sizes:
            with self.subTest(image_size=image_size):
                tolerance: float = 1e-4
                height, width = image_size

                # Generate a random RGB image with values in the range [0, 255] and shape (B, C, H, W)
                # Batch size is set to 2.
                random_image: torch.Tensor = torch.randint(
                    0, 256, (2, 3, height, width), dtype=torch.uint8
                )
                random_image = random_image.to(dtype=torch.float32)

                # Convert the random image to grayscale.
                grayscale_image: torch.Tensor = rgb_to_grayscale(random_image)

                # Compute the blockiness metric using the torch-based implementation.
                torch_jpeg_blockiness: torch.Tensor = calculate_image_blockiness(
                    grayscale_image
                )

                # Convert the first image of the batch to a numpy array for numpy processing.
                numpy_image: np.ndarray = grayscale_image[0].squeeze().numpy()

                # Compute the blockiness metric using the numpy-based implementation.
                numpy_blockiness: np.ndarray = process_image(numpy_image, DCT())

                # Convert the torch result to a numpy array for comparison.
                torch_jpeg_blockiness_numpy: np.ndarray = (
                    torch_jpeg_blockiness[0].detach().cpu().numpy()
                )

                # Compute the relative difference between the two implementations.
                epsilon: float = 1e-8
                relative_difference: float = np.abs(
                    torch_jpeg_blockiness_numpy - numpy_blockiness
                ) / (np.abs(numpy_blockiness) + epsilon)

                self.assertTrue(
                    relative_difference < tolerance,
                    (
                        f"Relative difference {relative_difference} exceeds tolerance {tolerance} "
                        f"for image size (height, width)=({height}, {width})."
                    ),
                )


if __name__ == "__main__":
    unittest.main()
