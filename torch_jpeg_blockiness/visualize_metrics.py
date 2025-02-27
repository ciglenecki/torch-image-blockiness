import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torchvision

from torch_jpeg_blockiness.blockiness import (
    calculate_image_blockiness,
    rgb_to_grayscale,
)

# Constants
DEFAULT_BLOCK_SIZE = 8
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_TRIPLEX


def add_text_to_image(
    img: np.ndarray,
    text: str,
    font: int = FONT,
) -> np.ndarray:
    """Annotate an image with the provided text using an outlined effect.

    The text is drawn twice to produce an outline: first with a thicker white stroke,
    then overlaid with a thinner black stroke.

    Args:
        img: The image on which to draw text.
        text: The text to annotate the image with.
        position: A tuple (x, y) indicating the position of the text.
        font: The font type to use.
        font_scale: The scale (size) of the font.
        font_thickness: The thickness of the font strokes.

    Returns:
        The image with text annotation.
    """
    try:
        # Draw the outline (white, thicker)
        img_width = img.shape[1]
        img_width_scale = img_width / 500
        position = (30, 150 + int(img_width_scale * 20))
        font_scale = 1.4 * img_width_scale + 0.3
        font_thickness = int(font_scale * 2 + 2)
        img_with_text = cv2.putText(
            img.copy(),
            text,
            position,
            font,
            font_scale,
            BLACK,
            font_thickness + 10,
            cv2.LINE_AA,
        )
        # Draw the main text (black, thinner)
        img_with_text = cv2.putText(
            img_with_text,
            text,
            position,
            font,
            font_scale,
            WHITE,
            font_thickness,
            cv2.LINE_AA,
        )
        return img_with_text
    except Exception as error:
        raise RuntimeError("Failed to add text to image.") from error


def process_and_save_image(
    input_path: Path, output_path: Path, block_size: int = DEFAULT_BLOCK_SIZE
) -> None:
    """Process a single image to compute its blockiness and save the annotated result.

    The image is loaded from the input path, converted to grayscale for metric computation,
    annotated with the blockiness score, and saved to the output path.

    Args:
        input_path: Path to the input image file.
        output_path: Path where the processed image will be saved.
        block_size: Block size parameter used in computing the blockiness metric.
    """
    if not input_path.is_file():
        raise FileNotFoundError(f"Input image file {input_path} does not exist.")

    img = torchvision.io.read_image(str(input_path))
    img_gray = rgb_to_grayscale(img)
    blockiness = calculate_image_blockiness(img_gray)
    text = f"Blockiness: {float(blockiness):.2f}"

    cv2_image = img.permute(1, 2, 0).numpy()
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    img_with_text = add_text_to_image(cv2_image, text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), img_with_text):
        raise RuntimeError(f"Failed to save processed image to {output_path}.")


def process_images_in_directory(
    input_dir: Path, output_dir: Path, block_size: int = DEFAULT_BLOCK_SIZE
) -> None:
    """Recursively process all image files in the input directory and save annotated images.

    The output directory will mirror the directory structure of the input directory.

    Args:
        input_dir: Directory containing the input images.
        output_dir: Directory where processed images will be saved.
        block_size: Block size parameter for computing the blockiness metric.
    """
    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input directory {input_dir} does not exist or is not a directory."
        )

    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    for img_path in input_dir.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            relative_path = img_path.relative_to(input_dir)
            output_file = output_dir / relative_path
            try:
                process_and_save_image(img_path, output_file, block_size)
                logging.info(f"Processed and saved image: {output_file}")
            except Exception as error:
                logging.error(f"Error processing image {img_path}: {error}")


def main() -> None:
    """Main function to load images from a directory, compute blockiness, annotate, and save them."""
    parser = argparse.ArgumentParser(
        description="Load images from an input directory, compute blockiness score, "
        "annotate the images, and save them to an output directory."
    )
    parser.add_argument(
        "--input_dir",
        help="Directory containing input images.",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to save annotated images.",
        required=True,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=DEFAULT_BLOCK_SIZE,
        help="Block size for computing blockiness metric.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        process_images_in_directory(input_dir, output_dir, args.block_size)
    except Exception as error:
        logging.critical(f"Critical error encountered: {error}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
