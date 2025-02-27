import torchvision

from torch_blockiness.blockiness import (
    calculate_image_blockiness,
    rgb_to_grayscale,
)


def main():
    img = torchvision.io.read_image("example_images/unsplash60.jpg")
    img_gray = rgb_to_grayscale(img)
    blockiness = calculate_image_blockiness(img_gray)
    blockiness_float = float(blockiness)
    print(f"Blockiness: {blockiness_float:.2f}")


if __name__ == "__main__":
    main()
