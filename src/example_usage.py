import torchvision
import torchvision.transforms.functional

from src.torch_blockiness import calculate_image_blockiness, rgb_to_grayscale


def main():
    img = torchvision.io.read_image("example_images/unsplash.jpg")
    img_gray = rgb_to_grayscale(img)
    blockiness = calculate_image_blockiness(img_gray)
    blockiness_float = float(blockiness)
    print(f"Blockiness: {blockiness_float:.2f}")


if __name__ == "__main__":
    main()
