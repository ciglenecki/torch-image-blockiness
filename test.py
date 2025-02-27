torch.set_float32_matmul_precision("highest")
torch.set_printoptions(precision=8)

for i in ["", 80, 60]:
    img = torchvision.io.read_image(f"unsplash{i}.jpg")
    img = torch.stack([img, img], dim=0)
    img_gray = rgb_to_grayscale(img)
    img_npy = img_gray[0].squeeze().numpy()
    tb = caculate_image_blockiness(img_gray)

    nb = process_image(img_npy, DCT())
    print()
    print("torch:", tb)
    print("npy:", nb)
    print()
