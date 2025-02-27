import json

import numpy as np
import torch
import torch.fft
import torchvision.transforms.functional

from original_blockiness import DCT, calc_DCT, calc_V, process_image

DEFAULT_BLOCK_SIZE = 8


import numpy as np
import torch
import torchvision


def rgb_to_grayscale(tensor):
    # Define luminance coefficients
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=tensor.device).view(
        1, 3, 1, 1
    )

    # Apply weighted sum across the channel dimension
    grayscale = (tensor * weights).sum(dim=1, keepdim=True)  # Shape: (B, 1, H, W)

    return grayscale


try:
    """
    Author: Ziyang Hu
    Author's repo: https://github.com/zh217/torch-dct
    Licence: MIT
    """
    # PyTorch 1.7.0 and newer versions
    import torch.fft

    def dct1_rfft_impl(x):
        return torch.view_as_real(torch.fft.rfft(x, dim=1))

    def dct_fft_impl(v):
        return torch.view_as_real(torch.fft.fft(v, dim=1))

except ImportError:

    def dct_fft_impl(v):
        return torch.rfft(v, 1, onesided=False)


def dct(x, norm=None):
    """
    Author: Ziyang Hu
    Author's repo: https://github.com/zh217/torch-dct
    Licence: MIT

    Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last dimension
    """
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

    Vc = dct_fft_impl(v)

    k = -torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    width_r = torch.cos(k)
    width_i = torch.sin(k)

    V = Vc[:, :, 0] * width_r - Vc[:, :, 1] * width_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def dct_2d(x, norm=None):
    """
    Author: Ziyang Hu
    Author's repo: https://github.com/zh217/torch-dct
    Licence: MIT

    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)


def calc_margin(
    height: int,
    width: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> tuple[int, int, int, int]:
    """
    Calculate margins for DCT processing.
    """
    height_margin: int = height % block_size
    width_margin: int = width % block_size
    cal_height: int = height - (
        height_margin if height_margin >= 4 else height_margin + block_size
    )
    cal_width: int = width - (
        width_margin if width_margin >= 4 else width_margin + block_size
    )
    height_margin = (height_margin + block_size) if height_margin < 4 else height_margin
    width_margin = (width_margin + block_size) if width_margin < 4 else width_margin
    return cal_height, cal_width, height_margin, width_margin


# def calc_v_torch(
#     dct_img, height_block_num, width_block_num, block_size=DEFAULT_BLOCK_SIZE
# ):
#     dct_img
#     h_block_num = height_block_num
#     w_block_num = width_block_num
#     block_size
#     # Number of blocks (note: h_offsets and w_offsets are built using 1 to h_block_num-2 (exclusive),
#     # which gives h_block_num-3 elements, same for width)
#     num_h = h_block_num - 3
#     num_w = w_block_num - 3

#     # Use the device of the input tensor for all created tensors
#     device = dct_img.device

#     # Compute the starting offset for each block.
#     # Each offset is: block_size + (block_index * block_size), where block_index goes from 1 to h_block_num-2 (exclusive)
#     h_offsets = (
#         block_size + torch.arange(1, h_block_num - 2, device=device) * block_size
#     )  # shape: (num_h,)
#     w_offsets = (
#         block_size + torch.arange(1, w_block_num - 2, device=device) * block_size
#     )  # shape: (num_w,)

#     # Create 4D index arrays for the row (r) and column (c) coordinates.
#     # r will have shape (num_h, 1, block_size, 1)
#     # c will have shape (1, num_w, 1, block_size)
#     # They broadcast to form full index arrays of shape (num_h, num_w, block_size, block_size)
#     r = (
#         h_offsets[:, None, None, None]
#         + torch.arange(block_size, device=device)[None, None, :, None]
#     )
#     c = (
#         w_offsets[None, :, None, None]
#         + torch.arange(block_size, device=device)[None, None, None, :]
#     )

#     # Ensure indices are of integer type (long)
#     r = r.to(dtype=torch.int)
#     c = c.to(dtype=torch.int)
#     dct_img = dct_img.squeeze(0).squeeze(0)

#     # Extract the central value (a) and its four neighbors:
#     # left (b_val), right (c_val), top (d_val), and bottom (e_val).
#     a = dct_img[..., r, c]
#     b_val = dct_img[..., r, c - block_size]
#     c_val = dct_img[..., r, c + block_size]
#     d_val = dct_img[..., r - block_size, c]
#     e_val = dct_img[..., r + block_size, c]

#     # Compute V for each block and each pixel in the block.
#     V = torch.sqrt((b_val + c_val - 2 * a) ** 2 + (d_val + e_val - 2 * a) ** 2)

#     # Average V over all blocks (i.e. over the first two dimensions)
#     V_average = V.sum(dim=(0, 1)) / ((h_block_num - 2) * (w_block_num - 2))

#     return V_average


def calc_v_torch(
    dct_img,
    height_block_num: int,
    width_block_num: int,
    block_size=DEFAULT_BLOCK_SIZE,
):
    # If input is (B,1,H,W), squeeze the channel dimension but keep the batch.
    if dct_img.dim() == 4:
        dct_img = dct_img.squeeze(1)  # now shape is (B, H, W)

    # Use the provided block numbers.
    h_block_num = height_block_num
    w_block_num = width_block_num
    # Note: Offsets are built from indices 1 to h_block_num-2 (exclusive)
    # which produces (h_block_num - 3) values, same for width.
    num_h = h_block_num - 3
    num_w = w_block_num - 3

    device = dct_img.device

    # Compute the starting offsets for each block.
    # Each offset is: block_size + (index * block_size), with index from 1 to (h_block_num-2)-1.
    h_offsets = (
        block_size + torch.arange(1, h_block_num - 2, device=device) * block_size
    )  # shape: (num_h,)
    w_offsets = (
        block_size + torch.arange(1, w_block_num - 2, device=device) * block_size
    )  # shape: (num_w,)

    # Build the row and column indices.
    # r will be built by adding torch.arange(block_size) to each h_offset,
    # and c similarly from each w_offset.
    # We first reshape the offsets so that broadcasting yields the desired (num_h, num_w, block_size, block_size) shape.
    r = h_offsets.view(num_h, 1, 1, 1) + torch.arange(block_size, device=device).view(
        1, 1, block_size, 1
    )
    c = w_offsets.view(1, num_w, 1, 1) + torch.arange(block_size, device=device).view(
        1, 1, 1, block_size
    )
    # Broadcast to full grid shape:
    r = r.expand(num_h, num_w, block_size, block_size).to(torch.int)
    c = c.expand(num_h, num_w, block_size, block_size).to(torch.int)

    # dct_img now has shape (B, H, W)
    B = dct_img.size(0)
    # Create a batch index tensor of shape (B, 1, 1, 1, 1) so we can index into the first dimension.
    batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1, 1)
    # Expand r and c to include the batch dimension:
    # Final shape: (B, num_h, num_w, block_size, block_size)
    r_exp = r.unsqueeze(0).expand(B, num_h, num_w, block_size, block_size)
    c_exp = c.unsqueeze(0).expand(B, num_h, num_w, block_size, block_size)

    # Now extract the central value (a) and its four neighbors:
    # left (b_val), right (c_val), top (d_val), and bottom (e_val).
    a = dct_img[batch_idx, r_exp, c_exp]
    b_val = dct_img[batch_idx, r_exp, c_exp - block_size]
    c_val = dct_img[batch_idx, r_exp, c_exp + block_size]
    d_val = dct_img[batch_idx, r_exp - block_size, c_exp]
    e_val = dct_img[batch_idx, r_exp + block_size, c_exp]

    # Compute V for each block and pixel within the block.
    V = torch.sqrt((b_val + c_val - 2 * a) ** 2 + (d_val + e_val - 2 * a) ** 2)

    # Average V over all blocks (dimensions 1 and 2) for each image.
    # (The original code divided by ((h_block_num - 2) * (w_block_num - 2)); we preserve that normalization.)
    V_average = V.sum(dim=(1, 2)) / ((h_block_num - 2) * (w_block_num - 2))

    return V_average


def blockwise_dct(
    gray_imgs: torch.Tensor,
    height_block_num: int,
    width_block_num: int,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> torch.Tensor:
    """Compute the DCT of an image block-wise using batched processing.

    This function divides the image into non-overlapping blocks of size block_size x block_size,
    applies a batched DCT transform (using dct2, which accepts batched input), and reconstructs the
    DCT image from the transformed blocks.

    Args:
        gray_imgs: Input image tensor with shape (H, W).
        dct: A DCT object with a method dct2 that accepts batched input.
        height_block_num: Number of blocks along the height.
        width_block_num: Number of blocks along the width.

    Returns:
        A tensor containing the DCT coefficients of the image blocks, arranged in the original block layout.
    """
    batch_size, channel_dim, h, w = gray_imgs.shape
    # assert channel_dim == 1, "Input image must be grayscale."
    if (
        gray_imgs.shape[-2] < height_block_num * block_size
        or gray_imgs.shape[-1] < width_block_num * block_size
    ):
        raise ValueError(f"Invalid image dimensions.{gray_imgs.shape}")

    # Divide the image into blocks of shape (height_block_num, width_block_num, block_size, block_size).
    blocks = gray_imgs.unfold(
        -2,
        block_size,
        block_size,
    ).unfold(
        -1,
        block_size,
        block_size,
    )
    blocks = blocks.contiguous().view(batch_size, -1, block_size, block_size)
    # Apply the batched DCT transform to all blocks at once.
    dct_blocks_flat: torch.Tensor = dct_2d(blocks, norm="ortho")

    dct_blocks = dct_blocks_flat.view(
        batch_size,
        height_block_num,
        width_block_num,
        block_size,
        block_size,
    )
    # dct_blocks = dct_blocks.permute(0, 2, 1, 3).contiguous()
    dct_blocks = dct_blocks.permute(0, 1, 3, 2, 4).contiguous()
    dct_blocks = dct_blocks.view(
        batch_size,
        height_block_num * block_size,
        width_block_num * block_size,
    )
    return dct_blocks


def caculate_image_blockiness(
    gray_images: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    assert gray_images.dim() == 4, "Input tensor must have shape (B, C, H, W)."
    height, width = gray_images.shape[-2:]
    cal_height, cal_width, height_margin, width_margin = calc_margin(
        height=height, width=width
    )

    height_block_num, width_block_num = (
        cal_height // block_size,
        cal_width // block_size,
    )

    gray_tensor_cut = gray_images[..., :cal_height, :cal_width]
    gray_offset = torch.zeros_like(gray_images)
    gray_offset[..., :-4, :-4] = gray_images[..., 4:, 4:]
    gray_offset = gray_offset[..., :cal_height, :cal_width]
    ctx = dict(
        height_block_num=height_block_num,
        width_block_num=width_block_num,
        gray_tensor_cut=gray_tensor_cut.shape,
        cal_height=cal_height,
        cal_width=cal_width,
        gray_offset=gray_offset.shape,
    )
    print(json.dumps(ctx, indent=4))

    dct_imgs = blockwise_dct(
        gray_imgs=gray_tensor_cut,
        height_block_num=height_block_num,
        width_block_num=width_block_num,
        block_size=block_size,
    )

    dct_offset_imgs = blockwise_dct(
        gray_imgs=gray_offset,
        height_block_num=height_block_num,
        width_block_num=width_block_num,
        block_size=block_size,
    )
    input(dct_imgs.shape)
    v_average = calc_v_torch(
        dct_img=dct_imgs,
        height_block_num=height_block_num,
        width_block_num=width_block_num,
        block_size=block_size,
    )
    v_offset_average = calc_v_torch(
        dct_img=dct_offset_imgs,
        height_block_num=height_block_num,
        width_block_num=width_block_num,
        block_size=block_size,
    )
    print(v_average.shape)
    d = torch.abs((v_offset_average - v_average)) / v_average
    print("D", d.shape)
    d_sum = torch.sum(d, dim=(1, 2))
    return d_sum


if __name__ == "__main__":
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
