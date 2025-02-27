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


# def calc_V_torch(dct_img, height_block_num, width_block_num, block_size):
#     device = "cpu"
#     # The original loops run for height_block in range(1, height_block_num-2) and
#     # width_block in range(1, width_block_num-2); note that the number of iterations
#     # is (height_block_num-3) and (width_block_num-3), but the average is divided by
#     # (height_block_num-2)*(width_block_num-2). We'll follow the same convention.
#     n_h = height_block_num - 3  # number of blocks vertically
#     n_w = width_block_num - 3  # number of blocks horizontally
#     denom = (height_block_num - 2) * (width_block_num - 2)

#     # Compute the “base” index for each inner block.
#     # For each block index in the loop, the row index used for 'a' is:
#     #    height_idx = block_size + height_block * block_size + j
#     # so we precompute:
#     height_blocks = (
#         block_size + torch.arange(1, height_block_num - 2, device=device) * block_size
#     )  # shape: (n_h,)
#     width_blocks = (
#         block_size + torch.arange(1, width_block_num - 2, device=device) * block_size
#     )  # shape: (n_w,)

#     # Create a meshgrid for the block-local offsets (j,i) in [0, block_size)
#     off_h, off_w = torch.meshgrid(
#         torch.arange(block_size, device=device),
#         torch.arange(block_size, device=device),
#         indexing="ij",
#     )
#     # off_h, off_w: shape (block_size, block_size)

#     # For each inner block, compute the grid indices for the center "a" pixel.
#     # The full indices will have shape (n_h, n_w, block_size, block_size).
#     a_h_idx = height_blocks.view(n_h, 1, 1, 1) + off_h.view(1, 1, block_size, block_size)
#     a_w_idx = width_blocks.view(1, n_w, 1, 1) + off_w.view(1, 1, block_size, block_size)
#     a_h_idx = a_h_idx.expand(n_h, n_w, block_size, block_size)
#     a_w_idx = a_w_idx.expand(n_h, n_w, block_size, block_size)
#     a = dct_img[a_h_idx, a_w_idx]

#     # For the neighbors, adjust the base indices:
#     # Left neighbor: subtract block_size from the w index.
#     b_w_idx = (width_blocks - block_size).view(1, n_w, 1, 1) + off_w.view(
#         1, 1, block_size, block_size
#     )
#     b_w_idx = b_w_idx.expand(n_h, n_w, block_size, block_size)
#     b = dct_img[a_h_idx, b_w_idx]

#     # Right neighbor: add block_size to the w index.
#     c_w_idx = (width_blocks + block_size).view(1, n_w, 1, 1) + off_w.view(
#         1, 1, block_size, block_size
#     )
#     c_w_idx = c_w_idx.expand(n_h, n_w, block_size, block_size)
#     c = dct_img[a_h_idx, c_w_idx]

#     # Upper neighbor: subtract block_size from the h index.
#     d_h_idx = (height_blocks - block_size).view(n_h, 1, 1, 1) + off_h.view(
#         1, 1, block_size, block_size
#     )
#     d_h_idx = d_h_idx.expand(n_h, n_w, block_size, block_size)
#     d = dct_img[d_h_idx, a_w_idx]

#     # Lower neighbor: add block_size to the h index.
#     e_h_idx = (height_blocks + block_size).view(n_h, 1, 1, 1) + off_h.view(
#         1, 1, block_size, block_size
#     )
#     e_h_idx = e_h_idx.expand(n_h, n_w, block_size, block_size)
#     e = dct_img[e_h_idx, a_w_idx]

#     # Compute V for each block and each (j,i)
#     V = torch.sqrt((b + c - 2 * a) ** 2 + (d + e - 2 * a) ** 2)
#     # Sum V over all inner blocks (dimensions 0 and 1)

#     V_sum = V.sum(dim=(0, 1))
#     V_average = V_sum / denom
#     return V_average


# def calc_v_torch(
#     dct_img: torch.Tensor,
#     height_block_num: int,
#     width_block_num: int,
#     block_size: int = DEFAULT_BLOCK_SIZE,
# ):
#     """
#     Compute the average gradient magnitude V for each image in a batched DCT image tensor.

#     Parameters:
#         dct_img: Tensor of shape (B, H, W) where B is the batch size.
#         height_block_num: Number of blocks along the height.
#         width_block_num: Number of blocks along the width.
#         block_size: Size of each block.

#     Returns:
#         V_average: Tensor of shape (B, block_size, block_size) with the averaged V for each image.
#     """
#     device = dct_img.device
#     b, c, h, w = dct_img.shape
#     assert c == 1, "Input image must be grayscale."
#     dct_img = dct_img.squeeze(1)
#     # The loops iterate for height_block in range(1, height_block_num-2) and width_block in range(1, width_block_num-2),
#     # so there are (height_block_num - 3) and (width_block_num - 3) inner blocks.
#     # The average is computed with denominator (height_block_num - 2)*(width_block_num - 2).
#     n_h = height_block_num - 3  # number of valid blocks vertically
#     n_w = width_block_num - 3  # number of valid blocks horizontally

#     denom = (height_block_num - 2) * (width_block_num - 2)

#     B = dct_img.shape[0]  # batch size

#     # Compute the "base" index for each inner block along each dimension.
#     # For a given block, the center pixel (a) has indices:
#     #   row: block_size + height_block * block_size + j
#     #   col: block_size + width_block * block_size + i
#     height_blocks = (
#         block_size + torch.arange(1, height_block_num - 2, device=device) * block_size
#     )  # shape: (n_h,)
#     width_blocks = (
#         block_size + torch.arange(1, width_block_num - 2, device=device) * block_size
#     )  # shape: (n_w,)

#     # Create a grid for the block-local offsets (j,i) in [0, block_size)
#     off_h, off_w = torch.meshgrid(
#         torch.arange(block_size, device=device),
#         torch.arange(block_size, device=device),
#         indexing="ij",
#     )
#     # off_h, off_w: shape (block_size, block_size)

#     # For each valid block, compute the grid indices for the center "a" pixel.
#     # The resulting a_h_idx and a_w_idx will be of shape: (n_h, n_w, block_size, block_size)
#     a_h_idx = height_blocks.view(n_h, 1, 1, 1) + off_h.view(
#         1, 1, block_size, block_size
#     )
#     a_w_idx = width_blocks.view(1, n_w, 1, 1) + off_w.view(1, 1, block_size, block_size)
#     # Broadcasting makes them (n_h, n_w, block_size, block_size)

#     # For neighbor pixels, adjust the base indices:
#     # Left neighbor (b): subtract block_size from the w index.
#     b_w_idx = (width_blocks - block_size).view(1, n_w, 1, 1) + off_w.view(
#         1, 1, block_size, block_size
#     )
#     # Right neighbor (c): add block_size to the w index.
#     c_w_idx = (width_blocks + block_size).view(1, n_w, 1, 1) + off_w.view(
#         1, 1, block_size, block_size
#     )
#     # Upper neighbor (d): subtract block_size from the h index.
#     d_h_idx = (height_blocks - block_size).view(n_h, 1, 1, 1) + off_h.view(
#         1, 1, block_size, block_size
#     )
#     # Lower neighbor (e): add block_size to the h index.
#     e_h_idx = (height_blocks + block_size).view(n_h, 1, 1, 1) + off_h.view(
#         1, 1, block_size, block_size
#     )

#     # Expand a_h_idx and a_w_idx to be used for batched indexing.
#     # New shape: (B, n_h, n_w, block_size, block_size)
#     a_h_idx = a_h_idx.unsqueeze(0).expand(B, n_h, n_w, block_size, block_size)
#     a_w_idx = a_w_idx.unsqueeze(0).expand(B, n_h, n_w, block_size, block_size)

#     # Similarly, expand the neighbor indices.
#     b_w_idx = (
#         b_w_idx.expand(n_h, n_w, block_size, block_size)
#         .unsqueeze(0)
#         .expand(B, n_h, n_w, block_size, block_size)
#     )
#     c_w_idx = (
#         c_w_idx.expand(n_h, n_w, block_size, block_size)
#         .unsqueeze(0)
#         .expand(B, n_h, n_w, block_size, block_size)
#     )
#     d_h_idx = (
#         d_h_idx.expand(n_h, n_w, block_size, block_size)
#         .unsqueeze(0)
#         .expand(B, n_h, n_w, block_size, block_size)
#     )
#     e_h_idx = (
#         e_h_idx.expand(n_h, n_w, block_size, block_size)
#         .unsqueeze(0)
#         .expand(B, n_h, n_w, block_size, block_size)
#     )

#     # Create a batch index for advanced indexing.
#     batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1, 1)

#     # Extract the patches from dct_img for each neighbor.
#     # a: center pixels.
#     a = dct_img[batch_idx, a_h_idx, a_w_idx]
#     # b: left neighbor (same rows as a, but shifted left)
#     b = dct_img[batch_idx, a_h_idx, b_w_idx]
#     # c: right neighbor (same rows as a, but shifted right)
#     c = dct_img[batch_idx, a_h_idx, c_w_idx]
#     # d: upper neighbor (shifted up, same columns as a)
#     d = dct_img[batch_idx, d_h_idx, a_w_idx]
#     # e: lower neighbor (shifted down, same columns as a)
#     e = dct_img[batch_idx, e_h_idx, a_w_idx]

#     # Compute the gradient magnitude V for each block and each pixel in the block.
#     V = torch.sqrt((b + c - 2 * a) ** 2 + (d + e - 2 * a) ** 2)

#     # Sum over the block grid dimensions (n_h and n_w) and average by the given denominator.
#     # V has shape (B, n_h, n_w, block_size, block_size); summing dims 1 and 2 yields (B, block_size, block_size).
#     V_sum = V.sum(dim=(1, 2))
#     V_average = V_sum / denom

#     return V_average


def calc_v_torch(
    dct_img, height_block_num, width_block_num, block_size=DEFAULT_BLOCK_SIZE
):
    dct_img
    h_block_num = height_block_num
    w_block_num = width_block_num
    block_size
    # Number of blocks (note: h_offsets and w_offsets are built using 1 to h_block_num-2 (exclusive),
    # which gives h_block_num-3 elements, same for width)
    num_h = h_block_num - 3
    num_w = w_block_num - 3

    # Use the device of the input tensor for all created tensors
    device = dct_img.device

    # Compute the starting offset for each block.
    # Each offset is: block_size + (block_index * block_size), where block_index goes from 1 to h_block_num-2 (exclusive)
    h_offsets = (
        block_size + torch.arange(1, h_block_num - 2, device=device) * block_size
    )  # shape: (num_h,)
    w_offsets = (
        block_size + torch.arange(1, w_block_num - 2, device=device) * block_size
    )  # shape: (num_w,)

    # Create 4D index arrays for the row (r) and column (c) coordinates.
    # r will have shape (num_h, 1, block_size, 1)
    # c will have shape (1, num_w, 1, block_size)
    # They broadcast to form full index arrays of shape (num_h, num_w, block_size, block_size)
    r = (
        h_offsets[:, None, None, None]
        + torch.arange(block_size, device=device)[None, None, :, None]
    )
    c = (
        w_offsets[None, :, None, None]
        + torch.arange(block_size, device=device)[None, None, None, :]
    )

    # Ensure indices are of integer type (long)
    r = r.to(dtype=torch.int)
    c = c.to(dtype=torch.int)
    dct_img = dct_img.squeeze(0).squeeze(0)

    # Extract the central value (a) and its four neighbors:
    # left (b_val), right (c_val), top (d_val), and bottom (e_val).
    a = dct_img[..., r, c]
    b_val = dct_img[..., r, c - block_size]
    c_val = dct_img[..., r, c + block_size]
    d_val = dct_img[..., r - block_size, c]
    e_val = dct_img[..., r + block_size, c]

    # Compute V for each block and each pixel in the block.
    V = torch.sqrt((b_val + c_val - 2 * a) ** 2 + (d_val + e_val - 2 * a) ** 2)

    # Average V over all blocks (i.e. over the first two dimensions)
    V_average = V.sum(dim=(0, 1)) / ((h_block_num - 2) * (w_block_num - 2))

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
    # batch_size, channel_dim, h, w = gray_imgs.shape
    # assert channel_dim == 1, "Input image must be grayscale."
    if (
        gray_imgs.shape[-2] < height_block_num * block_size
        or gray_imgs.shape[-1] < width_block_num * block_size
    ):
        raise ValueError("Invalid image dimensions.")

    # Divide the image into blocks of shape (height_block_num, width_block_num, block_size, block_size).
    print(gray_imgs.shape)
    blocks = gray_imgs.unfold(
        1,
        block_size,
        block_size,
    ).unfold(
        0,
        block_size,
        block_size,
    )
    # print("Blocks shape", blocks.shape)
    blocks = blocks.contiguous().view(-1, block_size, block_size)
    # print("Blocks shape", blocks.shape)
    # Apply the batched DCT transform to all blocks at once.
    # blocks_no_channel = blocks.squeeze(1)
    # input(blocks_no_channel.dtype)
    dct_blocks_flat: torch.Tensor = dct_2d(blocks, norm="ortho")

    dct_blocks = dct_blocks_flat.view(
        height_block_num, width_block_num, block_size, block_size
    )
    dct_blocks = dct_blocks.permute(0, 2, 1, 3).contiguous()
    dct_blocks = dct_blocks.view(
        height_block_num * block_size,
        width_block_num * block_size,
    )
    return dct_blocks


def caculate_image_blockiness(
    gray_images: torch.Tensor,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    # gray_images -> h, w
    gray_images = gray_images
    # gray_images = rgb_to_grayscale(images)
    height, width = gray_images.shape[-2:]
    cal_height, cal_width, height_margin, width_margin = calc_margin(
        height=height, width=width
    )

    height_block_num, width_block_num = (
        cal_height // block_size,
        cal_width // block_size,
    )
    # TODO: bring back
    gray_tensor_cut = gray_images[:cal_height, :cal_width]
    # gray_tensor_cut = gray_images

    # gray_offset = gray_images[..., offset:, offset:]
    # gray_offset = gray_offset[..., :cal_height, :cal_width]
    gray_offset = torch.zeros_like(gray_images)
    gray_offset[..., :-4, :-4] = gray_images[..., 4:, 4:]
    gray_offset = gray_offset[:cal_height, :cal_width]

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

    d = torch.abs((v_offset_average - v_average)) / v_average
    d_sum = torch.sum(d)
    return float(d_sum)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("highest")
    torch.set_printoptions(precision=8)

    # img_size = block_size * num_blocks
    # dct_img = torch.rand(2, block_size * num_blocks, block_size * num_blocks)
    # dct_img_npy = dct_img[0].numpy()
    for i in ["", 80, 60]:
        img = torchvision.io.read_image(f"unsplash{i}.jpg")
        img_gray = rgb_to_grayscale(img).squeeze()
        # img = torch.rand(2, 3, 80, 50) * 25
        img_npy = img_gray.numpy()
        # print(img_npy.shape)
        tb = caculate_image_blockiness(img_gray)

        nb = process_image(img_npy, DCT())
        print()
        print("torch:", tb)
        print("npy:", nb)
        print()
