# coding: utf-8

import numpy as np
import torch
import torch.fft

BLOCK_SIZE = 8  # Define BLOCK_SIZE explicitly


class DCT:
    """
    Discrete Cosine Transform (DCT) class.
    """

    def __init__(self, N: int = BLOCK_SIZE) -> None:
        assert N > 0, "Block size N must be positive."
        self.N: int = N
        self.phi_1d: np.ndarray = np.array([self._phi(i) for i in range(self.N)])
        self.phi_2d: np.ndarray = np.zeros((N, N, N, N))

        for i in range(N):
            for j in range(N):
                phi_i, phi_j = np.meshgrid(self.phi_1d[i], self.phi_1d[j])
                self.phi_2d[i, j] = phi_i * phi_j

    def dct2(self, data: np.ndarray) -> np.ndarray:
        """
        is the same as scipy dct2
        Perform a 2D Discrete Cosine Transform on the input data.
        """
        assert data.shape == (self.N, self.N), (
            f"Input data must have shape ({self.N}, {self.N})."
        )

        reshaped_data: np.ndarray = data.reshape(self.N * self.N)
        reshaped_phi_2d: np.ndarray = self.phi_2d.reshape(
            self.N * self.N, self.N * self.N
        )
        dct_result: np.ndarray = np.dot(reshaped_phi_2d, reshaped_data)
        return dct_result.reshape(self.N, self.N)

    def _phi(self, k: int) -> np.ndarray:
        """
        Compute the basis function for DCT.
        """
        if k == 0:
            return np.ones(self.N) / np.sqrt(self.N)
        return np.sqrt(2.0 / self.N) * np.cos(
            (k * np.pi / (2 * self.N)) * (np.arange(self.N) * 2 + 1)
        )


def calc_margin(height: int, width: int) -> tuple[int, int, int, int]:
    """
    Calculate margins for DCT processing.
    """
    h_margin: int = height % BLOCK_SIZE
    w_margin: int = width % BLOCK_SIZE
    cal_height: int = height - (h_margin if h_margin >= 4 else h_margin + BLOCK_SIZE)
    cal_width: int = width - (w_margin if w_margin >= 4 else w_margin + BLOCK_SIZE)
    h_margin = (h_margin + BLOCK_SIZE) if h_margin < 4 else h_margin
    w_margin = (w_margin + BLOCK_SIZE) if w_margin < 4 else w_margin
    return cal_height, cal_width, h_margin, w_margin


def dct1_rfft_impl(x):
    return torch.view_as_real(torch.fft.rfft(x, dim=1))


def dct_fft_impl(v):
    return torch.view_as_real(torch.fft.fft(v, dim=1))


def idct_irfft_impl(V):
    return torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)


def dct1(x):
    """
    Discrete Cosine Transform, Type I

    :param x: the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    x = torch.cat([x, x.flip([1])[:, 1:-1]], dim=1)

    return dct1_rfft_impl(x)[:, :, 0].view(*x_shape)


def idct1(X):
    """
    The inverse of DCT-I, which is just a scaled DCT-I

    Our definition if idct1 is such that idct1(dct1(x)) == x

    :param X: the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    """
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))


def dct(x, norm=None):
    """
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
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

    if norm == "ortho":
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2

    V = 2 * V.view(*x_shape)

    return V


def idct(X, norm=None):
    """
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct(dct(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension
    """

    x_shape = X.shape
    N = x_shape[-1]

    X_v = X.contiguous().view(-1, x_shape[-1]) / 2

    if norm == "ortho":
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = (
        torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :]
        * np.pi
        / (2 * N)
    )
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

    v = idct_irfft_impl(V)
    x = v.new_zeros(v.shape)
    x[:, ::2] += v[:, : N - (N // 2)]
    x[:, 1::2] += v.flip([1])[:, : N // 2]

    return x.view(*x_shape)


def dct_2d(x, norm=None):
    """
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


def idct_2d(X, norm=None):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III

    Our definition of idct is that idct_2d(dct_2d(x)) == x

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param X: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct(X, norm=norm)
    x2 = idct(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)


def dct_3d(x, norm=None):
    """
    3-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)

    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

    :param x: the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 3 dimensions
    """
    X1 = dct(x, norm=norm)
    X2 = dct(X1.transpose(-1, -2), norm=norm)
    X3 = dct(X2.transpose(-1, -3), norm=norm)
    return X3.transpose(-1, -3).transpose(-1, -2)


def calc_V(dct_img: np.ndarray, h_block_num: int, w_block_num: int) -> np.ndarray:
    """
    Compute the average gradient magnitude V for DCT blocks.
    """
    V_average: np.ndarray = np.zeros((BLOCK_SIZE, BLOCK_SIZE))

    for j in range(BLOCK_SIZE):
        for i in range(BLOCK_SIZE):
            V_sum: float = 0.0
            for h_block in range(1, h_block_num - 2):
                for w_block in range(1, w_block_num - 2):
                    w_idx, h_idx = (
                        BLOCK_SIZE + w_block * BLOCK_SIZE + i,
                        BLOCK_SIZE + h_block * BLOCK_SIZE + j,
                    )
                    a = dct_img[h_idx, w_idx]
                    b = dct_img[h_idx, w_idx - BLOCK_SIZE]
                    c = dct_img[h_idx, w_idx + BLOCK_SIZE]
                    d = dct_img[h_idx - BLOCK_SIZE, w_idx]
                    e = dct_img[h_idx + BLOCK_SIZE, w_idx]
                    V = np.sqrt((b + c - 2 * a) ** 2 + (d + e - 2 * a) ** 2)
                    V_sum += V
            V_average[j, i] = V_sum / ((h_block_num - 2) * (w_block_num - 2))

    return V_average


def calc_V_torch(dct_img, h_block_num, w_block_num, BLOCK_SIZE):
    device = "cpu"
    # The original loops run for h_block in range(1, h_block_num-2) and
    # w_block in range(1, w_block_num-2); note that the number of iterations
    # is (h_block_num-3) and (w_block_num-3), but the average is divided by
    # (h_block_num-2)*(w_block_num-2). We'll follow the same convention.
    n_h = h_block_num - 3  # number of blocks vertically
    n_w = w_block_num - 3  # number of blocks horizontally
    denom = (h_block_num - 2) * (w_block_num - 2)

    # Compute the “base” index for each inner block.
    # For each block index in the loop, the row index used for 'a' is:
    #    h_idx = BLOCK_SIZE + h_block * BLOCK_SIZE + j
    # so we precompute:
    h_blocks = (
        BLOCK_SIZE + torch.arange(1, h_block_num - 2, device=device) * BLOCK_SIZE
    )  # shape: (n_h,)
    w_blocks = (
        BLOCK_SIZE + torch.arange(1, w_block_num - 2, device=device) * BLOCK_SIZE
    )  # shape: (n_w,)

    # Create a meshgrid for the block-local offsets (j,i) in [0, BLOCK_SIZE)
    off_h, off_w = torch.meshgrid(
        torch.arange(BLOCK_SIZE, device=device),
        torch.arange(BLOCK_SIZE, device=device),
        indexing="ij",
    )
    # off_h, off_w: shape (BLOCK_SIZE, BLOCK_SIZE)

    # For each inner block, compute the grid indices for the center "a" pixel.
    # The full indices will have shape (n_h, n_w, BLOCK_SIZE, BLOCK_SIZE).
    a_h_idx = h_blocks.view(n_h, 1, 1, 1) + off_h.view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
    a_w_idx = w_blocks.view(1, n_w, 1, 1) + off_w.view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
    a_h_idx = a_h_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    a_w_idx = a_w_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    a = dct_img[a_h_idx, a_w_idx]

    # For the neighbors, adjust the base indices:
    # Left neighbor: subtract BLOCK_SIZE from the w index.
    b_w_idx = (w_blocks - BLOCK_SIZE).view(1, n_w, 1, 1) + off_w.view(
        1, 1, BLOCK_SIZE, BLOCK_SIZE
    )
    b_w_idx = b_w_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    b = dct_img[a_h_idx, b_w_idx]

    # Right neighbor: add BLOCK_SIZE to the w index.
    c_w_idx = (w_blocks + BLOCK_SIZE).view(1, n_w, 1, 1) + off_w.view(
        1, 1, BLOCK_SIZE, BLOCK_SIZE
    )
    c_w_idx = c_w_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    c = dct_img[a_h_idx, c_w_idx]

    # Upper neighbor: subtract BLOCK_SIZE from the h index.
    d_h_idx = (h_blocks - BLOCK_SIZE).view(n_h, 1, 1, 1) + off_h.view(
        1, 1, BLOCK_SIZE, BLOCK_SIZE
    )
    d_h_idx = d_h_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    d = dct_img[d_h_idx, a_w_idx]

    # Lower neighbor: add BLOCK_SIZE to the h index.
    e_h_idx = (h_blocks + BLOCK_SIZE).view(n_h, 1, 1, 1) + off_h.view(
        1, 1, BLOCK_SIZE, BLOCK_SIZE
    )
    e_h_idx = e_h_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    e = dct_img[e_h_idx, a_w_idx]

    # Compute V for each block and each (j,i)
    V = torch.sqrt((b + c - 2 * a) ** 2 + (d + e - 2 * a) ** 2)
    # Sum V over all inner blocks (dimensions 0 and 1)

    V_sum = V.sum(dim=(0, 1))
    V_average = V_sum / denom
    return V_average


def calc_V_torch2(
    dct_img: torch.Tensor, h_block_num: int, w_block_num: int
) -> torch.Tensor:
    """
    Compute the average gradient magnitude V for DCT blocks using PyTorch tensors.
    Avoids for loops by using vectorized operations.

    Args:
        dct_img: Input DCT image tensor of shape [H, W]
        h_block_num: Number of blocks in height dimension
        w_block_num: Number of blocks in width dimension

    Returns:
        V_average: Average gradient magnitude tensor of shape [BLOCK_SIZE, BLOCK_SIZE]
    """
    # Extract BLOCK_SIZE from input dimensions
    # For a 24x24 input with h_block_num=w_block_num=3, BLOCK_SIZE would be 8
    BLOCK_SIZE = dct_img.shape[0] // h_block_num

    # Create sliding window views for each position (center, left, right, top, bottom)
    # We'll only consider the inner blocks (excluding edge blocks)
    inner_h_start = BLOCK_SIZE
    inner_h_end = BLOCK_SIZE * (h_block_num - 1)
    inner_w_start = BLOCK_SIZE
    inner_w_end = BLOCK_SIZE * (w_block_num - 1)

    # Extract the regions we need for computation
    inner_region = dct_img[inner_h_start:inner_h_end, inner_w_start:inner_w_end]
    left_region = dct_img[
        inner_h_start:inner_h_end, inner_w_start - BLOCK_SIZE : inner_w_end - BLOCK_SIZE
    ]
    right_region = dct_img[
        inner_h_start:inner_h_end, inner_w_start + BLOCK_SIZE : inner_w_end + BLOCK_SIZE
    ]
    top_region = dct_img[
        inner_h_start - BLOCK_SIZE : inner_h_end - BLOCK_SIZE, inner_w_start:inner_w_end
    ]
    bottom_region = dct_img[
        inner_h_start + BLOCK_SIZE : inner_h_end + BLOCK_SIZE, inner_w_start:inner_w_end
    ]

    # Calculate horizontal and vertical gradients
    horizontal_grad = (left_region + right_region - 2 * inner_region).pow(2)
    vertical_grad = (top_region + bottom_region - 2 * inner_region).pow(2)

    # Calculate V for all positions simultaneously
    V = torch.sqrt(horizontal_grad + vertical_grad)

    # Reshape V to group by block positions
    V_reshaped = V.reshape(h_block_num - 2, BLOCK_SIZE, w_block_num - 2, BLOCK_SIZE)

    # Permute to get dimensions in order [BLOCK_SIZE, BLOCK_SIZE, h_blocks, w_blocks]
    V_permuted = V_reshaped.permute(1, 3, 0, 2)

    # Average over all blocks
    V_average = V_permuted.mean(dim=(2, 3))

    return V_average


def calc_V_torch_b(dct_img, h_block_num, w_block_num, BLOCK_SIZE):
    """
    Compute the average gradient magnitude V for each image in a batched DCT image tensor.

    Parameters:
      dct_img: Tensor of shape (B, H, W) where B is the batch size.
      h_block_num: Number of blocks along the height.
      w_block_num: Number of blocks along the width.
      BLOCK_SIZE: Size of each block.

    Returns:
      V_average: Tensor of shape (B, BLOCK_SIZE, BLOCK_SIZE) with the averaged V for each image.
    """
    device = dct_img.device
    # The loops iterate for h_block in range(1, h_block_num-2) and w_block in range(1, w_block_num-2),
    # so there are (h_block_num - 3) and (w_block_num - 3) inner blocks.
    # The average is computed with denominator (h_block_num - 2)*(w_block_num - 2).
    n_h = h_block_num - 3  # number of valid blocks vertically
    n_w = w_block_num - 3  # number of valid blocks horizontally
    denom = (h_block_num - 2) * (w_block_num - 2)

    B = dct_img.shape[0]  # batch size

    # Compute the "base" index for each inner block along each dimension.
    # For a given block, the center pixel (a) has indices:
    #   row: BLOCK_SIZE + h_block * BLOCK_SIZE + j
    #   col: BLOCK_SIZE + w_block * BLOCK_SIZE + i
    h_blocks = (
        BLOCK_SIZE + torch.arange(1, h_block_num - 2, device=device) * BLOCK_SIZE
    )  # shape: (n_h,)
    w_blocks = (
        BLOCK_SIZE + torch.arange(1, w_block_num - 2, device=device) * BLOCK_SIZE
    )  # shape: (n_w,)

    # Create a grid for the block-local offsets (j,i) in [0, BLOCK_SIZE)
    off_h, off_w = torch.meshgrid(
        torch.arange(BLOCK_SIZE, device=device),
        torch.arange(BLOCK_SIZE, device=device),
        indexing="ij",
    )
    # off_h, off_w: shape (BLOCK_SIZE, BLOCK_SIZE)

    # For each valid block, compute the grid indices for the center "a" pixel.
    # The resulting a_h_idx and a_w_idx will be of shape: (n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    a_h_idx = h_blocks.view(n_h, 1, 1, 1) + off_h.view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
    a_w_idx = w_blocks.view(1, n_w, 1, 1) + off_w.view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
    # Broadcasting makes them (n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)

    # For neighbor pixels, adjust the base indices:
    # Left neighbor (b): subtract BLOCK_SIZE from the w index.
    b_w_idx = (w_blocks - BLOCK_SIZE).view(1, n_w, 1, 1) + off_w.view(
        1, 1, BLOCK_SIZE, BLOCK_SIZE
    )
    # Right neighbor (c): add BLOCK_SIZE to the w index.
    c_w_idx = (w_blocks + BLOCK_SIZE).view(1, n_w, 1, 1) + off_w.view(
        1, 1, BLOCK_SIZE, BLOCK_SIZE
    )
    # Upper neighbor (d): subtract BLOCK_SIZE from the h index.
    d_h_idx = (h_blocks - BLOCK_SIZE).view(n_h, 1, 1, 1) + off_h.view(
        1, 1, BLOCK_SIZE, BLOCK_SIZE
    )
    # Lower neighbor (e): add BLOCK_SIZE to the h index.
    e_h_idx = (h_blocks + BLOCK_SIZE).view(n_h, 1, 1, 1) + off_h.view(
        1, 1, BLOCK_SIZE, BLOCK_SIZE
    )

    # Expand a_h_idx and a_w_idx to be used for batched indexing.
    # New shape: (B, n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    a_h_idx = a_h_idx.unsqueeze(0).expand(B, n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    a_w_idx = a_w_idx.unsqueeze(0).expand(B, n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)

    # Similarly, expand the neighbor indices.
    b_w_idx = (
        b_w_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
        .unsqueeze(0)
        .expand(B, n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    )
    c_w_idx = (
        c_w_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
        .unsqueeze(0)
        .expand(B, n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    )
    d_h_idx = (
        d_h_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
        .unsqueeze(0)
        .expand(B, n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    )
    e_h_idx = (
        e_h_idx.expand(n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
        .unsqueeze(0)
        .expand(B, n_h, n_w, BLOCK_SIZE, BLOCK_SIZE)
    )

    # Create a batch index for advanced indexing.
    batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1, 1)

    # Extract the patches from dct_img for each neighbor.
    # a: center pixels.
    a = dct_img[batch_idx, a_h_idx, a_w_idx]
    # b: left neighbor (same rows as a, but shifted left)
    b = dct_img[batch_idx, a_h_idx, b_w_idx]
    # c: right neighbor (same rows as a, but shifted right)
    c = dct_img[batch_idx, a_h_idx, c_w_idx]
    # d: upper neighbor (shifted up, same columns as a)
    d = dct_img[batch_idx, d_h_idx, a_w_idx]
    # e: lower neighbor (shifted down, same columns as a)
    e = dct_img[batch_idx, e_h_idx, a_w_idx]

    # Compute the gradient magnitude V for each block and each pixel in the block.
    V = torch.sqrt((b + c - 2 * a) ** 2 + (d + e - 2 * a) ** 2)

    # Sum over the block grid dimensions (n_h and n_w) and average by the given denominator.
    # V has shape (B, n_h, n_w, BLOCK_SIZE, BLOCK_SIZE); summing dims 1 and 2 yields (B, BLOCK_SIZE, BLOCK_SIZE).
    V_sum = V.sum(dim=(1, 2))
    V_average = V_sum / denom

    return V_average


def calc_V_torch3(
    dct_img: torch.Tensor, h_block_num: int, w_block_num: int, block_size: int = 8
) -> torch.Tensor:
    """
    Compute the average gradient magnitude V for DCT blocks in a fully vectorized manner.

    dct_img: 2D tensor representing the DCT image.
    h_block_num, w_block_num: number of blocks along height and width in the full dct_img.
    block_size: size of each block (default 8, so output will be 8x8).

    Returns:
      V_average: tensor of shape (block_size, block_size) containing the averaged V values.
    """
    # Create valid block indices (matching: for h_block in range(1, h_block_num-2))
    h_indices = torch.arange(
        1, h_block_num - 2, device=dct_img.device
    )  # shape (num_h,)
    w_indices = torch.arange(
        1, w_block_num - 2, device=dct_img.device
    )  # shape (num_w,)
    H_idx, W_idx = torch.meshgrid(h_indices, w_indices, indexing="ij")
    H_idx = H_idx.flatten()  # (n_blocks,)
    W_idx = W_idx.flatten()  # (n_blocks,)

    # Local coordinate grid within a block (shape: (block_size,))
    local_y = torch.arange(block_size, device=dct_img.device)
    local_x = torch.arange(block_size, device=dct_img.device)

    # For a given block at (h_block, w_block), the pixel at offset (j, i) is at:
    #   a: dct_img[block_size + h_block*block_size + j, block_size + w_block*block_size + i]
    #   b: dct_img[block_size + h_block*block_size + j,             w_block*block_size + i]
    #   c: dct_img[block_size + h_block*block_size + j, block_size + (w_block+1)*block_size + i]
    #   d: dct_img[             h_block*block_size + j, block_size + w_block*block_size + i]
    #   e: dct_img[block_size + (h_block+1)*block_size + j, block_size + w_block*block_size + i]

    # Compute starting indices for each patch
    # For a:
    start_a_r = block_size + H_idx * block_size  # starting row for each valid block
    start_a_c = block_size + W_idx * block_size  # starting col for each valid block

    # For b: shift columns left by block_size.
    start_b_r = start_a_r
    start_b_c = W_idx * block_size

    # For c: shift columns right by block_size.
    start_c_r = start_a_r
    start_c_c = block_size + (W_idx + 1) * block_size

    # For d: shift rows upward by block_size.
    start_d_r = H_idx * block_size
    start_d_c = start_a_c

    # For e: shift rows downward by block_size.
    start_e_r = block_size + (H_idx + 1) * block_size
    start_e_c = start_a_c

    # Function to extract an (n_blocks x block_size x block_size) patch tensor given starting rows/cols.
    def extract_patch(start_r, start_c):
        # start_r and start_c are (n_blocks,)
        # We add a local grid to each starting index to get the patch indices.
        # The resulting indexing adds dimensions:
        #   start_r[:, None, None] has shape (n_blocks, 1, 1)
        #   local_y[None, :, None] has shape (1, block_size, 1)
        #   local_x[None, None, :] has shape (1, 1, block_size)
        return dct_img[
            start_r[:, None, None] + local_y[None, :, None],
            start_c[:, None, None] + local_x[None, None, :],
        ]

    # Extract all patches
    a = extract_patch(start_a_r, start_a_c)
    b = extract_patch(start_b_r, start_b_c)
    c = extract_patch(start_c_r, start_c_c)
    d = extract_patch(start_d_r, start_d_c)
    e = extract_patch(start_e_r, start_e_c)

    # Compute gradient magnitude V for each patch:
    # V = sqrt((b + c - 2*a)^2 + (d + e - 2*a)^2)
    V = torch.sqrt((b + c - 2 * a) ** 2 + (d + e - 2 * a) ** 2)

    # Average over all valid blocks (the first dimension)
    V_average = V.mean(dim=0)  # shape: (block_size, block_size)

    return V_average


def calc_DCT_torch(
    img: torch.Tensor, h_block_num: int, w_block_num: int
) -> torch.Tensor:
    """Compute the DCT of an image block-wise using batched processing.

    This function divides the image into non-overlapping blocks of size BLOCK_SIZE x BLOCK_SIZE,
    applies a batched DCT transform (using dct2, which accepts batched input), and reconstructs the
    DCT image from the transformed blocks.

    Args:
        img: Input image tensor with shape (H, W).
        dct: A DCT object with a method dct2 that accepts batched input.
        h_block_num: Number of blocks along the height.
        w_block_num: Number of blocks along the width.

    Returns:
        A tensor containing the DCT coefficients of the image blocks, arranged in the original block layout.
    """
    # Validate that the image dimensions are sufficient for the required blocks.
    if (
        img.shape[0] < h_block_num * BLOCK_SIZE
        or img.shape[1] < w_block_num * BLOCK_SIZE
    ):
        raise ValueError("Invalid image dimensions.")

    # Divide the image into blocks of shape (h_block_num, w_block_num, BLOCK_SIZE, BLOCK_SIZE).
    blocks = img.unfold(0, BLOCK_SIZE, BLOCK_SIZE).unfold(1, BLOCK_SIZE, BLOCK_SIZE)
    blocks = blocks.contiguous().view(-1, BLOCK_SIZE, BLOCK_SIZE)

    print(blocks[-1, :, :])
    # Apply the batched DCT transform to all blocks at once.
    dct_blocks = dct_2d(blocks, norm="ortho")

    # Reshape back to (h_block_num, w_block_num, BLOCK_SIZE, BLOCK_SIZE).
    dct_blocks = dct_blocks.view(h_block_num, w_block_num, BLOCK_SIZE, BLOCK_SIZE)

    # Rearrange blocks to reconstruct the full DCT image.
    dct_img = (
        dct_blocks.permute(0, 2, 1, 3)
        .contiguous()
        .view(h_block_num * BLOCK_SIZE, w_block_num * BLOCK_SIZE)
    )
    return dct_img


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    block_size = 8
    num_blocks = 10
    # img_size = block_size * num_blocks
    dct_img = torch.rand(2, block_size * num_blocks, block_size * num_blocks)
    dct_img_npy = dct_img[0].numpy()
    # print(img)
    rv = calc_V_torch(dct_img[1], num_blocks, num_blocks, block_size)
    rvb = calc_V_torch_b(dct_img, num_blocks, num_blocks, block_size)
    # nv = calc_V(dct_img_npy, num_blocks, num_blocks)
    print(rv.shape)
    print(rv)
    print(rvb[1])
    # print(torch.abs(rv - nv))
    # print(nv.shape)
