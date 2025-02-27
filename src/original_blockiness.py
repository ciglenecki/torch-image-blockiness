"""
Author: Go Ohtani
Author's repo: https://github.com/gohtanii/DiverSeg-dataset
Licence: MIT
"""
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# This implementation is based on the following paper:
#
# Bhardwaj, Dinesh, and Vinod Pankajakshan. "A JPEG blocking artifact detector for image forensics."
# Signal Processing: Image Communication 68 (2018): 155-161.
#
# If you use this code or find it helpful in your research, please consider citing the original paper.
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# coding: utf-8
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

BLOCK_SIZE = 8


class DCT:
    """
    Discrete Cosine Transform (DCT) class.

    Original code reference:
    https://gist.github.com/TonyMooori/661a2da7cbb389f0a99c
    """

    def __init__(self, N=BLOCK_SIZE):
        self.N = N
        self.phi_1d = np.array([self._phi(i) for i in range(self.N)])
        self.phi_2d = np.zeros((N, N, N, N))
        for i in range(N):
            for j in range(N):
                phi_i, phi_j = np.meshgrid(self.phi_1d[i], self.phi_1d[j])
                self.phi_2d[i, j] = phi_i * phi_j

    def dct2(self, data):
        reshaped_data = data.reshape(self.N * self.N)
        reshaped_phi_2d = self.phi_2d.reshape(self.N * self.N, self.N * self.N)
        dct_result = np.dot(reshaped_phi_2d, reshaped_data)
        return dct_result.reshape(self.N, self.N)

    def _phi(self, k):
        if k == 0:
            return np.ones(self.N) / np.sqrt(self.N)
        else:
            return np.sqrt(2.0 / self.N) * np.cos(
                (k * np.pi / (2 * self.N)) * (np.arange(self.N) * 2 + 1)
            )


def calc_margin(height, width, block_size=BLOCK_SIZE):
    h_margin = height % block_size
    w_margin = width % block_size
    cal_height = height - (h_margin if h_margin >= 4 else h_margin + block_size)
    cal_width = width - (w_margin if w_margin >= 4 else w_margin + block_size)
    h_margin = (h_margin + block_size) if h_margin < 4 else h_margin
    w_margin = (w_margin + block_size) if w_margin < 4 else w_margin
    return cal_height, cal_width, h_margin, w_margin


def calc_DCT(img, dct: DCT, h_block_num, w_block_num):
    block_size = dct.N
    dct_img = np.zeros((h_block_num * block_size, w_block_num * block_size))
    for h_block in range(h_block_num):
        for w_block in range(w_block_num):
            dct_img[
                h_block * block_size : (h_block + 1) * block_size,
                w_block * block_size : (w_block + 1) * block_size,
            ] = dct.dct2(
                img[
                    h_block * block_size : (h_block + 1) * block_size,
                    w_block * block_size : (w_block + 1) * block_size,
                ]
            )
    return dct_img


def calc_v_npy(dct_img, h_block_num, w_block_num):
    # Number of blocks over which we'll average (the original loop iterates over h_block in range(1, h_block_num-2)
    # and w_block in range(1, w_block_num-2)
    # note that there are (h_block_num-3) x (w_block_num-3) such blocks)
    # num_h = h_block_num - 3
    # num_w = w_block_num - 3

    # Compute the starting offset for each block.
    # For each block, the pixel coordinate is computed as:
    #   row = BLOCK_SIZE + (block_index * BLOCK_SIZE) + j, with block_index from 1 to (h_block_num-2)-1.
    #   col = BLOCK_SIZE + (block_index * BLOCK_SIZE) + i, with block_index from 1 to (w_block_num-2)-1.
    h_offsets = (
        BLOCK_SIZE + np.arange(1, h_block_num - 2) * BLOCK_SIZE
    )  # shape: (num_h,)
    w_offsets = (
        BLOCK_SIZE + np.arange(1, w_block_num - 2) * BLOCK_SIZE
    )  # shape: (num_w,)

    # Create 4D index arrays for the row and column coordinates.
    # For each block (over num_h and num_w) and for each pixel offset (j, i) in the BLOCK_SIZE x BLOCK_SIZE block.
    # r will have shape (num_h, 1, BLOCK_SIZE, 1) and c will have shape (1, num_w, 1, BLOCK_SIZE).
    # They will broadcast to produce a full index array of shape (num_h, num_w, BLOCK_SIZE, BLOCK_SIZE).
    r = h_offsets[:, None, None, None] + np.arange(BLOCK_SIZE)[None, None, :, None]
    c = w_offsets[None, :, None, None] + np.arange(BLOCK_SIZE)[None, None, None, :]

    # Extract the central value (a) and its four neighbors: left (b), right (c), top (d), and bottom (e).
    a = dct_img[r, c]
    b_val = dct_img[r, c - BLOCK_SIZE]
    c_val = dct_img[r, c + BLOCK_SIZE]
    d_val = dct_img[r - BLOCK_SIZE, c]
    e_val = dct_img[r + BLOCK_SIZE, c]

    # Compute V for each block and each pixel in the block.
    V = np.sqrt((b_val + c_val - 2 * a) ** 2 + (d_val + e_val - 2 * a) ** 2)

    # Average V over all blocks (i.e. over the first two dimensions).
    V_average = V.sum(axis=(0, 1)) / ((h_block_num - 2) * (w_block_num - 2))

    return V_average


def calc_V(dct_img, h_block_num, w_block_num):
    V_average = np.zeros((BLOCK_SIZE, BLOCK_SIZE))
    for j in range(BLOCK_SIZE):
        for i in range(BLOCK_SIZE):
            V_sum = 0
            for h_block in range(1, h_block_num - 2):
                for w_block in range(1, w_block_num - 2):
                    w_idx = BLOCK_SIZE + w_block * BLOCK_SIZE + i
                    h_idx = BLOCK_SIZE + h_block * BLOCK_SIZE + j
                    a = dct_img[h_idx, w_idx]
                    b = dct_img[h_idx, w_idx - BLOCK_SIZE]
                    c = dct_img[h_idx, w_idx + BLOCK_SIZE]
                    d = dct_img[h_idx - BLOCK_SIZE, w_idx]
                    e = dct_img[h_idx + BLOCK_SIZE, w_idx]
                    V = np.sqrt((b + c - 2 * a) ** 2 + (d + e - 2 * a) ** 2)
                    V_sum += V
            V_average[j, i] = V_sum / ((h_block_num - 2) * (w_block_num - 2))
    return V_average


def process_image(gray_img, dct):
    height, width = gray_img.shape
    cal_height, cal_width, _, _ = calc_margin(height, width)
    h_block_num, w_block_num = cal_height // BLOCK_SIZE, cal_width // BLOCK_SIZE

    dct_img = calc_DCT(gray_img, dct, h_block_num, w_block_num)
    dct_cropped_img = calc_DCT(gray_img[4:, 4:], dct, h_block_num, w_block_num)

    V_average = calc_V(dct_img, h_block_num, w_block_num)
    Vc_average = calc_V(dct_cropped_img, h_block_num, w_block_num)
    D = np.abs((Vc_average - V_average) / V_average)
    return np.sum(D)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect JPEG blocking artifacts in images and save them."
    )
    parser.add_argument(
        "--rootpath",
        type=str,
        required=True,
        help="Root directory containing the images.",
    )
    parser.add_argument(
        "--savefile", type=str, required=True, help="File to save the results."
    )
    parser.add_argument(
        "--suffix",
        default="png",
        type=str,
        help="File extension of the images to process.",
    )

    args = parser.parse_args()

    dct = DCT(N=BLOCK_SIZE)

    paths = sorted(list(Path(args.rootpath).rglob(f"*.{args.suffix}")))

    with open(args.savefile, mode="w") as f:
        for path in tqdm(paths):
            img = cv2.imread(os.path.join(args.rootpath, path))
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            D_value = process_image(gray_img, dct)
            f.write(os.path.join(args.rootpath, path) + "\t" + str(D_value) + "\n")
