import os

import torch
import numpy as np
from PIL import Image
import glob


def get_data_batched(data_path, num_pixels, num_seq, max_seq_len):
    data = np.zeros([num_seq, max_seq_len, num_pixels, num_pixels])

    last_index = 0
    seq_num = 0
    images = glob.glob(os.path.join(data_path, "*.jpg"))
    for image_name in images:
        split = image_name.split('_')
        index = int(split[2]) - 1

        image = Image.open(image_name)  # Open an Image via PILLOW
        image = image.convert('L')  # Grayscale transformation
        image_vec = np.array(image)

        if index < last_index:
            seq_num += 1

        for i in range(0, num_pixels):
            for j in range(0, num_pixels):
                data[seq_num][index][i][j] = image_vec[i][j]
        last_index = index

    data = torch.from_numpy(data)
    data.view(num_seq, max_seq_len, num_pixels**2)

    return data


def get_data_list(data_path, num_pixels, max_seq_len):
    data = []
    data_line = np.zeros([max_seq_len, num_pixels, num_pixels])
    last_index = 0
    seq_num = 0
    images = glob.glob(os.path.join(data_path, "*.jpg"))
    for image_name in images:
        split = image_name.split('_')
        index = int(split[2]) - 1

        image = Image.open(image_name)  # Open an Image via PILLOW
        image = image.convert('L')  # Grayscale transformation
        image_vec = np.array(image)

        if index < last_index:
            seq_num += 1
            seq_length = last_index + 1
            data.append(data_line[:seq_length, :, :].view(seq_length, num_pixels ** 2))

        for i in range(0, num_pixels):
            for j in range(0, num_pixels):
                data[seq_num][index][i][j] = image_vec[i][j]
        last_index = index

    return data
